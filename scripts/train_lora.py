"""
Simple LoRA fine-tuning for Stable Audio 3.

Three dataset modes (exactly one required):

  --data_dir    Raw audio + caption pairs. Each clip needs a matching .txt file:
    data_dir/
      clip1.wav   (or .flac, .mp3, .ogg)
      clip1.txt   ← text prompt for clip1
      clip2.wav
      clip2.txt
    Or pass --caption_sidecar alongside --data_dir to source captions from a
    caption_tools sidecar (T1/T2/T3) keyed on the audio's path relative to
    data_dir, instead of per-clip .txt files — see make_data_dir_caption_fn.

  --encoded_dir Pre-encoded latents from pre_encode_dataset.py. Captions are
                already embedded in the .json metadata — no .txt files needed:
    encoded_dir/
      000000000000.npy
      000000000000.json
      000000000001.npy
      000000000001.json

  --arc-data    ARC-Forcing rollout .npz dir (shared data contract, SAO task #46):
                one .npz per sample (context_latent/target_latent/mask/prompt/meta)
                + manifest.json. Trains with the model's own drifted context
                CLAMPED as inpaint conditioning, loss on the free region only.

Saves .safetensors LoRA checkpoints compatible with the inference model and run_gradio.py.

Usage:
  uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data --save_dir ./lora_out
  uv run python scripts/train_lora.py --model medium-base --encoded_dir ./latents_out --save_dir ./lora_out
  uv run python scripts/train_lora.py --model medium-base --data_dir ./my_data --steps 500 --rank 8
"""

# Disable HuggingFace progress bars BEFORE any imports
# This must be at the very top to take effect
import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import argparse
import itertools
import json
from pathlib import Path
import torch
# LUMI 8-GCD DDP + live-encode: 8 ranks x --num_workers DataLoader procs pass big decoded
# audio tensors through the node's /dev/shm, which exhausts -> "unable to allocate shared
# memory (shm) ... Resource temporarily unavailable (11)" (crashed fullft_bigset 20687866,
# 2026-08-06). 'file_system' shares via /tmp files (bound in the container) not /dev/shm.
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass
import pytorch_lightning as pl

from stable_audio_3.data.dataset import (
    ArcRolloutDataset,
    LatentDatasetConfig,
    LocalDatasetConfig,
    PreEncodedDataset,
    SampleDataset,
    collation_fn,
)
from safetensors.torch import load_file
from stable_audio_3.loading_utils import copy_state_dict
from stable_audio_3.model_configs import base_models
from stable_audio_3.factory import create_diffusion_cond_from_config
from stable_audio_3.models.lora.utils import load_lora_checkpoint
from stable_audio_3.training.diffusion import (
    DiffusionCondTrainingWrapper,
    DiffusionCondInpaintDemoCallback,
)


# --base_precision mode -> (base-weight load dtype, wrapper downcast target, Lightning
# trainer precision). Two families (C's GOA-node TASK A, 2026-07-24):
#   "bf16"/"fp16"        -> DOWNCAST the frozen base weights (memory saving; LoRA stays
#                           fp32) + matching autocast. "bf16" is the pre-existing default.
#   "bf16-mixed"/"fp16-mixed" -> base stays FP32 (fp32 master weights) + autocast compute.
# The wrapper's cast_base_to_precision already handles bf16 AND fp16 base casts, and
# Lightning's "16-mixed" precision supplies the fp16 GradScaler automatically — so this
# is pure mode-mapping, no new casting/scaler code.
PRECISION_MODES = {
    "fp32":       (torch.float32,  None,   "32-true"),
    "float32":    (torch.float32,  None,   "32-true"),
    "bf16":       (torch.bfloat16, "bf16", "bf16-mixed"),
    "bfloat16":   (torch.bfloat16, "bf16", "bf16-mixed"),
    "fp16":       (torch.float16,  "fp16", "16-mixed"),
    "float16":    (torch.float16,  "fp16", "16-mixed"),
    "bf16-mixed": (torch.float32,  None,   "bf16-mixed"),
    "fp16-mixed": (torch.float32,  None,   "16-mixed"),
}


def load_layer_update_weights(spec):
    """Parse --layer-update-weights (PATH or PATH#CURVE) -> {int layer: float mult} or None.
    A multi-curve file ({"curves": {name: {layer: mult}}}, e.g. C's lumi/layer_curves/curves.json)
    REQUIRES a #CURVE selector; a flat {layer: mult} file is used as-is."""
    if not spec:
        return None
    path, _, curve = spec.partition("#")
    data = json.load(open(path))
    if isinstance(data, dict) and "curves" in data:
        avail = list(data["curves"])
        if not curve:
            raise SystemExit(f"--layer-update-weights: {path} is multi-curve; use PATH#CURVE "
                             f"(available: {avail})")
        if curve not in data["curves"]:
            raise SystemExit(f"--layer-update-weights: curve '{curve}' not in {path} (available: {avail})")
        data = data["curves"][curve]
    elif curve and isinstance(data, dict) and curve in data:
        data = data[curve]
    return {int(k): float(v) for k, v in data.items()}


def load_model(model_name: str, device: torch.device,
               dtype: torch.dtype = torch.bfloat16):
    if model_name not in base_models:
        raise ValueError(
            f"LoRA training requires a base model. Got '{model_name}', valid: {list(base_models)}"
        )
    model_cfg = base_models[model_name]
    local_config, local_ckpt = model_cfg.resolve()
    with open(local_config) as f:
        model_config = json.load(f)
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(local_ckpt))
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.enable_grad = False
    return model, model_config


def compute_warmup_steps(warmup_steps, warmup_frac, dataset_len, batch_size,
                          accumulate_grad_batches, epochs, steps):
    """FusionOpt warmup_steps, resolved from either a raw count or a fraction of the
    run's total OPTIMIZER steps (--warmup-frac wins when given -- Kim direct 2026-08-14,
    "aim for 5% at least of the intended length": for a 20-epoch run that's exactly one
    epoch, independent of corpus size, since 5% of N*epochs is always epochs/20 of an
    epoch's worth... i.e. epochs/20 epochs -- trivially 1 epoch at the 20-epoch recipes
    we actually run). Total steps = (dataset_len // batch_size // accumulate_grad_batches)
    * epochs in --epochs mode (Lightning's max_epochs), else --steps directly (already
    optimizer-step units, matching Trainer's max_steps)."""
    if warmup_frac is None:
        return warmup_steps
    steps_per_epoch = dataset_len // batch_size // max(1, accumulate_grad_batches)
    total_steps = steps_per_epoch * epochs if epochs else steps
    return round(warmup_frac * total_steps)


def caption_metadata_fn(info, audio):
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text().strip()}


def make_data_dir_caption_fn(sidecar_path, probs, track_type_prob):
    """--data_dir sibling of the --encoded_dir make_caption_sampler wiring.

    LocalDataset (live-encode) has no per-clip .txt file for a sidecar-driven corpus
    (e.g. goa_archive's #90 big-set), so caption_metadata_fn's .txt lookup would
    __reject__ every clip. Reuses caption_tools.make_caption_sampler (don't duplicate
    its tier-sampling logic here — CONTINUITY owns that file) with a key_fn on the FULL
    relpath (dataset.py's LocalDataset sets info["relpath"] = os.path.relpath(audio_filename,
    root_path) — i.e. relative to --data_dir), matching the sidecar's own rel-path keys
    (goa_archive's 'rel' field, e.g. build_goa_archive_sidecar.py). A relpath miss or an
    entry with no usable tier text still __reject__'s, same as the .txt-missing case, rather
    than silently training on an unset prompt.
    """
    from caption_tools import make_caption_sampler
    sampler = make_caption_sampler(sidecar_path, probs=probs, track_type_prob=track_type_prob,
                                    key_fn=lambda info: info.get("relpath"))

    def fn(info, audio):
        result = sampler(info, audio)
        if not result.get("prompt"):
            return {"__reject__": True}
        return result

    return fn


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


def train(args):
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")

    seed = args.seed

    pl.seed_everything(seed, workers=True)

    _base_load_dtype, _base_cast, _trainer_precision = PRECISION_MODES[args.base_precision]
    model, model_config = load_model(
        args.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=_base_load_dtype,
    )

    if args.glitch:
        # weight-garden experiment: train adapters ON TOP of a mutated base.
        # Recipe is seed-reproducible (weight_mutations.py); scoped to the DiT
        # (model.model) so conditioner/pretransform stay pristine.
        from weight_mutations import Condition, apply_condition
        _recipe = json.loads(args.glitch)
        _gsum = apply_condition(model.model, Condition(**_recipe))
        print(f"[glitch] {_recipe['name']}: {_gsum['params_touched']} params, "
              f"blocks {_gsum['blocks_touched'][:6]}..")

    if args.full_finetune:
        # FULL fine-tune (no adapter). load_model() froze EVERYTHING
        # (requires_grad_(False)); here we unfreeze ONLY the DiT (model.model =
        # the DiffusionTransformer, ~1.4B params) so all of it trains. The
        # pretransform (SAME autoencoder — not even in the loop on pre-encoded
        # latents) and the conditioner (T5-Gemma text encoder) stay FROZEN, the
        # standard full-finetune recipe. No LoRA/DoRA is injected: lora_config is
        # set to None below, so the wrapper skips the adapter-injection path and
        # configure_optimizers routes the whole DiT (self.diffusion.model) into
        # the FusionOpt spectral/scalar groups (build_fusion_param_groups filters
        # to requires_grad, so the frozen AE/conditioner are excluded).
        model.model.requires_grad_(True).train()
        if model.pretransform is not None:
            model.pretransform.requires_grad_(False)
            model.pretransform.enable_grad = False
        model.conditioner.requires_grad_(False)
        _dit = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        _pt = sum(p.numel() for p in model.pretransform.parameters()) if model.pretransform is not None else 0
        _cond = sum(p.numel() for p in model.conditioner.parameters())
        print(f"[full-finetune] DiT trainable params: {_dit:,} "
              f"(pretransform {_pt:,} frozen, conditioner {_cond:,} frozen)")

    sample_rate = model.sample_rate
    ds_ratio = model.pretransform.downsampling_ratio

    # Align to downsampling ratio. --frames sets the exact latent T (multiple-of-256
    # convention, MASTER §5) and overrides --duration, avoiding seconds->ds rounding
    # that yields ragged tiles like T=506/1012 (Kim DIRECT 2026-07-13).
    if args.frames:
        sample_size = args.frames * ds_ratio
    else:
        sample_size = (int(args.duration * sample_rate) // ds_ratio) * ds_ratio
    _T = sample_size // ds_ratio
    print(f"[crop] T={_T} latent frames ({sample_size} samples, "
          f"{sample_size / sample_rate:.2f}s, ds_ratio={ds_ratio})")
    if args.frames and _T != args.frames:
        raise SystemExit(f"[crop] BUG: requested --frames {args.frames} but resolved T={_T}")

    # Extract tokenizers from conditioners for pre-tokenization in DataLoader workers
    tokenizers = {}
    if hasattr(model, "conditioner"):
        for key, cond in model.conditioner.conditioners.items():
            if hasattr(cond, "tokenizer") and hasattr(cond, "max_length"):
                tokenizers[key] = (cond.tokenizer, cond.max_length)

    if args.arc_data:
        # ARC-Forcing rollouts: fixed-length samples carrying their own inpaint
        # conditioning (drifted context clamped); no cropping on this path.
        dataset = ArcRolloutDataset(
            args.arc_data,
            sample_rate=sample_rate,
            downsampling_ratio=ds_ratio,
        )
        # Contract checks: multiple-of-256 rule (MASTER §5) and, when --frames is
        # given, that it matches the data (ARC samples are never cropped).
        _arc_lat, _ = dataset[0]
        _arc_T = _arc_lat.shape[-1]
        if _arc_T % 256 != 0:
            raise SystemExit(f"[arc] sample T={_arc_T} violates the multiple-of-256 rule (MASTER §5)")
        if args.frames and _arc_T != args.frames:
            raise SystemExit(f"[arc] --frames {args.frames} != dataset T={_arc_T} "
                             "(ARC samples are fixed-length; --frames must match the contract dir)")
        print(f"[arc] {len(dataset)} rollout samples, T={_arc_T} "
              f"({_arc_T * ds_ratio / sample_rate:.2f}s)")
    elif args.encoded_dir:
        # Multi-source: --encoded_dir and --caption_sidecar accept comma-separated
        # PARALLEL lists (per-source latent pools kept separate on disk, Kim's
        # layout rule; composed here into one LatentDatasetConfig list so each
        # source keeps its OWN caption sidecar — avoids cross-source stem collisions).
        dirs = [d.strip() for d in args.encoded_dir.split(",") if d.strip()]
        sidecars = [s.strip() for s in args.caption_sidecar.split(",")] if args.caption_sidecar else []
        if sidecars and len(sidecars) != len(dirs):
            raise ValueError(f"got {len(dirs)} encoded_dirs but {len(sidecars)} caption_sidecars "
                             "— pass one sidecar per dir (empty string to skip a source)")
        weights = [float(w) for w in args.source_weights.split(",")] if args.source_weights else [1.0] * len(dirs)
        if len(weights) != len(dirs):
            raise ValueError(f"got {len(dirs)} encoded_dirs but {len(weights)} source_weights")
        from caption_tools import make_caption_sampler
        probs = tuple(float(x) for x in args.caption_probs.split(","))
        configs = []
        for i, d in enumerate(dirs):
            sc = sidecars[i] if i < len(sidecars) and sidecars[i] else None
            fn = make_caption_sampler(sc, probs=probs,
                                      track_type_prob=args.track_type_prob) if sc else None
            configs.append(LatentDatasetConfig(id=f"train{i}", path=d, weight=weights[i],
                                               custom_metadata_fn=fn))
        dataset = PreEncodedDataset(
            configs,
            latent_crop_length=sample_size // ds_ratio,
            random_crop=True,
            beat_aware_crop=args.beat_aware_crop,
        )
    else:
        if args.caption_sidecar:
            # sidecar-driven captions (e.g. #90 goa_archive big-set): NOT
            # make_caption_sampler's per-clip-.txt path, see make_data_dir_caption_fn.
            probs = tuple(float(x) for x in args.caption_probs.split(","))
            md_fn = make_data_dir_caption_fn(args.caption_sidecar, probs, args.track_type_prob)
        else:
            md_fn = caption_metadata_fn
        dataset = SampleDataset(
            [
                LocalDatasetConfig(
                    id="train",
                    path=args.data_dir,
                    custom_metadata_fn=md_fn,
                )
            ],
            sample_size=sample_size,
            sample_rate=sample_rate,
            force_channels="stereo",
        )
    # Live-encode augmentation (--data_dir path only): append to the dataset's post-crop aug chain
    # so it hits the [C,T] audio BEFORE the in-loop SAME encode. Kept out of upstream dataset.py —
    # we mutate the instance's .augs here (train_lora is our territory). No-op for --encoded_dir.
    if getattr(args, "augment", False) and not args.encoded_dir:
        from audio_augment import AudioAugment   # sibling in scripts/
        dataset.augs = torch.nn.Sequential(
            *dataset.augs,
            AudioAugment(
                sample_rate=sample_rate,
                polarity_prob=0.0,   # PhaseFlipper (already in dataset.augs) covers polarity
                pitch_shift_prob=(0.3 if getattr(args, "aug_heavy", False) else 0.0),
                time_stretch_prob=(0.2 if getattr(args, "aug_heavy", False) else 0.0),
            ),
        )
        print(f"[augment] live-encode audio augmentation ON "
              f"({len(dataset.augs)} transforms; heavy={getattr(args, 'aug_heavy', False)})")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collation_fn,
        worker_init_fn=lambda worker_id: torch.manual_seed(seed + worker_id),
    )

    lora_state_dict = None
    if args.lora_checkpoint:
        lora_state_dict, _ = load_lora_checkpoint(args.lora_checkpoint)
    elif args.warm_start_ckpt:
        # Warm-start path for OLD-format ckpts that trainer.fit(ckpt_path=...)
        # rejects (saved before the on_save_checkpoint Lightning-key fix):
        # adapter weights load here, optimizer/scheduler state is restored by
        # the OptimizerWarmStart callback at on_train_start. Only loop counters
        # (epoch numbering / shuffle position) are lost vs a true resume.
        lora_state_dict, _ = load_lora_checkpoint(args.warm_start_ckpt)

    # --full-finetune => lora_config None: the wrapper injects NO adapter and
    # trains the (already-unfrozen) full DiT. Otherwise the normal adapter path.
    lora_config = None if args.full_finetune else {
        "rank": args.rank,
        "alpha": args.lora_alpha if args.lora_alpha is not None else args.rank,
        "adapter_type": args.adapter_type,
        "dropout": args.dropout,
        "include": args.include,
        "exclude": args.exclude,
        "phm_n": args.phm_n,
    }
    # Decoupled weight decay: None => optimiser-appropriate default preserving prior
    # behaviour (adamw 0.01, fusion 0.0). See --weight_decay help + the 2026-08-10
    # full-FT latent-runaway incident: fusion full-FT MUST override this (~0.1).
    _wd = args.weight_decay if args.weight_decay is not None else (0.0 if args.optimizer == "fusion" else 0.01)
    _warmup_steps = compute_warmup_steps(
        args.warmup_steps, args.warmup_frac, len(dataset), args.batch_size,
        args.accumulate_grad_batches, args.epochs, args.steps)
    if _warmup_steps:
        print(f"[fusion] warmup: {_warmup_steps} steps"
              + (f" ({args.warmup_frac:.0%} of total)" if args.warmup_frac is not None else ""))
    # force_scalar routing: regexes whose matching param NAMES are forced onto the scalar
    # (AdamW) path regardless of shape. --force-scalar-output expands to the SA3-DiT final
    # output projection + AdaLN modulation emitter (both otherwise route spectral). Empty
    # list => no key threaded into param_groups => byte-identical routing.
    _force_scalar = [pat for pat in args.force_scalar.split(",") if pat.strip()]
    if args.force_scalar_output:
        _force_scalar += [r"\.project_out\.", r"\.global_cond_embedder\.2\."]
    if args.optimizer == "fusion":
        # FusionOpt with ALL components enabled (full Fusion): Muon NS5 +
        # SF-NorMuon row-scaling + Schedule-Free averaging + MONA curvature
        # momentum + KL-Shampoo two-sided preconditioner. The wrapper's
        # configure_optimizers() routes the trainable LoRA adapters (model +
        # conditioner, requires_grad only) into spectral/scalar param groups via
        # build_fusion_param_groups(). hot_dtype=bf16 is the production default
        # (fp32-range exponent, can't overflow the NS5 quintic). lr from --lr is
        # the Schedule-Free Polyak gamma_base.
        optimizer_config = {
            "diffusion": {
                "optimizer": {
                    "type": "FusionOpt",
                    "config": {
                        "lr": args.lr,
                        # NOTE: this constructor-level weight_decay is only a FALLBACK for groups
                        # that don't set their own. build_fusion_param_groups DOES set a per-group
                        # weight_decay (spectral 0.01, scalar 0.0), which OVERRIDES this. So the
                        # EFFECTIVE knob for the DiT matrices is param_groups.spectral_wd below —
                        # NOT this line. (Left as the documented fallback; see the 2026-08-10
                        # full-FT latent-runaway incident.)
                        "weight_decay": _wd,
                        # shampoo default-OFF: KL-Shampoo preconditioners OOM rank-128 on 16GB
                        # LOCAL VRAM — a 64GB LUMI GCD fits them, hence the opt-in flag
                        # (--fusion-shampoo, alpha-campaign arm 7: "does KL-Shampoo help now
                        # that VRAM allows it").
                        # HYPERBALL is its own norm-constrained iterate → incompatible with
                        # Schedule-Free ('sf'); when --hyperball is set we DROP 'sf' (=>
                        # ['mona','ns5','normuon']) and pass hyperball=True. Off => unchanged.
                        "components": ((["mona", "ns5", "normuon"] if args.hyperball
                                        else ["mona", "ns5", "normuon", "sf"])
                                       + (["shampoo"] if args.fusion_shampoo else [])
                                       + (["cautious"] if args.cautious else [])),
                        "hyperball": args.hyperball,
                        "hot_dtype": "bf16",
                        "warmup_steps": _warmup_steps,
                        # per-DiT-layer update-weight schedule (None = uniform; splats to FusionOpt)
                        "layer_update_weights": load_layer_update_weights(args.layer_update_weights),
                    },
                    # param-group routing knobs (splat into build_fusion_param_groups in
                    # configure_optimizers, NOT into FusionOpt) — qkv row-block split, off by default.
                    # EFFECTIVE weight decay for full-FT lives HERE (routes to
                    # build_fusion_param_groups -> per-group group["weight_decay"], which
                    # overrides the constructor fallback above). --weight_decay sets spectral_wd
                    # (the 2D DiT matrices where the latent-scale runaway lives); scalar_wd stays
                    # 0.0 (never decay norms/biases). Omit --weight_decay => builder default 0.01
                    # (== what #68 ran, which was too weak for NS5/Muon -> drone). Pass ~0.1.
                    # force_scalar: regex list forcing matching param names onto the scalar
                    # (AdamW) path regardless of shape (--force-scalar / --force-scalar-output).
                    # Empty => no key threaded => byte-identical routing.
                    "param_groups": {"split_qkv": args.fusion_split_qkv,
                                     "split_adaln": args.fusion_split_adaln,
                                     **({"force_scalar": _force_scalar} if _force_scalar else {}),
                                     **({"spectral_wd": args.weight_decay}
                                        if args.weight_decay is not None else {})},
                }
            }
        }
    else:
        # AdamW. With --stochastic-rounding, swap in AdamWSR (bf16 master + bf16 Adam
        # states + stochastic-rounded param writeback) — the desktop-GPU (16 GB) full-FT
        # enabler. Byte-identical to plain AdamW when the flag is absent.
        adamw_type = "AdamWSR" if args.stochastic_rounding else "AdamW"
        optimizer_config = {
            "diffusion": {
                "optimizer": {
                    "type": adamw_type,
                    "config": {
                        "lr": args.lr,
                        "weight_decay": _wd,
                        "betas": [0.9, 0.95],
                    },
                }
            }
        }

    training_wrapper = DiffusionCondTrainingWrapper(
        model,
        mask_loss_weight=1.0,
        mask_padding_attention=True,
        silence_extension_scale_seconds=4.0,
        use_ema=(args.use_ema and args.full_finetune),
        ema_beta=args.ema_beta,
        ema_update_every=args.ema_update_every,
        ema_update_after_step=args.ema_warmup_steps,
        log_loss_info=False,
        optimizer_configs=optimizer_config,
        pre_encoded=bool(args.encoded_dir or args.arc_data),
        timestep_sampler="trunc_logit_normal",
        timestep_sampler_options={},
        inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}},
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", 44100),
        sample_size=model_config.get("sample_size"),
        lora_config=lora_config,
        lora_state_dict=lora_state_dict,
        svd_bases_path=args.svd_bases_path,
        log_every_n_steps=args.log_every,
        ot_coupling=True,
        # None skips the wrapper's downcast (base stays fp32) — fp32 AND the *-mixed
        # modes; "bf16"/"fp16" pass the cast target so cast_base_to_precision downcasts.
        base_precision=_base_cast,
        familiarity_beta=args.familiarity_beta,
        stereo_loss_weight=args.stereo_loss_weight,
        stereo_loss_tmax=args.stereo_loss_tmax,
        stereo_loss_subbatch=args.stereo_loss_subbatch,
        subspace_loss_basis=args.subspace_loss_basis,
        subspace_loss_weight=args.subspace_loss_weight,
        x0_equiv_loss=args.x0_equiv_loss,
        x0_loss_weight=args.x0_loss_weight,
        # Full-FT regularization A/B (2026-08-10): AGC grad-clip mode + targeted output-std
        # penalty. All OFF by default (grad-clip-mode norm, output-std-penalty 0.0) => the
        # wrapper path is byte-identical to prior runs.
        grad_clip_mode=args.grad_clip_mode,
        agc_lambda=args.agc_lambda,
        output_std_penalty=args.output_std_penalty,
        output_std_t_gate=args.output_std_t_gate,
    )

    if args.compile:
        # Compile the inner DiT — frozen base + LoRA parametrization passes through.
        # First step pays Triton autotune ~60-120s; Inductor cache persists via the
        # TRITON_CACHE_DIR set by rocm_env.yaml.
        print("[compile] torch.compile(training_wrapper.diffusion.model)")
        training_wrapper.diffusion.model = torch.compile(
            training_wrapper.diffusion.model
        )

    exc_callback = ExceptionCallback()

    if args.logger == "wandb":
        # WANDB_PROJECT groups an A/B's arms under ONE project (arm = run name) so they
        # overlay on a single dashboard; unset => project=args.name (prior behaviour).
        logger = pl.loggers.WandbLogger(
            project=os.environ.get("WANDB_PROJECT") or args.name, name=args.name)
        logger.watch(training_wrapper)

        if args.save_dir and isinstance(logger.experiment.id, str):
            checkpoint_dir = os.path.join(
                args.save_dir,
                logger.experiment.project,
                logger.experiment.id,
                "checkpoints",
            )
        else:
            checkpoint_dir = None
    elif args.logger == "comet":
        logger = pl.loggers.CometLogger(project=args.name)
        if args.save_dir and isinstance(logger.version, str):
            checkpoint_dir = os.path.join(
                args.save_dir, logger.name, logger.version, "checkpoints"
            )
        else:
            print(
                f"No save_dir specified, using {args.save_dir if args.save_dir else None}."
            )
            checkpoint_dir = args.save_dir if args.save_dir else None
    elif args.logger == "csv":
        logger = pl.loggers.CSVLogger(args.save_dir)
        checkpoint_dir = args.save_dir if args.save_dir else None
    else:
        logger = None
        checkpoint_dir = args.save_dir if args.save_dir else None

    if args.checkpoint_every_epochs:
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=args.checkpoint_every_epochs, dirpath=checkpoint_dir, save_top_k=-1
        )
    else:
        ckpt_callback = pl.callbacks.ModelCheckpoint(
            every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1
        )

    demo_dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        drop_last=True,
        collate_fn=collation_fn,
    )

    # Pre-fetch the first batch and cycle it so demos always use the same samples
    demo_batch = next(iter(demo_dl))
    _, metadata = demo_batch
    for j in range(min(4, len(metadata))):
        md = metadata[j]
        print(
            f"Demo sample {j}: prompt={md.get('prompt', '')} seconds_total={md.get('seconds_total', '')}"
        )
    demo_dl = itertools.cycle([demo_batch])

    demo_callback = DiffusionCondInpaintDemoCallback(
        demo_every=args.demo_every,
        sample_size=model_config.get("sample_size"),
        sample_rate=model_config.get("sample_rate"),
        demo_steps=50,
        num_demos=4,
        demo_cfg_scales=[2, 4, 7],
        demo_dl=demo_dl,
    )

    callbacks = [ckpt_callback, exc_callback]
    if not args.no_demos:
        callbacks.append(demo_callback)
    if args.warm_start_ckpt:
        from warm_start import OptimizerWarmStart
        callbacks.append(OptimizerWarmStart(args.warm_start_ckpt))

    # Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})

    if args.logger == "comet":
        logger.log_hyperparams(args_dict)

    if not hasattr(args, "gradient_clip_val") or args.gradient_clip_val == 0:
        args.gradient_clip_val = None

    summary = pl.callbacks.ModelSummary(max_depth=2)
    callbacks.append(summary)

    trainer = pl.Trainer(
        devices=(args.devices if args.devices else "auto"),
        num_nodes=args.num_nodes,
        accelerator="auto",
        # Explicit DDP when --devices N>1 (mirrors upstream stable-audio-tools train.py:239).
        # find_unused_parameters variant: adapter training freezes most of the DiT and not
        # every trainable param necessarily receives a grad each step — plain "ddp" can hang.
        strategy=("ddp_find_unused_parameters_true"
                  if (args.devices or 0) > 1 else "auto"),
        precision=_trainer_precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        max_steps=(-1 if args.epochs else args.steps),
        max_epochs=(args.epochs if args.epochs else None),
        default_root_dir=args.save_dir,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,  # If you need to debug validation, change this line
    )

    # ckpt_path resumes optimizer + LR-scheduler + epoch/step from a full Lightning
    # checkpoint (true continuation, not just LoRA-weight reload). max_epochs is the
    # TOTAL target: resuming an epoch-4 ckpt with --epochs 8 runs 3 more epochs.
    if args.resume_ckpt:
        # DoRA fats strip the frozen base at save time (on_save_checkpoint), so
        # Lightning's default STRICT restore rejects them wholesale (key-asymmetry
        # error, job 20173283: all 8 continuation arms dead in 3m36s). The base
        # weights come from from_pretrained and are frozen — a partial restore is
        # CORRECT for these checkpoints. PL2's module-level knob:
        training_wrapper.strict_loading = False
        print(f"[resume] strict_loading=False for {args.resume_ckpt} "
              f"(stripped-base DoRA/fullft fat; missing frozen-base keys are expected)")
    trainer.fit(training_wrapper, dataloader, ckpt_path=args.resume_ckpt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Simple LoRA fine-tuning for Stable Audio 3"
    )
    p.add_argument("--model", choices=list(base_models), default="medium-base")
    p.add_argument(
        "--data_dir",
        default=None,
        help="Folder with audio files and matching .txt captions",
    )
    p.add_argument(
        "--encoded_dir",
        default=None,
        help="Pre-encoded latent directory from pre_encode_dataset.py (.npy/.json pairs; captions embedded in .json, no .txt needed)",
    )
    p.add_argument(
        "--arc-data", "--arc_data",
        dest="arc_data",
        default=None,
        help="ARC-Forcing rollout .npz directory (shared data contract: per-sample "
             "context_latent/target_latent/mask/prompt/meta + manifest.json). Trains "
             "with the drifted rollout context CLAMPED via the inpaint conditioning "
             "keys; loss on the free (mask=0) region only. Mutually exclusive with "
             "--data_dir/--encoded_dir; samples are fixed-length (T multiple of 256).",
    )
    p.add_argument(
        "--beat-aware-crop",
        dest="beat_aware_crop",
        action="store_true",
        help="When using --encoded_dir, start each random sub-crop on a downbeat "
             "(from sibling .TIMESERIES.npz downbeat_activation_ts; falls back to "
             "beat_activation_ts, then uniform random if unavailable).",
    )
    p.add_argument("--rank", type=int, default=16)
    p.add_argument(
        "--lora_alpha",
        type=float,
        default=None,
        help="LoRA alpha scaling factor (default: same as rank)",
    )
    p.add_argument(
        "--adapter_type",
        choices=[
            "lora",
            "dora",
            "dora-rows",
            "dora-cols",
            "bora",
            "lora-xs",
            "dora-rows-xs",
            "dora-cols-xs",
            "bora-xs",
            "phm",
            "phm2",
        ],
        default="dora-rows",
    )
    p.add_argument(
        "--phm_n",
        type=int,
        default=4,
        help="PHM adapter algebra size n (adapter_type=phm; delta W = sum_i A_i kron B_i "
             "with learned n x n A_i). phm2 forces n=2 regardless of this flag.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout probability applied to LoRA inputs",
    )
    p.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Only apply LoRA to modules whose name contains one of these substrings",
    )
    p.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Skip modules whose name contains one of these substrings",
    )
    p.add_argument(
        "--svd_bases_path",
        default=None,
        help="Path to pre-computed SVD bases (.pt) for -XS adapter types",
    )
    p.add_argument(
        "--base_precision",
        choices=["bf16", "bfloat16", "fp16", "float16",
                 "bf16-mixed", "fp16-mixed", "fp32", "float32"],
        default="bf16",
        help="Precision mode (LoRA params always fp32; see PRECISION_MODES). "
             "bf16/fp16 = DOWNCAST the frozen base weights (memory) + matching autocast. "
             "bf16-mixed/fp16-mixed = base stays FP32 (fp32 master) + bf16/fp16 autocast "
             "(fp16-mixed's GradScaler is supplied by Lightning's 16-mixed). "
             "fp32 = full-precision: base fp32 + 32-true trainer. At long T with fp32, "
             "pair with SA3_SDPA_CAST_BF16=1 unless the SDPA backend supports fp32 "
             "(see transformer.py apply_attn) — the math fallback materializes "
             "T×T attention and OOMs at T=4096.",
    )
    p.add_argument(
        "--lora_checkpoint",
        default=None,
        help="Path to an existing LoRA .safetensors checkpoint to resume from",
    )
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the inner diffusion model (after LoRA is applied). "
                        "First step pays Triton/Inductor autotune cost; the cache persists "
                        "via the TRITON_CACHE_DIR set by rocm_env.yaml. Skip if it errors.")
    p.add_argument("--no_demos", action="store_true",
                   help="Skip the inpaint-demo callback (each demo = 3 cfg scales × 50 ODE "
                        "steps × ~4 s ≈ 10 min). Use for short tuning runs.")
    p.add_argument(
        "--optimizer",
        choices=["adamw", "fusion"],
        default="adamw",
        help="Optimizer for the trainable LoRA params. 'adamw' (default) = "
             "AdamW(lr, wd=0.01, betas=[0.9,0.95]). 'fusion' = FusionOpt with "
             "ALL components on (Muon NS5 + SF-NorMuon + Schedule-Free + MONA + "
             "KL-Shampoo, hot_dtype=bf16); routes the LoRA params into "
             "spectral/scalar groups automatically.",
    )
    p.add_argument(
        "--stochastic-rounding", "--stochastic_rounding",
        dest="stochastic_rounding",
        action="store_true",
        help="DESKTOP-GPU (16 GB) full-FT ENABLER. With --optimizer adamw, use AdamWSR "
             "instead of AdamW: master weights AND Adam states kept in bf16 (half the "
             "memory of an fp32 master + fp32 states), update math in fp32, and the new "
             "param written back to the bf16 master via STOCHASTIC ROUNDING. This lets "
             "small (sub-dominant, e.g. melody) gradient updates ACCUMULATE across steps "
             "instead of being deterministically truncated to zero on a bf16 master — the "
             "mechanism that makes a 1.4B DiT full-finetune fit and still learn on a 16 GB "
             "desktop card. Pair with --precision bf16 (bf16 master params) and "
             "--full-finetune. No effect with --optimizer fusion (FusionOpt-SF keeps its "
             "own fp32 master; this flag is the AdamW-path alternative). Absent = byte-"
             "identical to plain AdamW. NOT for LUMI runs.",
    )
    p.add_argument(
        "--fusion-split-qkv", action="store_true",
        help="FusionOpt only: orthogonalise fused ATTENTION up-projections (to_qkv/to_q/to_kv) "
             "per dim-row block instead of whole — q/k/v (+ the differential-attention diff "
             "blocks) each get their own NS5. Block count is read from each param's shape; "
             "non-attention params unchanged. Off by default.",
    )
    p.add_argument(
        "--fusion-split-adaln", action="store_true",
        help="FusionOpt only (CMuon, arXiv 2608.02502): orthogonalise the fused AdaLN modulation "
             "projection (global_cond_embedder.2, a [k*dim, dim] tensor emitting scale/shift/gate "
             "for self+ff) per dim-row block instead of whole. The k sub-matrices are functionally "
             "distinct but fused for efficiency; whole-tensor NS5 couples their subspaces. Block "
             "count k is read from the param's shape (shape[0]//shape[1]); non-AdaLN params "
             "unchanged. Composes with --fusion-split-qkv. Off by default.",
    )
    p.add_argument(
        "--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=0,
        help="FusionOpt only: linear ramp of the effective step size (gamma_t) from 0->1 "
             "over this many OPTIMIZER steps. Default 0 = no warmup (matches every Fusion "
             "run to date -- warmup_steps was a FusionOpt constructor param never threaded "
             "to a flag before 2026-08-14). No effect for adamw.",
    )
    p.add_argument(
        "--warmup-frac", "--warmup_frac", dest="warmup_frac", type=float, default=None,
        help="FusionOpt only: set --warmup-steps as a FRACTION of the run's total optimizer "
             "steps instead of a raw count (overrides --warmup-steps when given). E.g. 0.05 "
             "= 5%% of total steps -- exactly 1 epoch for any 20-epoch recipe, independent "
             "of corpus size (Kim direct 2026-08-14: 'aim for 5%% at least of the intended "
             "length'). Computed from len(dataset)//batch_size//accumulate_grad_batches * "
             "epochs (or --steps directly if --epochs is unset).",
    )
    p.add_argument(
        "--force-scalar", "--force_scalar", dest="force_scalar", default="",
        metavar="RE[,RE...]",
        help="FusionOpt only: comma-separated regexes (Python re, matched with .search "
             "against each param NAME) whose matching params are FORCED onto the AdamW/"
             "scalar path regardless of shape — the escape hatch for 2D matrices that "
             "misbehave under the spectral (NS5/Muon) update. Threaded into "
             "build_fusion_param_groups(force_scalar=[...]). Empty (default) = off = "
             "byte-identical routing. No effect for adamw.",
    )
    p.add_argument(
        "--force-scalar-output", "--force_scalar_output", dest="force_scalar_output",
        action="store_true",
        help="FusionOpt convenience: expand to the SA3-DiT regexes for the FINAL OUTPUT "
             "PROJECTION (transformer.project_out) and the AdaLN modulation emitter "
             "(global_cond_embedder.2, the [6*dim, dim] scale/shift/gate Linear) so you "
             "don't retype them — both currently route SPECTRAL (2D, min-dim>=128) and this "
             "forces them SCALAR. Merged with (does not override) any --force-scalar "
             "patterns. Off by default. No effect for adamw.",
    )
    p.add_argument(
        "--layer-update-weights", default=None, metavar="PATH[#CURVE]",
        help="FusionOpt only: per-DiT-layer update-weight schedule. PATH to a JSON "
             "{layer_index: multiplier}, or PATH#CURVE to select one curve from a multi-curve "
             "file (e.g. lumi/layer_curves/curves.json#protect_melody_log). Scales each "
             "transformer.layers.N param's post-NS5 spectral step by curve[N]; unset = uniform 1.0.",
    )
    p.add_argument(
        "--full-finetune", "--full_finetune",
        dest="full_finetune",
        action="store_true",
        help="FULL fine-tune the DiT instead of training a LoRA/DoRA adapter. "
             "Unfreezes ALL ~1.4B DiT params (model.model); the pretransform "
             "(SAME autoencoder) and conditioner (T5-Gemma) stay FROZEN. No adapter "
             "is injected. With --optimizer fusion, FusionOpt routes the whole DiT "
             "into spectral (2D matrices, min(shape)>=128) / scalar (biases, norms, "
             "convs, small 2D) groups. Checkpoints are FULL-MODEL (whole DiT state, "
             "not adapter deltas). Mutually exclusive with --lora_checkpoint / "
             "--warm_start_ckpt (adapter-resume paths). VRAM-heavy: pair with a "
             "modest --frames/--batch_size (see the feasibility table).",
    )
    p.add_argument("--use-ema", "--use_ema", dest="use_ema", action="store_true",
                   help="Maintain an EMA of the DiT weights (full-finetune ONLY — ignored for "
                        "LoRA/DoRA, which the wrapper force-disables). The EMA shadow is used for "
                        "demos/eval and saved in the checkpoint. Standing recipe finding: EMA damps "
                        "late-training drift; ~0.9999 tracks late sharpening. Was silently absent "
                        "(train_lora hardcoded use_ema=False + the SA3 fork stripped the build).")
    p.add_argument("--ema-beta", "--ema_beta", dest="ema_beta", type=float, default=0.9999,
                   help="EMA decay (higher = slower/smoother). Default 0.9999.")
    p.add_argument("--ema-update-every", "--ema_update_every", dest="ema_update_every", type=int, default=1,
                   help="Apply the EMA update every N optimizer steps (default 1).")
    p.add_argument("--ema-warmup-steps", "--ema_warmup_steps", dest="ema_warmup_steps", type=int, default=100,
                   help="Hard-track online weights for the first N steps so the EMA starts from a "
                        "real point, not init (default 100).")
    p.add_argument("--augment", action="store_true",
                   help="Live-encode ONLY (--data_dir): per-batch waveform augmentation before the "
                        "in-loop SAME encode — sub-frame phase shift (SAME latent is phase-variant), "
                        "gain, stereo width. See scripts/audio_augment.py. No effect on --encoded_dir.")
    p.add_argument("--aug-heavy", "--aug_heavy", dest="aug_heavy", action="store_true",
                   help="Also enable the content-altering axes (pitch shift, time stretch) — off by default.")
    p.add_argument("--cautious", action="store_true",
                   help="add cautious masking (C-Muon) to FusionOpt: zero update coords that "
                        "fight the gradient, rescale survivors. Otherwise identical to --optimizer "
                        "fusion. No effect for adamw.")
    p.add_argument("--fusion-shampoo", "--fusion_shampoo", dest="fusion_shampoo",
                   action="store_true",
                   help="add the KL-Shampoo two-sided preconditioner to FusionOpt's components "
                        "(default OFF: its preconditioners OOM rank-128 on 16GB local VRAM; a "
                        "64GB LUMI GCD fits them — alpha-campaign arm 7). No effect for adamw.")
    p.add_argument("--hyperball", action="store_true",
                   help="FusionOpt only (arXiv 2606.16899): constrain each spectral 2D weight "
                        "matrix to the hypersphere of radius R=‖W0‖_F (the loaded weight's norm, "
                        "captured once). Replaces the direct-to-p step with the Hyperball "
                        "retraction. Hyperball is its own iterate → INCOMPATIBLE with Schedule-"
                        "Free: setting this DROPS 'sf' from the components (=> ['mona','ns5',"
                        "'normuon']). Weight decay is IGNORED under hyperball (the norm "
                        "constraint replaces it). No effect for adamw. Off = unchanged.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=None,
                   help="decoupled weight decay. None => optimiser default (adamw 0.01, "
                        "fusion 0.0 — preserves prior behaviour). NOTE (2026-08-10 full-FT "
                        "latent-runaway incident): FusionOpt DEFAULTS TO 0.0 and with no decay "
                        "the NS5/Muon update norm is grad-magnitude-independent, so full-FT "
                        "weight norms grow unbounded -> latent scale explodes -> drone. Pass "
                        "~0.1 for full-FT fusion (AdamW's 0.01 is too weak — see "
                        "stable-audio-tools/tests/test_fusion_weight_decay.py). Pair with "
                        "--gradient_clip_val for a full-FT run.")
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument(
        "--duration",
        type=float,
        default=380.0,
        help="Maximum clip duration in seconds (default 380)",
    )
    p.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Exact latent crop length T, a MULTIPLE OF 256 (MASTER §5 convention). "
             "Overrides --duration to dodge seconds->ds-ratio rounding. "
             "T512/1024/2048 = 47.56/95.11/190.22s (Kim DIRECT 2026-07-13).",
    )
    p.add_argument("--epochs", type=int, default=None,
                   help="Train for N epochs (overrides --steps; uses Trainer max_epochs).")
    p.add_argument("--devices", type=int, default=None,
                   help="Number of GPUs for this training (Lightning Trainer devices=N). "
                        "Default None = devices='auto' (single-GPU today's behavior). N>1 also "
                        "forces strategy='ddp_find_unused_parameters_true' (upstream SAT "
                        "train.py:239 convention) instead of strategy='auto' — the auto path "
                        "SILENTLY falls back to N independent single-GPU trainers when a SLURM "
                        "cgroup (--gpus-per-task) hides the other devices (LUMI job 20413874: "
                        "8 duplicate trainers, versioned -vN ckpts, node burned for 3h). With an "
                        "explicit N, that same mis-launch dies loudly at startup instead.")
    p.add_argument("--num_nodes", "--num-nodes", dest="num_nodes", type=int, default=1,
                   help="Number of NODES for multi-node DDP (Lightning Trainer num_nodes=N). "
                        "Default 1 = single-node (all prior behavior). For a 4-node/32-GCD run: "
                        "--num_nodes 4 --devices 8, launched under `srun --nodes=4 "
                        "--ntasks-per-node=8` (world_size = num_nodes * devices = 32). Lightning's "
                        "SLURMEnvironment derives rank/world/MASTER_ADDR from the SLURM env; the "
                        "inter-node fabric (RCCL over Slingshot) needs the aws-ofi-rccl plugin + "
                        "NCCL_*/FI_CXI_* exports in the sbatch — a wrong fabric env HANGS at "
                        "rendezvous (validate with a small smoke first). Untested >1 node as of "
                        "2026-08-09; single-node DDP (--devices 8, num_nodes 1) is the proven path.")
    p.add_argument("--accumulate_grad_batches", type=int, default=1,
                   help="Gradient accumulation steps; effective batch = batch_size * this.")
    p.add_argument("--gradient_clip_val", type=float, default=0.0,
                   help="Gradient clip norm (0 = disabled). Applies in --grad-clip-mode norm.")
    p.add_argument("--grad-clip-mode", "--grad_clip_mode", dest="grad_clip_mode",
                   choices=["norm", "agc"], default="norm",
                   help="Gradient-clip strategy (2026-08-10 full-FT regularization A/B). "
                        "'norm' (default) = Lightning global-norm clip honouring --gradient_clip_val "
                        "(byte-identical to prior runs). 'agc' = scale-relative Adaptive Gradient "
                        "Clipping (NFNets/Brock): each 2D+ param's grad is clipped per output-unit "
                        "relative to that unit's own weight norm (--agc-lambda), reining in only "
                        "large-grad-relative-to-scale units — the gentler alternative to broad weight "
                        "decay for the FusionOpt full-FT latent-scale runaway. Overrides the "
                        "global-norm clip when set to agc; 1D/scalar params are left untouched.")
    p.add_argument("--agc-lambda", "--agc_lambda", dest="agc_lambda", type=float, default=0.01,
                   help="AGC clipping ratio lambda (--grad-clip-mode agc): coef = min(1, "
                        "lambda*||w||_unit/(||g||_unit+1e-6)). Smaller = tighter. Default 0.01.")
    p.add_argument("--output-std-penalty", "--output_std_penalty", dest="output_std_penalty",
                   type=float, default=0.0,
                   help="Targeted output-scale regularizer lambda_out (2026-08-10 full-FT A/B). "
                        "0 (default) = OFF, training path byte-identical. >0 adds "
                        "lambda_out * mean_c (std_c(z0_hat) - std_c(z0_data))^2 to the loss, where "
                        "z0_hat = noised - t*v_pred is the reconstructed clean latent and std_c is "
                        "the per-channel std over (batch,time). Penalizes the exact measured failure "
                        "(per-channel latent std running away) directly in the loss, not the weights, "
                        "so it can't broadly shrink features. Gated to low noise (t < "
                        "--output-std-t-gate) where z0_hat is reliable. Try 0.1 / 1.0. Logs "
                        "train/std_z0hat_mean + train/std_z0real_mean to the CSV each step.")
    p.add_argument("--output-std-t-gate", "--output_std_t_gate", dest="output_std_t_gate",
                   type=float, default=0.5,
                   help="Only apply the output-std penalty to samples with t < this (low noise, "
                        "where z0_hat = noised - t*v_pred is a meaningful clean estimate). Default 0.5.")
    p.add_argument("--checkpoint_every_epochs", type=int, default=None,
                   help="Checkpoint every N epochs (epoch mode; else use --checkpoint_every steps).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--caption_sidecar", "--caption-sidecar", dest="caption_sidecar", default=None,
                   help="captions.json sidecar (caption_tools.generate_sidecar) — enables "
                        "tiered T1/T2/T3 prompt sampling, overriding the latent jsons' prompts")
    p.add_argument("--subspace_loss_basis", "--subspace-loss-basis", dest="subspace_loss_basis",
                   default=None,
                   help="npz with orthonormal latent-subspace rows (`basis15` [k,C], or first-15 "
                        "of `melody_basis`); with --subspace-loss-weight K != 1, the RF error's "
                        "component inside this subspace is upweighted K x (spectral-bias counter, "
                        "arXiv 2503.03206; melody subspace = 9.3%% of corpus variance). Designed "
                        "for loss_normalization='none'.")
    p.add_argument("--subspace_loss_weight", "--subspace-loss-weight", dest="subspace_loss_weight",
                   type=float, default=1.0,
                   help="K multiplier for the subspace error share; 1.0 = off (byte-identical path)")
    p.add_argument("--x0_equiv_loss", "--x0-equiv-loss", dest="x0_equiv_loss",
                   action="store_true",
                   help="E1a (JLT arXiv 2605.27102 port): weight the RF loss by sigma^2*3 "
                        "— exactly equivalent to an x0-space MSE for a v-output head; zero "
                        "inference changes. Tests the loss-geometry half of clean-latent "
                        "prediction. Pre-test: v-trained base recovers low-variance "
                        "eigendirections 2-8x worse per unit signal.")
    p.add_argument("--x0-loss-weight", "--x0_loss_weight", dest="x0_loss_weight",
                   type=float, default=0.0,
                   help="x0-reconstruction ADD-a-term weight lambda_x0 (distinct from "
                        "--x0_equiv_loss, which REPLACES the v-loss with its sigma^2-weighted "
                        "form). 0 (default) = OFF, training path byte-identical. >0 adds "
                        "lambda_x0 * MSE(z0_hat, z0) to the loss, where z0_hat = noised - "
                        "t*v_pred is the reconstructed clean latent (rf_z0_hat, the same "
                        "reconstruction the output-std/stereo blocks use). Predicting toward "
                        "the clean latent removes the ambient isotropic-noise component that "
                        "pressures output-scale growth (arXiv 2605.27102 JLT). RF objective "
                        "only. Logs train/x0_recon_loss. Try 0.1 / 1.0.")
    p.add_argument("--familiarity_beta", "--familiarity-beta", dest="familiarity_beta",
                   type=float, default=0.0,
                   help="familiarity-normalized loss weighting exponent (scripts/"
                        "familiarity.py): >0 down-weights crops the model already fits "
                        "(per-crop EMA of relative loss), keeps remote crops hot; 0=off")
    p.add_argument("--stereo_loss_weight", "--stereo-loss-weight", dest="stereo_loss_weight",
                   type=float, default=0.0,
                   help="Stereo-preservation aux loss weight (training/stereo_loss.py). "
                        "0 (default) = OFF, training path byte-identical to baseline. "
                        ">0 decodes z0_hat + the target latent to audio and matches the "
                        "SIDE (L-R)/2 channel (RMS + multi-res STFT), penalizing stereo "
                        "collapse — the meter-in-the-gradient fix for DoRA going mono. "
                        "Try 0.1 / 0.3. VRAM: decode is in-loop, gate + sub-batch below.")
    p.add_argument("--stereo_loss_tmax", "--stereo-loss-tmax", dest="stereo_loss_tmax",
                   type=float, default=0.3,
                   help="Only apply the stereo aux loss to rows with t < tmax (low noise, "
                        "where z0_hat = noised - t*v_pred is a meaningful clean estimate). "
                        "Also caps decode cost — high-noise rows are skipped entirely.")
    p.add_argument("--stereo_loss_subbatch", "--stereo-loss-subbatch", dest="stereo_loss_subbatch",
                   type=int, default=2,
                   help="Max rows to decode for the stereo aux loss per step (VRAM cap; "
                        "in-loop decode of stereo audio is the expensive part).")
    p.add_argument("--track_type_prob", type=float, default=0.0,
                   help="probability of prepending 'TrackType: Music, VocalType: "
                        "Instrumental, ' to sampled captions (SA3 paper §5.1: base "
                        "trained ~50%% with AudioSparx prefixes; 0.5 mirrors that)")
    p.add_argument("--caption_probs", default="0.6,0.3,0.1",
                   help="sampling probabilities for caption tiers t1,t2,t3")
    p.add_argument("--source_weights", default=None,
                   help="comma-separated per-source sample weights (parallel to --encoded_dir); "
                        "default 1.0 each = natural per-crop proportions")
    p.add_argument("--resume_ckpt", "--resume-ckpt", dest="resume_ckpt", default=None,
                   help="full Lightning .ckpt to RESUME from (restores optimizer/scheduler/epoch); "
                        "--epochs is the total target, so resuming an ep-4 ckpt with --epochs 8 = 3 more")
    p.add_argument("--warm_start_ckpt", "--warm-start-ckpt", dest="warm_start_ckpt", default=None,
                   help="OLD-format .ckpt (pre Lightning-key fix, rejected by --resume_ckpt) to "
                        "warm-start from: loads adapter weights + optimizer/scheduler state, but "
                        "epoch/step counters restart — so --epochs is the ADDITIONAL count "
                        "(continuing an ep-4 ckpt for 3 more = --epochs 3)")
    p.add_argument("--glitch", default=None,
                   help="JSON Condition kwargs (scripts/weight_mutations.py) applied to "
                        "the frozen base DiT before adapter attach — trains a LoRA/DoRA "
                        "on a deliberately mutated base (weight-garden healing experiment)")
    p.add_argument("--logger", choices=["wandb", "comet", "csv", "none"], default="csv")
    p.add_argument("--name", type=str, default="lora-finetune")
    p.add_argument("--save_dir", type=str, default="./lora_checkpoints")
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--demo_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()
    if args.use_ema and not args.full_finetune:
        print("[ema] NOTE: --use-ema is full-finetune only (LoRA/DoRA force EMA off) — ignoring it.")
    if args.warm_start_ckpt and args.resume_ckpt:
        p.error("--warm_start_ckpt and --resume_ckpt are mutually exclusive "
                "(use --resume_ckpt for new-format ckpts, --warm_start_ckpt for old)")
    if args.full_finetune and (args.lora_checkpoint or args.warm_start_ckpt):
        p.error("--full-finetune injects NO adapter, so --lora_checkpoint / "
                "--warm_start_ckpt (adapter-resume paths) do not apply. Use "
                "--resume_ckpt to continue a full-finetune Lightning checkpoint.")
    if args.arc_data and (args.encoded_dir or args.data_dir):
        p.error("--arc-data is mutually exclusive with --data_dir/--encoded_dir")
    if not args.encoded_dir and not args.data_dir and not args.arc_data:
        p.error("one of --data_dir, --encoded_dir or --arc-data is required")
    train(args)


if __name__ == "__main__":
    main()
