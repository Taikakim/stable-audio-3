#!/usr/bin/env python3
"""Experimental Modular Stage-Based Optimizer Trainer for SA3 LoRA.

A standalone copy of train_lora.py, stripped of research-specific features (glitch,
mir_ctrl, traj_sketch, subspace loss, familiarity) and extended with:
  - --optimizer modular: the ModularOptimizer 6-stage pipeline
  - --subset300: shortcut to /home/kim/Projects/latents_sa3_subset300
  - Modular sub-flags: --modular-whitening, --modular-escape-velocity, etc.

Origin: Kim & Antigravity.Neuromancer
"""

import os

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
_rocm_core = "/home/kim/Projects/SAO/.venv/lib/python3.13/site-packages/_rocm_sdk_core/lib"
_rocm_math = "/home/kim/Projects/SAO/.venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/host-math/lib"
os.environ["LD_LIBRARY_PATH"] = f"{_rocm_core}:{_rocm_math}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import argparse
import itertools
import json
import resource
from pathlib import Path

# Bump open file descriptor limit to avoid "Too many open files" across epochs/workers
try:
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))
except Exception:
    pass

# Apply ROCm/AMD environment BEFORE importing torch
import stable_audio_tools
import torch

try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass

try:
    from stable_audio_tools.training.modular_opt.stack import LIFOTransformationStack
    torch.serialization.add_safe_globals([LIFOTransformationStack])
except Exception:
    pass

import pytorch_lightning as pl

from stable_audio_3.data.dataset import (
    LatentDatasetConfig,
    PreEncodedDataset,
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

# Precision mode matrix (identical to train_lora.py)
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

# Canonical subset path (symlinks)
SUBSET_300_PATH = "/home/kim/Projects/latents_sa3_subset300"


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


class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f"{type(err).__name__}: {err}")


# ---------------------------------------------------------------------------
# ModularTrainingWrapper — subclass that adds ModularOptimizer dispatch
# ---------------------------------------------------------------------------

class ModularTrainingWrapper(DiffusionCondTrainingWrapper):
    """Extends DiffusionCondTrainingWrapper with ModularOptimizer parameter routing.

    When opt_type == 'ModularOptimizer', uses build_modular_param_groups from
    stable_audio_tools.training.modular_opt to create role-based parameter groups,
    then instantiates ModularOptimizer directly.

    For all other optimizer types, delegates to the parent's configure_optimizers().
    """

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_type = diffusion_opt_config['optimizer'].get('type')

        if opt_type != 'ModularOptimizer':
            # Delegate to parent for FusionOpt, AdamW, Lion, etc.
            return super().configure_optimizers()

        from stable_audio_tools.training.modular_opt import (
            ModularOptimizer,
            build_modular_param_groups,
            summarise_modular_groups,
        )

        opt_cfg = diffusion_opt_config['optimizer'].get('config', {})
        pg_cfg = diffusion_opt_config['optimizer'].get('param_groups', {}) or {}

        # Route over the whole self.diffusion so conditioner LoRA params are included
        route_root = self.diffusion if self.lora_config is not None else self.diffusion.model
        opt_params = build_modular_param_groups(route_root, **pg_cfg)

        try:
            from torch.distributed import get_rank
            rank = get_rank()
        except Exception:
            rank = 0
        if rank == 0:
            print("ModularOptimizer param groups:")
            print(summarise_modular_groups(opt_params))

        optimizer = ModularOptimizer(opt_params, **opt_cfg)
        print(optimizer.summary())

        return [optimizer]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Remap stripped LoRA checkpoint keys to the wrapper's module hierarchy.

        Checkpoints saved by on_save_checkpoint carry keys relative to self.diffusion.model
        ('model.transformer...') and self.diffusion.conditioner ('conditioners...').
        During trainer.fit(ckpt_path=...), Lightning expects keys relative to the wrapper root
        ('diffusion.model.model...' and 'diffusion.conditioner.conditioners...').
        """
        if "state_dict" in checkpoint:
            remapped_sd = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("diffusion."):
                    remapped_sd[k] = v
                elif k.startswith("model."):
                    remapped_sd[f"diffusion.model.{k}"] = v
                elif k.startswith("conditioner.") or k.startswith("conditioners."):
                    remapped_sd[f"diffusion.conditioner.{k}"] = v
                else:
                    remapped_sd[f"diffusion.{k}"] = v
            checkpoint["state_dict"] = remapped_sd
            print(f"[on_load_checkpoint] Remapped {len(remapped_sd)} checkpoint keys to wrapper hierarchy.")


# ---------------------------------------------------------------------------
# train()
# ---------------------------------------------------------------------------

def train(args):
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")

    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    seed = args.seed
    pl.seed_everything(seed, workers=True)

    _base_load_dtype, _base_cast, _trainer_precision = PRECISION_MODES[args.base_precision]
    model, model_config = load_model(
        args.model, torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        dtype=_base_load_dtype,
    )

    sample_rate = model.sample_rate
    ds_ratio = model.pretransform.downsampling_ratio

    # Align to downsampling ratio
    if args.frames:
        sample_size = args.frames * ds_ratio
    else:
        sample_size = (int(args.duration * sample_rate) // ds_ratio) * ds_ratio
    _T = sample_size // ds_ratio
    print(f"[crop] T={_T} latent frames ({sample_size} samples, "
          f"{sample_size / sample_rate:.2f}s, ds_ratio={ds_ratio})")

    # ---- Dataset ----
    encoded_dir = args.encoded_dir
    if args.subset300:
        if not os.path.isdir(SUBSET_300_PATH):
            raise SystemExit(f"--subset300: directory not found: {SUBSET_300_PATH}")
        encoded_dir = SUBSET_300_PATH
        print(f"[subset300] Using canonical 300-track subset: {SUBSET_300_PATH}")

    if not encoded_dir:
        raise SystemExit("Must specify --encoded_dir or --subset300")

    dirs = [d.strip() for d in encoded_dir.split(",") if d.strip()]
    sidecars = [s.strip() for s in args.caption_sidecar.split(",")] if args.caption_sidecar else []
    if sidecars and len(sidecars) != len(dirs):
        raise ValueError(f"got {len(dirs)} encoded_dirs but {len(sidecars)} caption_sidecars")
    weights = [1.0] * len(dirs)

    configs = []
    for i, d in enumerate(dirs):
        sc = sidecars[i] if i < len(sidecars) and sidecars[i] else None
        fn = None
        if sc:
            from caption_tools import make_caption_sampler
            fn = make_caption_sampler(sc, probs=(0, 0.9, 0.1))
            print(f"[captions] source {i} ({os.path.basename(d)}): sidecar={os.path.basename(sc)}")
        configs.append(LatentDatasetConfig(id=f"train{i}", path=d, weight=weights[i],
                                           custom_metadata_fn=fn))

    dataset = PreEncodedDataset(
        configs,
        latent_crop_length=sample_size // ds_ratio,
        random_crop=True,
    )
    print(f"[dataset] {len(dataset)} samples from {len(dirs)} source(s)")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        collate_fn=collation_fn,
        worker_init_fn=lambda worker_id: torch.manual_seed(seed + worker_id),
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # ---- LoRA config ----
    lora_state_dict = None
    if args.lora_checkpoint:
        lora_state_dict, _ = load_lora_checkpoint(args.lora_checkpoint)

    lora_config = {
        "rank": args.rank,
        "alpha": args.lora_alpha if args.lora_alpha is not None else args.rank,
        "adapter_type": args.adapter_type,
        "dropout": args.dropout,
        "include": args.include,
        "exclude": args.exclude,
    }

    # ---- Optimizer config ----
    _wd = args.weight_decay if args.weight_decay is not None else 0.01

    if args.optimizer == "modular":
        optimizer_config = {
            "diffusion": {
                "optimizer": {
                    "type": "ModularOptimizer",
                    "config": {
                        "lr": args.lr,
                        "beta1": args.modular_beta1,
                        "beta_precond": args.modular_beta_precond,
                        "precond_delta": 1e-4,
                        "precond_update_freq": args.modular_precond_freq,
                        "precond_alpha": args.modular_precond_alpha,
                        "precond_bottleneck": args.modular_precond_bottleneck,
                        "ns_poly": args.modular_ns_poly,
                        "escape_velocity": args.modular_escape_velocity,
                        "ev_beta": args.modular_ev_beta,
                        "ev_max": args.modular_ev_max,
                        "snr_gate": args.modular_snr_gate,
                        "snr_beta": 0.9,
                        "snr_floor": 0.0,
                        "weight_decay": _wd,
                        "wd_overtraining": args.wd_overtraining,
                        "steps_per_epoch": len(dataset) / (args.batch_size * max(1, args.accumulate_grad_batches)),
                        "normuon": args.modular_normuon,
                        "normuon_beta": args.modular_normuon_beta,
                        "schedule_free": args.modular_schedule_free,
                        "sf_beta": 0.9,
                        "sf_c_warmup": getattr(args, "modular_sf_c_warmup", None),
                        "sf_r": getattr(args, "modular_sf_r", 1.0),
                        "warmup_steps": args.warmup_steps,
                        "var_dampening_threshold": (args.var_dampening if getattr(args, "var_damp_opt", True) else None),
                        "var_dampening_power": getattr(args, "var_damp_power", 1.0),
                        "var_wd_boost": getattr(args, "var_wd_boost", 0.0),
                        "radial_brake": getattr(args, "modular_radial_brake", 1.0),
                    },
                    "param_groups": {
                        "default_whitening": args.modular_whitening,
                        "split_qkv": args.modular_split_qkv,
                        "split_adaln": args.modular_split_adaln,
                        **({"spectral_wd": args.weight_decay}
                           if args.weight_decay is not None else {}),
                    },
                }
            }
        }
    elif args.optimizer == "fusion":
        optimizer_config = {
            "diffusion": {
                "optimizer": {
                    "type": "FusionOpt",
                    "config": {
                        "lr": args.lr,
                        "weight_decay": _wd,
                        "components": ["mona", "ns5", "normuon", "sf"],
                        "hot_dtype": "bf16",
                        "warmup_steps": args.warmup_steps,
                    },
                    "param_groups": {
                        "split_qkv": True,
                        "split_adaln": True,
                    },
                }
            }
        }
    elif args.optimizer == "lion":
        optimizer_config = {
            "diffusion": {
                "optimizer": {
                    "type": "LionSR",
                    "config": {
                        "lr": args.lr,
                        "weight_decay": _wd,
                        "betas": [0.9, 0.99],
                    },
                }
            }
        }
    else:  # adamw
        optimizer_config = {
            "diffusion": {
                "optimizer": {
                    "type": "AdamW",
                    "config": {
                        "lr": args.lr,
                        "weight_decay": _wd,
                        "betas": [0.9, 0.95],
                    },
                }
            }
        }

    # ---- Training wrapper ----
    training_wrapper = ModularTrainingWrapper(
        model,
        mask_loss_weight=1.0,
        mask_padding_attention=True,
        silence_extension_scale_seconds=4.0,
        use_ema=False,
        log_loss_info=False,
        optimizer_configs=optimizer_config,
        pre_encoded=True,
        timestep_sampler="trunc_logit_normal",
        timestep_sampler_options={},
        inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}},
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", 44100),
        sample_size=model_config.get("sample_size"),
        lora_config=lora_config,
        lora_state_dict=lora_state_dict,
        log_every_n_steps=args.log_every,
        ot_coupling=True,
        base_precision=_base_cast,
        grad_clip_mode="norm",
        latent_var_barrier=(args.var_dampening if args.var_dampening is not None else 0.0),
        latent_var_weight=(args.var_barrier_weight if args.var_dampening is not None else 0.0),
    )

    # ---- Callbacks ----
    exc_callback = ExceptionCallback()
    run_dir = os.path.join(args.save_dir, args.name) if (args.save_dir and args.name) else (args.save_dir or "./")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")

    if args.logger == "wandb":
        wandb_proj = getattr(args, "wandb_project", None) or os.environ.get("WANDB_PROJECT") or args.name
        logger = pl.loggers.WandbLogger(
            project=wandb_proj, name=args.name)
        logger.watch(training_wrapper)
    elif args.logger == "csv":
        logger = pl.loggers.CSVLogger(run_dir)
    else:
        logger = None

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=args.checkpoint_every, dirpath=checkpoint_dir, save_top_k=-1
    )

    demo_dl = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=collation_fn,
    )
    demo_batch = next(iter(demo_dl))
    _, metadata = demo_batch
    for j in range(min(4, len(metadata))):
        md = metadata[j]
        print(f"Demo sample {j}: prompt={md.get('prompt', '')} "
              f"seconds_total={md.get('seconds_total', '')}")
    demo_dl = itertools.cycle([demo_batch])

    callbacks = [ckpt_callback, exc_callback, pl.callbacks.ModelSummary(max_depth=2)]

    # Opt-in NaN localisation: inert unless SA3_NAN_TRIPWIRE=1 is exported.
    from scripts.nan_tripwire_callback import maybe_build as _maybe_tripwire
    _tripwire = _maybe_tripwire()
    if _tripwire is not None:
        callbacks.append(_tripwire)

    # Wide-side covariance spectrum probe: inert unless SA3_COV_PROBE=1.
    from scripts.wide_covariance_probe import maybe_build as _maybe_cov
    _cov = _maybe_cov(out_dir=run_dir, args=args)
    if _cov is not None:
        callbacks.append(_cov)

    # Mechanism audit: report which optimizer mechanisms actually affect the run.
    # ON by default (print-only, reads already-computed telemetry); SA3_MECHANISM_AUDIT=0 silences.
    from scripts.mechanism_audit import maybe_build as _maybe_audit
    _audit = _maybe_audit(args, report_every=max(500, (args.log_every or 1) * 10))
    if _audit is not None:
        callbacks.append(_audit)

    if args.eval_demos:
        from scripts.eval_demo_callback import ModularDemoAndLossGuardCallback
        eval_cb = ModularDemoAndLossGuardCallback(
            save_dir=run_dir,
            loss_guard_threshold=args.loss_guard_threshold,
            step_milestones=tuple(args.eval_milestones) if args.eval_milestones else (100, 300, 600, 1200, 1800, 2400, 3000),
            demo_steps=24,
            demo_cfg=7.0,
            frames_native=args.frames if args.frames else 512,
            render_continuations=getattr(args, "eval_continuations", False),
            num_prompts=getattr(args, "eval_num_prompts", 3),
            cfg_rescale=getattr(args, "demo_cfg_rescale", 0.0),
            max_latent_std=getattr(args, "demo_latent_clamp", None),
        )
        callbacks.append(eval_cb)
        args.no_demos = True

    if not args.no_demos:
        demo_callback = DiffusionCondInpaintDemoCallback(
            demo_every=args.demo_every,
            sample_size=model_config.get("sample_size"),
            sample_rate=model_config.get("sample_rate"),
            demo_steps=50,
            num_demos=4,
            demo_cfg_scales=[2, 4, 7],
            demo_dl=demo_dl,
        )
        callbacks.append(demo_callback)

    # ---- Trainer ----
    if not hasattr(args, "gradient_clip_val") or args.gradient_clip_val == 0:
        args.gradient_clip_val = None

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        strategy="auto",
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
        num_sanity_val_steps=0,
    )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        meta_path = os.path.join(args.save_dir, "run_meta.json")
        meta = {
            "what_this_is": "LoRA training with Modular Stage-Based Optimizer (ModularOptimizer) on canonical 300 subset.",
            "optimizer": args.optimizer,
            "lr": args.lr,
            "weight_decay": _wd,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "steps": args.steps,
            "frames": args.frames,
            "seed": args.seed,
            "eval_demos": args.eval_demos,
            "loss_guard_threshold": args.loss_guard_threshold,
            "recipe": {
                "model": args.model,
                "adapter": f"{args.adapter_type} rank{args.rank} alpha{args.lora_alpha or args.rank}",
                "whitening": getattr(args, "modular_whitening", None),
                "split_qkv": getattr(args, "modular_split_qkv", None),
                "split_adaln": getattr(args, "modular_split_adaln", None),
            }
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[run_meta] Wrote run metadata to {meta_path}")

    if args.resume_ckpt:
        training_wrapper.strict_loading = False
        print(f"[resume] strict_loading=False for {args.resume_ckpt} "
              f"(stripped-base DoRA fat; missing frozen-base keys are expected)")

    trainer.fit(training_wrapper, dataloader, ckpt_path=args.resume_ckpt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Experimental Modular Stage-Based Optimizer Trainer for SA3 LoRA"
    )

    # ---- Model ----
    p.add_argument("--model", type=str, default="medium-base", choices=list(base_models))

    # ---- Data ----
    p.add_argument("--encoded_dir", type=str, default=None,
                   help="Pre-encoded latent directory (or comma-separated list)")
    p.add_argument("--subset300", action="store_true",
                   help=f"Use canonical 300-track subset at {SUBSET_300_PATH}")
    p.add_argument("--caption_sidecar", type=str, default=None,
                   help="Caption JSON sidecar (comma-separated for multi-source)")
    p.add_argument("--duration", type=float, default=380.0,
                   help="Audio duration in seconds (default: 380)")
    p.add_argument("--frames", type=int, default=None,
                   help="Exact latent length T (multiple of 256, overrides --duration)")

    # ---- Adapter ----
    p.add_argument("--rank", type=int, default=128, help="LoRA rank (default: 128)")
    p.add_argument("--lora_alpha", type=int, default=None,
                   help="LoRA alpha (default: same as rank)")
    p.add_argument("--adapter_type", type=str, default="dora-rows",
                   choices=["lora", "dora", "dora-rows", "dora-cols", "bora"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--include", type=str, nargs="*", default=None,
                   help="Regex patterns for layers to include in LoRA")
    p.add_argument("--exclude", type=str, nargs="*", default=None,
                   help="Regex patterns for layers to exclude from LoRA")
    p.add_argument("--lora_checkpoint", type=str, default=None,
                   help="Resume adapter weights from this checkpoint")

    # ---- Optimizer ----
    p.add_argument("--optimizer", type=str, default="modular",
                   choices=["modular", "fusion", "adamw", "lion"],
                   help="Optimizer type (default: modular)")
    p.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    p.add_argument("--weight_decay", "--modular-wd", "--modular_wd", type=float, default=0.01,
                   dest="weight_decay", help="Weight decay (default: 0.01)")
    p.add_argument("--warmup-steps", "--warmup_steps", type=int, default=0, dest="warmup_steps")

    # ---- Modular sub-flags ----
    mod = p.add_argument_group("Modular optimizer flags")
    mod.add_argument("--modular-precond-alpha", type=float, default=0.125,
                     dest="modular_precond_alpha",
                     help="Mousse Spectral Tempering: curvature exponent alpha in Lambda^-alpha "
                          "(Algorithm 1 line 6). Default 0.125 per the paper's Figure 7a ablation, "
                          "which beats the classic Shampoo 0.25.")
    mod.add_argument("--modular-no-precond-bottleneck", action="store_false", default=True,
                     dest="modular_precond_bottleneck",
                     help="Precondition BOTH sides. Off by default: on LoRA factors the wide side "
                          "is up to 12288, and one such eigh measured 7.2 s on this card.")
    mod.add_argument("--cov-probe", action="store_true", default=False, dest="cov_probe",
                     help="Measure the wide-side gradient covariance spectrum and write "
                          "wide_covariance_spectrum.json. Decides whether a low-rank sketch "
                          "of the wide Kronecker factor is viable at all.")
    mod.add_argument("--cov-probe-snaps", type=int, default=0, dest="cov_probe_snaps",
                     help="Gradient snapshots to capture (default 64).")
    mod.add_argument("--cov-probe-every", type=int, default=0, dest="cov_probe_every",
                     help="Capture one snapshot every N optimizer steps (default 4). Use 1 "
                          "on short runs so the horizon fits inside the run.")
    mod.add_argument("--modular-whitening", type=str, default="none",
                     choices=["none", "shampoo", "soap"],
                     dest="modular_whitening",
                     help="Preconditioner for forward whitening (default: none)")
    mod.add_argument("--modular-ns-poly", "--modular-lmo-poly", type=str, default="quintic",
                     choices=["quintic", "cubic", "cubic5"],
                     dest="modular_ns_poly",
                     help="Newton-Schulz polynomial for SpectralLMO (default: quintic, cubic5 eliminates A@A)")
    mod.add_argument("--modular-split-qkv", action="store_true", default=True,
                     dest="modular_split_qkv",
                     help="Split fused QKV attention into per-head blocks (default: True)")
    mod.add_argument("--modular-no-split-qkv", action="store_false",
                     dest="modular_split_qkv")
    mod.add_argument("--modular-split-adaln", action="store_true", default=True,
                     dest="modular_split_adaln",
                     help="Split fused AdaLN emitters into functional blocks (default: True)")
    mod.add_argument("--modular-no-split-adaln", action="store_false",
                     dest="modular_split_adaln")
    mod.add_argument("--modular-escape-velocity", "--modular-ev", action="store_true", default=False,
                     dest="modular_escape_velocity",
                     help="Enable Prodigy escape velocity (dual-norm coupled)")
    mod.add_argument("--modular-ev-beta", type=float, default=0.999,
                     dest="modular_ev_beta",
                     help="Prodigy escape velocity EMA beta (default: 0.999)")
    mod.add_argument("--modular-ev-max", type=float, default=2.0,
                     dest="modular_ev_max",
                     help="Maximum step multiplier cap for escape velocity (default: 2.0)")
    mod.add_argument("--modular-snr-gate", action="store_true", default=False,
                     dest="modular_snr_gate",
                     help="Enable SNR gate on raw gradient")
    mod.add_argument("--modular-normuon", action="store_true", default=True,
                     dest="modular_normuon",
                     help="Enable NorMuon per-neuron row scaling (default: True)")
    mod.add_argument("--modular-no-normuon", action="store_false",
                     dest="modular_normuon",
                     help="Disable NorMuon per-neuron row scaling")
    mod.add_argument("--modular-normuon-beta", type=float, default=0.95,
                     dest="modular_normuon_beta",
                     help="NorMuon row-norm EMA beta (default: 0.95)")
    mod.add_argument("--modular-schedule-free", action="store_true", default=True,
                     dest="modular_schedule_free",
                     help="Enable Schedule-Free averaging (default: True)")
    mod.add_argument("--modular-no-schedule-free", action="store_false",
                     dest="modular_schedule_free",
                     help="Disable Schedule-Free averaging")
    mod.add_argument("--modular-sf-c-warmup", type=int, default=None,
                     dest="modular_sf_c_warmup",
                     help="Schedule-Free burn-in steps before iterate averaging kicks in (default: 2*warmup)")
    mod.add_argument("--modular-sf-r", type=float, default=1.0,
                     dest="modular_sf_r",
                     help="Schedule-Free power weighting power r (default: 1.0)")
    mod.add_argument("--modular-beta1", type=float, default=0.9,
                     dest="modular_beta1",
                     help="Momentum beta (default: 0.9)")
    mod.add_argument("--modular-beta-precond", type=float, default=0.95,
                     dest="modular_beta_precond",
                     help="Preconditioner covariance EMA beta (default: 0.95)")
    mod.add_argument("--modular-precond-freq", type=int, default=10,
                     dest="modular_precond_freq",
                     help="Preconditioner eigendecomposition refresh interval. Default 10, "
                          "matching the Mousse reference (dion/dion/mousse.py:80). At 1 this "
                          "is an eigh per tensor per step.")
    mod.add_argument("--wd-overtraining", "--wd_overtraining", "--modular-wd-overtraining", action="store_true", default=False,
                     dest="wd_overtraining",
                     help="Scale weight decay by sqrt(epochs) according to overtraining factor (Everett & Qiu 2026)")
    mod.add_argument("--modular-radial-brake", type=float, default=1.0,
                     dest="modular_radial_brake",
                     help="Radial brake soft-limiting scale for parameter norm expansion (NVIDIA RadialBrakeHook, e.g. 0.8; 1.0=disabled)")

    # ---- Variance-Aware Dynamic Dampening (VADD) ----
    vadd = p.add_argument_group("Variance-Aware Dynamic Dampening (VADD)")
    vadd.add_argument("--var-dampening", type=float, default=None,
                      dest="var_dampening",
                      help="Master latent variance threshold (e.g. 1.20). Activates loss barrier & optimizer step dampening.")
    vadd.add_argument("--var-barrier-weight", type=float, default=0.1,
                      dest="var_barrier_weight",
                      help="Weight lambda for quadratic hinge barrier loss (default: 0.1)")
    vadd.add_argument("--var-damp-opt", action="store_true", default=True,
                      dest="var_damp_opt",
                      help="Enable optimizer step & EV dampening when variance exceeds threshold (default: True)")
    vadd.add_argument("--var-no-damp-opt", action="store_false",
                      dest="var_damp_opt",
                      help="Disable optimizer step dampening")
    vadd.add_argument("--var-damp-power", type=float, default=1.0,
                      dest="var_damp_power",
                      help="Power p for optimizer dampening factor (tau/sigma)^p (default: 1.0)")
    vadd.add_argument("--var-wd-boost", type=float, default=0.0,
                      dest="var_wd_boost",
                      help="Weight decay boost factor during high-variance episodes (default: 0.0)")
    vadd.add_argument("--demo-cfg-rescale", type=float, default=0.0,
                      dest="demo_cfg_rescale",
                      help="CFG rescale phi for demo generation (e.g. 0.7; 0.0=disabled)")
    vadd.add_argument("--demo-latent-clamp", type=float, default=None,
                      dest="demo_latent_clamp",
                      help="Clamp generated demo latent std to this ceiling before VAE decode (e.g. 1.20)")

    # ---- Training ----
    p.add_argument("--epochs", type=int, default=None,
                   help="Train for N epochs (overrides --steps; uses Trainer max_epochs).")
    p.add_argument("--steps", type=int, default=10000, help="Max training steps")
    p.add_argument("--batch_size", "--batch-size", type=int, default=8, dest="batch_size")
    p.add_argument("--accumulate_grad_batches", "--accumulate-grad-batches",
                   type=int, default=1, dest="accumulate_grad_batches")
    p.add_argument("--gradient_clip_val", type=float, default=1.0)
    p.add_argument("--base_precision", type=str, default="bf16",
                   choices=list(PRECISION_MODES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", "--num-workers", type=int, default=8,
                   dest="num_workers", help="DataLoader workers (default: 8)")
    p.add_argument("--resume_ckpt", type=str, default=None,
                   help="Full Lightning checkpoint to resume from")

    # ---- Logging & Evaluation ----
    p.add_argument("--logger", type=str, default=None,
                   choices=["wandb", "csv", None])
    p.add_argument("--wandb-project", "--wandb_project", type=str, default=None,
                   dest="wandb_project", help="W&B project name (overrides WANDB_PROJECT env var)")
    p.add_argument("--name", type=str, default="modular_test")
    p.add_argument("--save_dir", "--output-dir", "--output_dir", type=str, default=None,
                   dest="save_dir")
    p.add_argument("--checkpoint_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--demo_every", type=int, default=2000)
    p.add_argument("--no_demos", action="store_true", default=False)
    p.add_argument("--eval_demos", action="store_true", default=False,
                   help="Render milestone (step 100) and between-epoch demos with async CPU VAE")
    p.add_argument("--eval_milestones", type=int, nargs="*", default=[100, 500, 1000, 2000, 3000],
                   help="Steps at which to render canonical demo suites (default: 100 500 1000 2000 3000)")
    p.add_argument("--eval_continuations", action="store_true", default=False,
                   help="Also render long continuations (ext512, ext303) during eval demos (default: False)")
    p.add_argument("--eval_num_prompts", type=int, default=3,
                   help="Number of canonical prompts to render during eval (default: 3)")
    p.add_argument("--loss_guard_threshold", type=float, default=1.0,
                   help="Mean epoch loss threshold to trigger abort and emergency save (default: 1.0)")

    args = p.parse_args()

    if not args.encoded_dir and not args.subset300:
        p.error("Must specify --encoded_dir or --subset300")

    train(args)


if __name__ == "__main__":
    main()
