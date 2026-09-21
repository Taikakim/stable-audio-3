import math
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CometLogger
import os
import torch
import gc
import typing as tp
import torchaudio

from einops import rearrange
from safetensors.torch import save_file
from functools import partial
from torch.nn import functional as F

from ..interface.aeiou import audio_spectrogram_image
from ..inference.sampling import truncated_logistic_normal_rescaled, sample_timesteps_logsnr, sample_timesteps_logsnr_uniform, sample_diffusion
from ..models.diffusion import ConditionedDiffusionModelWrapper
from ..models.inpainting import random_inpaint_mask, MaskType
from ..models.lora import add_lora, get_lora_params, get_lora_state_dict, LoRAParametrization, get_lora_layers, save_lora_safetensors, resolve_adapter_type, prepare_dora_state_dict, cast_base_to_precision
from .utils import create_optimizer_from_config, create_scheduler_from_config, log_audio, log_image, log_metric, get_rank, create_augmented_padding_mask, compute_masked_loss, compute_normalized_mse, resize_padding_mask, StaggeredLogger, compute_per_elem_trim, trim_and_concat
from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

def get_alphas_sigmas(t):
    """Cosine noise schedule for the v-prediction objective (t in [0,1]: t=0 clean, t=1 noise;
    noised = x0*alpha + noise*sigma, v-target = noise*alpha - x0*sigma). This was CALLED by
    validation_step (and needed by training_step) but DEFINED nowhere in this fork — the upstream
    'v' objective (the factory default!) shipped gutted, so any v-pred run crashed with NameError.
    Restored 2026-08-03 (additive; see stable-audio-3/CLAUDE.md). Needed by #65/#66 v-pred work."""
    return torch.cos(t * torch.pi / 2), torch.sin(t * torch.pi / 2)


class SimpleEMA(torch.nn.Module):
    """Minimal self-contained weight-EMA (ema_pytorch was stripped from this SA3 fork; the
    wrapper's .ema_model reads survived but the build+update did not). Shadows `model`'s params
    with an exponential moving average. `.ema_model` is the shadow used for demos/eval/save.
    Registered as a submodule so it round-trips through the Lightning checkpoint automatically.
    Full-finetune only (LoRA keeps EMA off — see wrapper __init__)."""

    def __init__(self, model, beta=0.9999, update_every=1, update_after_step=0):
        super().__init__()
        import copy
        self.ema_model = copy.deepcopy(model).eval()
        self.ema_model.requires_grad_(False)
        self.beta = float(beta)
        self.update_every = int(update_every)
        self.update_after_step = int(update_after_step)
        self.register_buffer("ema_step", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, model):
        self.ema_step += 1
        step = int(self.ema_step.item())
        # Warmup: hard-track the online weights so the EMA starts from a real point, not init.
        if step <= self.update_after_step:
            for ep, mp in zip(self.ema_model.parameters(), model.parameters()):
                ep.copy_(mp.detach())
            for eb, mb in zip(self.ema_model.buffers(), model.buffers()):
                eb.copy_(mb)
            return
        if self.update_every > 1 and step % self.update_every != 0:
            return
        for ep, mp in zip(self.ema_model.parameters(), model.parameters()):
            ep.lerp_(mp.detach().to(ep.dtype), 1.0 - self.beta)
        for eb, mb in zip(self.ema_model.buffers(), model.buffers()):
            eb.copy_(mb)


@torch.no_grad()
def agc_clip_grads_(params, agc_lambda: float, eps: float = 1e-3, eps2: float = 1e-6) -> None:
    """Adaptive Gradient Clipping (NFNets / Brock et al.), in-place on .grad.

    For each param p (ndim>=2) with grad g, per OUTPUT-UNIT (row = dim 0; reduce over all
    other dims, keepdim) scale the gradient toward the param's own scale:

        pn   = clamp(||p||_unit, min=eps)          # unit weight norm, floored
        gn   = ||g||_unit                          # unit grad norm
        coef = min(1, agc_lambda * pn / (gn+eps2))  # per-unit shrink factor
        g   *= coef

    Only large-grad-relative-to-scale units are reined in; small-weight/near-noise units keep
    moving. ndim<2 params (biases, norms, scalars) are SKIPPED (left untouched — the caller's
    norm-mode default handles them if desired). Norms are computed in fp32 for stability, then
    the coefficient is cast back to the grad dtype. Mutates p.grad in place; returns nothing.
    """
    for p in params:
        g = getattr(p, "grad", None)
        if g is None or p.ndim < 2:
            continue
        dims = tuple(range(1, p.ndim))
        pn = p.detach().float().pow(2).sum(dim=dims, keepdim=True).sqrt().clamp_min(eps)
        gn = g.detach().float().pow(2).sum(dim=dims, keepdim=True).sqrt()
        coef = (agc_lambda * pn / (gn + eps2)).clamp_max(1.0)
        g.mul_(coef.to(g.dtype))


class DiffusionCondTrainingWrapper(pl.LightningModule):
    '''
    Wrapper for training a conditional audio diffusion model.
    '''
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_loss_weight: float = 0.0,
            mask_padding_attention: bool = False,
            silence_extension_scale_seconds: float = 0.0,
            use_ema: bool = True,
            ema_beta: float = 0.9999,
            ema_update_every: int = 1,
            ema_update_after_step: int = 0,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal", "log_snr", "log_snr_uniform"] = "uniform",
            timestep_sampler_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            p_one_shot: float = 0.0,
            inpainting_config: dict = None,
            use_effective_length_for_schedule: bool = False,
            sample_rate: int = 44100,
            sample_size: int = None,
            loss_normalization: tp.Literal["none", "timestep", "sample", "sample_channel"] = "none",
            loss_norm_eps: float = 1e-6,
            lora_config: tp.Optional[tp.Dict[str, tp.Any]] = None,
            lora_state_dict: tp.Optional[tp.Dict[str, tp.Any]] = None,
            svd_bases_path: tp.Optional[str] = None,
            log_every_n_steps: int = 10,
            ot_coupling: bool = False,
            base_precision: tp.Optional[str] = None,
            familiarity_beta: float = 0.0,
            stereo_loss_weight: float = 0.0,
            stereo_loss_tmax: float = 0.3,
            stereo_loss_subbatch: int = 2,
            subspace_loss_basis: str = None,
            subspace_loss_weight: float = 1.0,
            subspace_loss_tgate: str = None,
            subspace_loss_tgate_mode: str = "r2",
            subspace_loss_tgate_floor: float = 0.05,
            x0_equiv_loss: bool = False,
            x0_loss_weight: float = 0.0,
            grad_clip_mode: str = "norm",
            agc_lambda: float = 0.01,
            output_std_penalty: float = 0.0,
            output_std_t_gate: float = 0.5,
            latent_var_barrier: float = 0.0,
            latent_var_weight: float = 0.0,
    ):
        super().__init__()

        self.ot_coupling = ot_coupling

        # Full-FT regularization A/B (2026-08-10, spec fullft-regularization-ab). Two levers
        # for the FusionOpt full-FT latent-scale runaway, both OFF by default (byte-identical):
        #   - grad_clip_mode "agc": scale-relative per-output-unit gradient clipping in
        #     configure_gradient_clipping (NFNets AGC) instead of the global-norm clip. "norm"
        #     (default) keeps Lightning's --gradient_clip_val behaviour.
        #   - output_std_penalty (lambda_out) > 0: penalize the reconstructed clean latent's
        #     per-channel std drifting off the real data's, gated to t < output_std_t_gate.
        self.grad_clip_mode = grad_clip_mode
        self.agc_lambda = agc_lambda
        self.output_std_penalty = output_std_penalty
        self.output_std_t_gate = output_std_t_gate

        # Variance-Aware Dynamic Dampening (VADD): quadratic hinge barrier on reconstructed latent std
        self.latent_var_barrier = latent_var_barrier
        self.latent_var_weight = latent_var_weight
        self.running_latent_std = 1.13

        # Stereo-preservation auxiliary loss (training/stereo_loss.py). weight=0
        # (default) => OFF and the training path is byte-identical to before.
        # See the module docstring + MASTER §4 meter-in-the-gradient note.
        self.stereo_loss_weight = stereo_loss_weight
        self.stereo_loss_tmax = stereo_loss_tmax
        self.stereo_loss_subbatch = stereo_loss_subbatch

        # Familiarity-normalized loss weighting (scripts/familiarity.py): per-crop
        # EMA of relative loss -> down-weight familiar crops, keep remote ones hot.
        # beta=0 (default) = off. State is per-run (not checkpointed).
        self.familiarity = None
        if familiarity_beta and familiarity_beta > 0:
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "scripts"))
            from familiarity import FamiliarityReweighter
            self.familiarity = FamiliarityReweighter(beta=familiarity_beta)

        # Subspace-weighted RF loss (spectral-bias counter, tau ~ 1/lambda —
        # arXiv 2503.03206): upweight the flow-matching error's component inside a
        # measured latent subspace (e.g. the 15-dim melody subspace, 9.3% of corpus
        # variance) so low-variance structure isn't gradient-starved. weight=1.0
        # (default) => OFF, path byte-identical. Basis npz: rows of `basis15` (or the
        # first k rows of `melody_basis`) = orthonormal components [k, C]. The extra
        # term is computed on the RAW error (designed for loss_normalization="none").
        # E1a: sigma^2-weighted v-loss == x0-space MSE for a v-output head (JLT port,
        # spec 2026-07-31). False (default) = byte-identical path.
        self.x0_equiv_loss = x0_equiv_loss

        # x0-reconstruction ADD-a-term (distinct from x0_equiv_loss above, which
        # REPLACES the v-loss with its sigma^2-weighted form). This ADDS an explicit
        # clean-latent MSE term: lambda_x0 * MSE(z0_hat, z0), with z0_hat = noised -
        # t*v_pred (rf_z0_hat, the SAME reconstruction the output-std/stereo blocks
        # use). Predicting toward the clean latent removes the ambient isotropic-noise
        # component that pressures output-scale growth (arXiv 2605.27102 JLT; the
        # x0equiv thread). weight==0 (default) => NO-OP, loss byte-identical, zero
        # overhead. RF objective only (z0_hat formula is RF-specific).
        self.x0_loss_weight = x0_loss_weight

        self.subspace_loss_weight = subspace_loss_weight
        if subspace_loss_basis is not None and subspace_loss_weight != 1.0:
            import numpy as _np
            _z = _np.load(subspace_loss_basis)
            _b = _z["basis15"] if "basis15" in _z.files else _z["melody_basis"][:15]
            self.register_buffer(
                "subspace_basis", torch.tensor(_b, dtype=torch.float32),
                persistent=False)
        else:
            self.subspace_basis = None
        # R²(t)-derived gate on the subspace term (C 2026-08-19; stable_audio_3/training/tgate.py):
        # multiplies the per-sample subspace error energy by g(t), mean-1 over the measured grid,
        # so K stays the AVERAGE multiplier and the budget moves to the noise levels where the
        # melody subspace is actually recoverable. None (default) = flat gate = byte-identical.
        self.subspace_tgate = None
        if self.subspace_basis is not None and subspace_loss_tgate:
            from .tgate import load_tgate
            _gt, _gg = load_tgate(subspace_loss_tgate, mode=subspace_loss_tgate_mode,
                                  floor=subspace_loss_tgate_floor)
            self.register_buffer("subspace_tgate_t", _gt, persistent=False)
            self.register_buffer("subspace_tgate_g", _gg, persistent=False)
            self.subspace_tgate = subspace_loss_tgate
            print(f"[subspace] t-gate {subspace_loss_tgate_mode} from {subspace_loss_tgate}: "
                  f"grid {_gt.numel()} pts, gate range {float(_gg.min()):.2f}-{float(_gg.max()):.2f}")

        self.diffusion = model

        self.lora_config = lora_config
        if self.lora_config is not None:
            # Don't use EMA with LoRA
            use_ema = False
            # Freeze the pre-trained model weights
            self.diffusion.model.eval().requires_grad_(False)
            self.diffusion.conditioner.eval().requires_grad_(False)
            rank = self.lora_config.get("rank", 8)
            lora_alpha = self.lora_config.get("alpha", rank)
            adapter_type = self.lora_config.get("adapter_type", "lora")
            include = self.lora_config.get("include", None)
            exclude = self.lora_config.get("exclude", None)
            phm_n = self.lora_config.get("phm_n", 4)
            # Resolve legacy "dora" to rows/cols variant
            adapter_type = resolve_adapter_type(adapter_type, lora_state_dict)
            print(f"LoRA config: rank={rank}, alpha={lora_alpha}, adapter_type={adapter_type}")
            if include:
                print(f"  include: {include}")
            if exclude:
                print(f"  exclude: {exclude}")
            # Load pre-computed SVD bases for -XS adapter types
            svd_bases = None
            if svd_bases_path is not None:
                print(f"Loading SVD bases from {svd_bases_path}")
                svd_bases = torch.load(svd_bases_path, map_location="cpu", weights_only=True)
            elif adapter_type.endswith("-xs"):
                print("WARNING: -XS adapter without svd_bases_path — SVD will be computed per layer")
            lora_config = {
                torch.nn.Linear: {
                    "weight": partial(LoRAParametrization.from_linear, rank=rank, lora_alpha=lora_alpha, adapter_type=adapter_type, phm_n=phm_n),
                },
                torch.nn.Conv1d: {
                    "weight": partial(LoRAParametrization.from_conv1d, rank=rank, lora_alpha=lora_alpha, adapter_type=adapter_type, phm_n=phm_n),
                }
            }
            # Add LoRA to the model
            add_lora(self.diffusion.model, lora_config, include=include, exclude=exclude, svd_bases=svd_bases)
            # Add LoRA to the conditioner
            add_lora(self.diffusion.conditioner, lora_config, include=include, exclude=exclude, svd_bases=svd_bases)
            print("lora layers:", len(get_lora_layers(self.diffusion)))

            if lora_state_dict is not None:
                # Old DoRA checkpoints saved magnitude as 2D (1,fan_in) or (fan_out,1);
                # current code expects 1D. Squeeze so old checkpoints still load.
                prepare_dora_state_dict(lora_state_dict)
                self.diffusion.model.load_state_dict(lora_state_dict, strict=False)
                self.diffusion.conditioner.load_state_dict(lora_state_dict, strict=False)

            # Cast frozen base weights to lower precision if requested
            if base_precision:
                cast_base_to_precision(self.diffusion.model, base_precision)
                cast_base_to_precision(self.diffusion.conditioner, base_precision)
                if self.diffusion.pretransform is not None:
                    self.diffusion.pretransform.to(
                        torch.bfloat16 if base_precision in ("bf16", "bfloat16") else torch.float16
                    )

        self.diffusion_ema = None
        if use_ema:
            # Full-finetune only (LoRA set use_ema=False above). Shadows the DiT weights; the
            # demo/eval/save hooks already prefer diffusion_ema.ema_model when it's not None.
            self.diffusion_ema = SimpleEMA(
                self.diffusion.model, beta=ema_beta,
                update_every=ema_update_every, update_after_step=ema_update_after_step,
            )
            print(f"[ema] SimpleEMA on the DiT: beta={ema_beta} update_every={ema_update_every} "
                  f"warmup={ema_update_after_step}")
        self.mask_loss_weight = mask_loss_weight

        # Attention masking for padded tokens
        # Backward compat: if passed from training config, propagate to model
        if mask_padding_attention and not self.diffusion.mask_padding_attention:
            import warnings
            warnings.warn("mask_padding_attention in training config is deprecated. Move to model.diffusion config.", FutureWarning)
            self.diffusion.mask_padding_attention = mask_padding_attention
        self.mask_padding_attention = self.diffusion.mask_padding_attention
        self.silence_extension_scale_seconds = silence_extension_scale_seconds

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler     

        self.timestep_sampler_options = {} if timestep_sampler_options is None else timestep_sampler_options

        if self.timestep_sampler == "log_snr":
            self.mean_logsnr = self.timestep_sampler_options.get("mean_logsnr", -1.2)
            self.std_logsnr = self.timestep_sampler_options.get("std_logsnr", 2.0)
        elif self.timestep_sampler == "log_snr_uniform":
            self.min_logsnr = self.timestep_sampler_options.get("min_logsnr", -6.0)
            self.max_logsnr = self.timestep_sampler_options.get("max_logsnr", 5.0)

        self.p_one_shot = p_one_shot

        self.diffusion_objective = model.diffusion_objective

        self.log_loss_info = log_loss_info

        self._staggered_logger = StaggeredLogger(every_n_steps=log_every_n_steps)

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        # Loss normalization by target magnitude
        # Options: "none", "timestep", "sample", "sample_channel"
        self.loss_normalization = loss_normalization
        self.loss_norm_eps = loss_norm_eps

        # Inpainting
        self.inpainting_config = inpainting_config
        
        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})

        # Per-element schedule shift based on effective (unpadded) sequence length
        # Backward compat: if passed from training config, propagate to model
        if use_effective_length_for_schedule and not self.diffusion.use_effective_length_for_schedule:
            import warnings
            warnings.warn("use_effective_length_for_schedule in training config is deprecated. Move to model.diffusion config.", DeprecationWarning)
            self.diffusion.use_effective_length_for_schedule = use_effective_length_for_schedule
        self.use_effective_length_for_schedule = self.diffusion.use_effective_length_for_schedule
        self.sample_rate = sample_rate
        self.sample_size = sample_size

        # FSDP
        self.use_fsdp = False

        # Validation
        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_type = diffusion_opt_config['optimizer'].get('type')

        if opt_type == 'FusionOpt':
            # FusionOpt routes 2D matrices (min(shape) >= 128) to the spectral
            # Muon+MONA+KL-Shampoo path and everything else to a ScheduleFree-AdamW
            # scalar path. NOTE: the min(shape)>=128 threshold was designed for the
            # 5M-param LatCH heads (design doc §2); SA3's diffusion backbone is
            # much larger, but the threshold still partitions cleanly. Use the
            # optimizer config's `force_scalar` (passed through `param_groups`) or
            # `spectral_lr`/`scalar_lr` to tune per-group LR if desired.
            #
            # build_fusion_param_groups filters to requires_grad params, so under
            # LoRA it captures EXACTLY the trainable LoRA adapters and routes them
            # into spectral/scalar groups (the frozen base is excluded). We route
            # over the whole self.diffusion so conditioner LoRA params are included
            # too; for a full (non-LoRA) finetune we keep the original model-only
            # root.
            from stable_audio_tools.training.fusion_groups import (
                build_fusion_param_groups, summarise_groups,
            )
            pg_cfg = diffusion_opt_config['optimizer'].get('param_groups', {}) or {}
            route_root = self.diffusion if self.lora_config is not None else self.diffusion.model
            opt_params = build_fusion_param_groups(route_root, **pg_cfg)
            if get_rank() == 0:
                print("FusionOpt param groups:")
                print(summarise_groups(opt_params))
        elif self.lora_config is not None:
            opt_params = [*get_lora_params(self.diffusion.model), *get_lora_params(self.diffusion.conditioner)]
            # B7 mir_ctrl (2026-08-21): modular local-cond projections are trainable
            # non-LoRA params installed post-load; get_lora_params cannot see them,
            # and __init__'s LoRA freeze cleared their requires_grad (train_lora
            # re-enables after wrapper construction). Without this append they are
            # silently excluded and stay exactly zero-init — caught by the in-training
            # control-ablation meter (gain pinned at 0.0000, proj weights bit-zero).
            opt_params += [p for n, p in self.diffusion.model.named_parameters()
                           if "modular_local_embeds" in n and p.requires_grad]
        elif opt_type == 'MuonAdamW':
            # Pass (name, param) tuples so MuonAdamW can match fused layer patterns
            opt_params = [(n, p) for n, p in self.diffusion.named_parameters() if p.requires_grad]
        else:
            # Only include parameters that require gradients (excludes frozen pretransform, conditioner, etc.)
            opt_params = [p for p in self.diffusion.parameters() if p.requires_grad]

        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], opt_params)

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None,
                                    gradient_clip_algorithm=None):
        """Full-FT regularization A/B (2026-08-10). FusionOpt is a custom optimizer, so
        Lightning's built-in AGC path isn't wired — override here.

        grad_clip_mode == "agc": scale-relative Adaptive Gradient Clipping (per output-unit,
        relative to the unit's own weight norm) via agc_clip_grads_, replacing the global-norm
        clip. Runs after backward / before optimizer.step (grads populated; bf16-mixed has no
        GradScaler so grads are true-scale here). ndim<2 params are left to the default below.

        grad_clip_mode == "norm" (default): the stock Lightning behaviour — honours
        --gradient_clip_val (None/0 => no-op), so existing runs are byte-identical."""
        if self.grad_clip_mode == "agc":
            for group in optimizer.param_groups:
                agc_clip_grads_(group["params"], self.agc_lambda)
            return
        super().configure_gradient_clipping(
            optimizer, gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        p.tick("setup")

        #with torch.amp.autocast(device_type="cuda"):
        conditioning = self.diffusion.conditioner(metadata, self.device)

        # Create batch tensor of padding masks from the metadata
        # If padding_mask not provided, assume all positions are valid (no padding)
        if all("padding_mask" in md for md in metadata):
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device)  # Shape (batch_size, sequence_length)
        else:
            # All-True mask: everything is signal, no padding
            padding_masks = torch.ones(diffusion_input.shape[0], diffusion_input.shape[-1], dtype=torch.bool, device=self.device)

        p.tick("conditioning")

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast(), torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    p.tick("pretransform")
                    padding_masks = resize_padding_mask(padding_masks, diffusion_input.shape[-1])
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale


                if padding_masks.shape[-1] != diffusion_input.shape[-1]:
                    padding_masks = resize_padding_mask(padding_masks, diffusion_input.shape[-1])

        if self.timestep_sampler == "uniform":
            # Draw uniformly distributed continuous timesteps
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
        elif self.timestep_sampler == "trunc_logit_normal":
            # Draw from logistic truncated normal distribution
            t = truncated_logistic_normal_rescaled(reals.shape[0]).to(self.device)

            # Flip the distribution
            t = 1 - t
        elif self.timestep_sampler == "log_snr":
            t = sample_timesteps_logsnr(reals.shape[0], mean_logsnr=self.mean_logsnr, std_logsnr=self.std_logsnr).to(self.device)
        elif self.timestep_sampler == "log_snr_uniform":
            t = sample_timesteps_logsnr_uniform(reals.shape[0], min_logsnr=self.min_logsnr, max_logsnr=self.max_logsnr).to(self.device)
        else:
            raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

        if self.diffusion.dist_shift is not None:
            # Compute sequence length for schedule shift
            if self.use_effective_length_for_schedule:
                # Use per-element effective lengths derived from seconds_total (rounded up)
                # This matches inference which computes effective length from seconds_total conditioning
                # Fall back to padding_masks.sum() if seconds_total is not available
                if all("seconds_total" in md for md in metadata):
                    downsampling_ratio = self.diffusion.pretransform.downsampling_ratio if self.diffusion.pretransform is not None else 1
                    effective_seq_len = torch.tensor(
                        [int(math.ceil(int(md["seconds_total"] * self.sample_rate) / downsampling_ratio)) for md in metadata],
                        device=self.device
                    )
                else:
                    # Fallback: use padding mask sum
                    effective_seq_len = padding_masks.sum(dim=-1)
            else:
                # Use total sequence length (original behavior)
                effective_seq_len = diffusion_input.shape[2]
            
            # Shift the distribution
            t = self.diffusion.dist_shift.shift(t, effective_seq_len)

        if self.p_one_shot > 0:
            # Set t to 1 with probability p_one_shot
            t = torch.where(torch.rand_like(t) < self.p_one_shot, torch.ones_like(t), t)

        # Calculate the noise schedule parameters for those timesteps
        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)     # was missing -> NameError on the 'v' default (upstream-gutted)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            alphas, sigmas = 1-t, t

        # Combine the ground truth data and the noise
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)

        # Minibatch OT coupling: find optimal noise permutation for straighter transport paths
        # Based on MelodyFlow (arXiv:2407.03648v2) Section 2.5.2
        # Uses GPU-only Sinkhorn approximation to avoid CPU sync
        if self.ot_coupling and diffusion_input.shape[0] > 1:
            with torch.no_grad():
                # Flatten to [batch, features] for distance computation
                data_flat = diffusion_input.reshape(diffusion_input.shape[0], -1)
                noise_flat = noise.reshape(noise.shape[0], -1)
                # Squared L2 cost via matmul (faster than cdist, same optimal assignment)
                aa = (data_flat * data_flat).sum(dim=1, keepdim=True)
                bb = (noise_flat * noise_flat).sum(dim=1, keepdim=True)
                cost_matrix = aa + bb.T - 2.0 * (data_flat @ noise_flat.T)
                # Sinkhorn assignment (GPU-only, no CPU sync)
                log_P = -cost_matrix / cost_matrix.detach().mean() # normalize for numerical stability
                for _ in range(20):
                    log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
                    log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
                # Sequential assignment from soft permutation matrix (guarantees valid permutation)
                P = log_P.exp()
                B = P.shape[0]
                col_indices = torch.empty(B, dtype=torch.long, device=P.device)
                used = torch.zeros(B, dtype=torch.bool, device=P.device)
                for i in range(B):
                    P[i, used] = -1
                    col_indices[i] = P[i].argmax()
                    used[col_indices[i]] = True
                noise = noise[col_indices]

        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            targets = noise - diffusion_input

        p.tick("noise")

        extra_args = {}

        # Compute downsampling ratio for attention mask creation
        downsampling_ratio = self.diffusion.pretransform.downsampling_ratio if self.diffusion.pretransform is not None else 1

        # Create augmented padding mask with random silence extension
        if self.mask_padding_attention and self.silence_extension_scale_seconds > 0:
            augmented_padding_mask = create_augmented_padding_mask(
                padding_masks,
                silence_extension_scale_seconds=self.silence_extension_scale_seconds,
                sample_rate=self.sample_rate,
                downsampling_ratio=downsampling_ratio,
            )
        else:
            augmented_padding_mask = padding_masks

        # Loss mask defines signal vs padding regions for loss computation
        # - mask_loss_weight controls padding contribution (0 = signal only)
        # - When mask_padding_attention=True: only compute loss on signal (padding saw no attention)
        loss_mask = augmented_padding_mask.to(torch.bool)

        # Pass padding mask for attention masking - model handles prepend extension
        if self.mask_padding_attention:
            extra_args["padding_mask"] = augmented_padding_mask

        # ARC-Forcing: batches whose metadata carry a precomputed clamp mask +
        # masked input (ArcRolloutDataset) ship their OWN inpaint conditioning —
        # the model's drifted rollout context — routed through the same
        # conditioning keys inference uses ('inpaint_mask'/'inpaint_masked_input'
        # -> local_add_cond). Takes precedence over random_inpaint_mask.
        batch_inpaint = all(
            "inpaint_mask" in md and "inpaint_masked_input" in md for md in metadata
        )
        if batch_inpaint:
            inpaint_mask = torch.stack(
                [md["inpaint_mask"][0] for md in metadata], dim=0
            ).to(self.device, dtype=diffusion_input.dtype)  # (B, 1, T)
            inpaint_masked_input = torch.stack(
                [md["inpaint_masked_input"][0] for md in metadata], dim=0
            ).to(self.device, dtype=diffusion_input.dtype)  # (B, C, T)

            # Keep the visible context in the same (scaled) latent space as diffusion_input
            if (self.pre_encoded and self.diffusion.pretransform is not None
                    and hasattr(self.diffusion.pretransform, "scale")
                    and self.diffusion.pretransform.scale != 1.0):
                inpaint_masked_input = inpaint_masked_input / self.diffusion.pretransform.scale

            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            # Loss on the free (mask=0) region ONLY — the clamped frames are
            # drifted rollout latents, not reconstruction targets, so no context
            # loss either (drift-recovery objective; see guard below)
            loss_mask = loss_mask & ~inpaint_mask.squeeze(1).to(torch.bool)

        elif self.inpainting_config is not None:

            # Max mask size is the full sequence length
            max_mask_length = diffusion_input.shape[2]

            # Create a mask of random length for a random slice of the input
            inpaint_masked_input, inpaint_mask = random_inpaint_mask(diffusion_input, padding_masks=augmented_padding_mask, mask_padding=self.mask_padding_attention, **self.inpaint_mask_kwargs)

            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

            # Only compute loss on inpainted region (where model is generating)
            loss_mask = loss_mask & ~inpaint_mask.squeeze(1).to(torch.bool)

        output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = self.cfg_dropout_prob, **extra_args)
        p.tick("diffusion")

        if self.log_loss_info:
            # Loss debugging logs
            num_loss_buckets = 10
            bucket_size = 1 / num_loss_buckets
            loss_all = F.mse_loss(output, targets, reduction="none")

            sigmas = rearrange(self.all_gather(sigmas), "w b c n -> (w b) c n").squeeze()

            # gather loss_all across all GPUs
            loss_all = rearrange(self.all_gather(loss_all), "w b c n -> (w b) c n")

            # Bucket loss values based on corresponding sigma values, bucketing sigma values by bucket_size
            loss_all = torch.stack([loss_all[(sigmas >= i) & (sigmas < i + bucket_size)].mean() for i in torch.arange(0, 1, bucket_size).to(self.device)])

            # Log bucketed losses with corresponding sigma bucket values, if it's not NaN
            debug_log_dict = {
                f"model/loss_all_{i/num_loss_buckets:.1f}": loss_all[i].detach() for i in range(num_loss_buckets) if not torch.isnan(loss_all[i])
            }

            self.log_dict(debug_log_dict)

        p.tick("loss_debug")

        # Compute std only over non-padded positions when masking is active
        if loss_mask is not None and self.mask_padding_attention:
            mask_expanded = loss_mask.unsqueeze(1)  # [B, 1, T]
            std_data = diffusion_input[mask_expanded.expand_as(diffusion_input)].std()
            std_targets = targets[mask_expanded.expand_as(targets)].std().detach()
        else:
            std_data = diffusion_input.std()
            std_targets = targets.std().detach()

        log_dict = {
            'train/std_data': std_data,
            'train/std_targets': std_targets,
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        p.tick("std_compute")

        # Compute normalized MSE (normalization only affects non-"none" modes)
        mse_loss_full = compute_normalized_mse(output, targets, loss_mask, self.loss_normalization, self.loss_norm_eps)

        # Familiarity-normalized weighting: scale each sample's loss surface by its
        # crop's (past-visits) familiarity weight BEFORE the masked/context
        # reductions, so every downstream term inherits the weighting consistently.
        if self.familiarity is not None:
            with torch.no_grad():
                _mask = loss_mask.unsqueeze(1).to(mse_loss_full.dtype)
                _denom = (_mask.sum(dim=(1, 2)) * mse_loss_full.shape[1]).clamp_min(1.0)
                _per = (mse_loss_full.detach() * _mask).sum(dim=(1, 2)) / _denom
            _ids = [md.get("latent_filename", md.get("path", f"idx{j}"))
                    for j, md in enumerate(metadata)]
            _w = self.familiarity.weights(_ids, _per.tolist())
            _w_t = torch.tensor(_w, device=mse_loss_full.device, dtype=mse_loss_full.dtype)
            mse_loss_full = mse_loss_full * _w_t[:, None, None]
            log_dict["train/familiarity_w_min"] = float(min(_w))
            log_dict["train/familiarity_w_max"] = float(max(_w))

        p.tick("mse_loss")

        # Compute loss with signal/padding separation (returns already-detached metrics)
        loss, signal_mean, padding_mean = compute_masked_loss(
            mse_loss_full, loss_mask, self.mask_padding_attention, self.mask_loss_weight
        )
        mse_loss = loss

        # x0-equivalent loss weighting (E1a, spec 2026-07-31-reality-structured-model-
        # experiments.md): with a v-output head, an x0-space MSE is EXACTLY a sigma^2-
        # weighted v-loss (x_hat = z_t - t*v_hat is linear in v_hat with factor -t).
        # This tests JLT's (arXiv 2605.27102) loss-geometry claim with ZERO inference
        # changes; the E1 pre-test measured the v-trained base recovering low-variance
        # eigendirections 2-8x worse per unit signal. weight OFF (False) = byte-identical.
        if self.x0_equiv_loss:
            _s2 = (sigmas.squeeze(-1).squeeze(-1) ** 2)[:, None, None].to(loss.dtype)
            # renormalize so the EXPECTED loss scale matches plain v-loss (E[t^2]=1/3
            # under uniform t) — keeps lr/optimizer tuning comparable across arms
            x0_loss, _sig, _pad = compute_masked_loss(
                mse_loss_full * _s2 * 3.0, loss_mask, self.mask_padding_attention,
                self.mask_loss_weight)
            loss = x0_loss
            log_dict["train/x0equiv_loss"] = x0_loss.detach()

        # Subspace-weighted RF loss (see __init__): add (K-1)x the error energy inside
        # the measured subspace, masked like the main loss and scaled per-element (/C)
        # so K literally multiplies that subspace's share of the MSE.
        if self.subspace_basis is not None:
            err32 = (output - targets).float()                       # [B, C, T]
            proj = torch.einsum("kc,bct->bkt", self.subspace_basis, err32)
            sub_energy = proj.pow(2).sum(dim=1)                      # [B, T]
            if self.subspace_tgate is not None:
                from .tgate import interp_gate
                _tg = interp_gate(t.detach().float().reshape(-1), self.subspace_tgate_t,
                                  self.subspace_tgate_g).to(sub_energy.dtype)   # [B]
                sub_energy = sub_energy * _tg[:, None]
                log_dict["train/subspace_tgate"] = _tg.mean().detach()
            if loss_mask is not None and self.mask_padding_attention:
                _m = loss_mask.to(sub_energy.dtype)
                sub_mean = (sub_energy * _m).sum() / (_m.sum().clamp_min(1.0) * output.shape[1])
            else:
                sub_mean = sub_energy.mean() / output.shape[1]
            loss = loss + (self.subspace_loss_weight - 1.0) * sub_mean
            log_dict["train/subspace_loss"] = sub_mean.detach()

        p.tick("masked_loss")

        # When attention masking is on, compute_masked_loss excludes everything outside
        # loss_mask (which now excludes inpaint context). Add context reconstruction loss
        # so the model learns to preserve context regions during inpainting.
        # (When mask_padding_attention=False, context is already included via mask_loss_weight.)
        # Skipped for ARC batch-supplied masks: their clamped frames are drifted
        # rollout latents, not ground truth to reconstruct.
        context_loss_mean = torch.tensor(0.0, device=loss.device)
        if (self.inpainting_config is not None
                and not batch_inpaint
                and self.mask_padding_attention
                and self.mask_loss_weight > 0):
            # Context = inpaint_mask=1 (keep) AND padding_mask=1 (real audio, not padding)
            inpaint_context = inpaint_mask.squeeze(1).to(torch.bool) & augmented_padding_mask.to(torch.bool)
            n_ctx = inpaint_context.sum(dim=1) * mse_loss_full.shape[1]  # per-sample count
            if n_ctx.sum() > 0:
                context_vals = torch.where(inpaint_context.unsqueeze(1), mse_loss_full, 0.0)
                context_loss_mean = (context_vals.sum(dim=(1, 2)) / (n_ctx + 1e-8)).mean()
                loss = loss + context_loss_mean * self.mask_loss_weight

        # Log separate signal/padding/context losses for monitoring
        log_dict["train/mse_signal"] = signal_mean
        log_dict["train/mse_masked_loss"] = padding_mean
        log_dict["train/mse_context_loss"] = context_loss_mean.detach()

        # Stereo-preservation auxiliary loss (OFF when weight==0 -> path unchanged).
        # Reconstruct the clean latent z0_hat = noised - t*v_pred, decode BOTH it
        # and the ground-truth latent (diffusion_input) to stereo audio through the
        # frozen pretransform, and match predicted SIDE (L-R)/2 to target side.
        # Gated to low noise (t < tmax, z0_hat meaningful) + a K-row sub-batch for VRAM.
        if self.stereo_loss_weight > 0 and self.diffusion.pretransform is not None:
            from .stereo_loss import rf_z0_hat, compute_stereo_loss
            z0_hat = rf_z0_hat(noised_inputs, output.to(noised_inputs.dtype), t)
            stereo_loss = compute_stereo_loss(
                self.diffusion.pretransform, z0_hat, diffusion_input, t,
                t_max=self.stereo_loss_tmax, subbatch=self.stereo_loss_subbatch,
            )
            loss = loss + self.stereo_loss_weight * stereo_loss
            log_dict["train/stereo_loss"] = stereo_loss.detach()

        # Output-std penalty (2026-08-10 full-FT regularization A/B, spec fullft-regularization-ab).
        # Directly bound the measured latent-scale runaway: penalize the reconstructed clean
        # latent z0_hat's per-channel std drifting off the real data's per-channel std.
        # RF convention (this file, above): noised = z0*(1-t) + noise*t, v = noise - z0
        #   => z0_hat = noised - t*v_pred   (exact when v_pred exact; biased at high t -> t-gate).
        # std_c = per-channel std over (batch,time); penalty applied only where t < t_gate (z0_hat
        # reliable at low noise). Grad flows through z0_hat -> output -> DiT; the real latents are
        # detached. weight==0 (default) => NO-OP, loss byte-identical + zero overhead.
        if self.output_std_penalty > 0:
            tb = t.view(-1, *([1] * (noised_inputs.ndim - 1))).to(noised_inputs.dtype)
            z0_hat = noised_inputs - tb * output.to(noised_inputs.dtype)   # (B, C, T)
            gate = t < self.output_std_t_gate
            if bool(gate.any()):
                zc = z0_hat[gate].float()                       # (Bg, C, T)
                zr = diffusion_input[gate].detach().float()     # (Bg, C, T)
                std_hat = zc.std(dim=(0, 2))                    # (C,) per-channel std over batch+time
                std_real = zr.std(dim=(0, 2))                   # (C,)
                out_std_loss = (std_hat - std_real).pow(2).mean()
                loss = loss + self.output_std_penalty * out_std_loss
                log_dict["train/out_std_loss"] = out_std_loss.detach()
                # Log both stds so the A/B reads the runaway from the CSV without a render.
                log_dict["train/std_z0hat_mean"] = std_hat.detach().mean()
                log_dict["train/std_z0real_mean"] = std_real.detach().mean()

        # Variance-Aware Dynamic Dampening (VADD): quadratic hinge barrier on latent std
        if self.latent_var_barrier > 0:
            tb = t.view(-1, *([1] * (noised_inputs.ndim - 1))).to(noised_inputs.dtype)
            z0_hat = noised_inputs - tb * output.to(noised_inputs.dtype)
            gate = t < self.output_std_t_gate
            if bool(gate.any()):
                zc = z0_hat[gate].float()
                std_hat = zc.std(dim=(0, 2))  # (C,) per-channel std
                cur_mean_std = float(std_hat.mean().item())
                self.running_latent_std = 0.95 * getattr(self, "running_latent_std", cur_mean_std) + 0.05 * cur_mean_std
                log_dict["train/running_latent_std"] = self.running_latent_std

                if self.latent_var_weight > 0:
                    excess = torch.relu(std_hat - self.latent_var_barrier)
                    var_barrier_loss = excess.pow(2).mean()
                    loss = loss + self.latent_var_weight * var_barrier_loss
                    log_dict["train/var_barrier_loss"] = var_barrier_loss.detach()

        # x0-reconstruction ADD-a-term (see __init__): reconstruct the clean latent
        # z0_hat = noised - t*v_pred (rf_z0_hat, SAME as the output-std/stereo blocks)
        # and add lambda_x0 * MSE(z0_hat, z0) toward the ground-truth clean latent.
        # This is the ADD form (loss += term), NOT the x0_equiv REPLACE form. Grad
        # flows through z0_hat -> output -> DiT; the real latents are detached.
        # weight==0 (default) => NO-OP (guarded), loss byte-identical + zero overhead.
        if self.x0_loss_weight > 0 and self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            from .stereo_loss import rf_z0_hat
            z0_hat = rf_z0_hat(noised_inputs, output.to(noised_inputs.dtype), t)
            x0_recon_loss = F.mse_loss(z0_hat, diffusion_input.detach().to(z0_hat.dtype))
            loss = loss + self.x0_loss_weight * x0_recon_loss
            log_dict["train/x0_recon_loss"] = x0_recon_loss.detach()

        log_dict["train/mse_loss"] = mse_loss.detach()
        log_dict["train/loss"] = loss.detach()

        # Stash for external callbacks (e.g. loss-by-timestep logging)
        self._last_t = t.detach()
        self._last_per_elem_loss = mse_loss_full.detach().mean(dim=(1, 2))

        self._staggered_logger.log(log_dict, self)

        # FusionOpt: pipe the current loss into the optimiser BEFORE Lightning
        # calls .step(); Polyak step size γ_t = γ_base·clamp(loss_ema / gnorm_ema)
        # needs the on-device loss tensor. No-op for other optimisers.
        opt = self._fusion_opt()
        if opt is not None:
            opt.set_loss(loss)

        #p.tick("log_dict")
        #print(f"Profiler: {p}")
        return loss

    def validation_step(self, batch, batch_idx):

        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        diffusion_input = reals

        with torch.amp.autocast("cuda"), torch.no_grad():
            conditioning = self.diffusion.conditioner(metadata, self.device)

        # Create batch tensor of padding masks from the metadata
        if all("padding_mask" in md for md in metadata):
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device)
        else:
            padding_masks = torch.ones(diffusion_input.shape[0], diffusion_input.shape[-1], dtype=torch.bool, device=self.device)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.amp.autocast("cuda"), torch.no_grad():
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
                    padding_masks = resize_padding_mask(padding_masks, diffusion_input.shape[-1])
            else:
                # Apply scale to pre-encoded latents if needed, as the pretransform encode function will not be run
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

                if padding_masks.shape[-1] != diffusion_input.shape[-1]:
                    padding_masks = resize_padding_mask(padding_masks, diffusion_input.shape[-1])

        # Use padding mask directly for validation (no silence extension augmentation)
        loss_mask = padding_masks.to(torch.bool)

        extra_args = {}
        if self.mask_padding_attention:
            extra_args["padding_mask"] = padding_masks

        # Set up inpainting conditioning for validation (FULL_MASK: all zeros)
        if self.inpainting_config is not None:
            inpaint_mask = torch.zeros(diffusion_input.shape[0], 1, diffusion_input.shape[2], device=self.device)
            inpaint_masked_input = torch.zeros_like(diffusion_input)
            conditioning['inpaint_mask'] = [inpaint_mask]
            conditioning['inpaint_masked_input'] = [inpaint_masked_input]

        for validation_timestep in self.validation_timesteps:

            t = torch.full((reals.shape[0],), validation_timestep, device=self.device)

            # Calculate the noise schedule parameters for those timesteps
            if self.diffusion_objective in ["v"]:
                alphas, sigmas = get_alphas_sigmas(t)
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                alphas, sigmas = 1-t, t

            # Combine the ground truth data and the noise
            alphas = alphas[:, None, None]
            sigmas = sigmas[:, None, None]
            noise = torch.randn_like(diffusion_input)
            noised_inputs = diffusion_input * alphas + noise * sigmas

            if self.diffusion_objective == "v":
                targets = noise * alphas - diffusion_input * sigmas
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                targets = noise - diffusion_input

            with torch.amp.autocast("cuda"), torch.no_grad():
                output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = 0, **extra_args)

                mse_loss_full = compute_normalized_mse(output, targets, loss_mask, self.loss_normalization, self.loss_norm_eps)
                val_loss, _, _ = compute_masked_loss(
                    mse_loss_full, loss_mask, self.mask_padding_attention, self.mask_loss_weight
                )

                self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'].append(val_loss.item())

    def on_validation_epoch_end(self):
        log_dict = {}
        for validation_timestep in self.validation_timesteps:
            outputs_key = f'val/loss_{validation_timestep:.1f}'
            val_loss = sum(self.validation_step_outputs[outputs_key]) / len(self.validation_step_outputs[outputs_key])

            # Gather losses across all GPUs
            val_loss = self.all_gather(val_loss).mean().item()

            log_metric(self.logger, outputs_key, val_loss, step=self.global_step)

        # Get average over all timesteps
        val_loss = torch.tensor([val for val in self.validation_step_outputs.values()]).mean()

        # Gather losses across all GPUs
        val_loss = self.all_gather(val_loss).mean().item()

        log_metric(self.logger, 'val/avg_loss', val_loss, step=self.global_step)

        # Reset validation losses
        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []


    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

    def export_lora_safetensors(self, path):
        """Export LoRA weights as a safetensors file with embedded config."""
        if self.lora_config is None:
            raise ValueError("No LoRA config -- this wrapper is not in LoRA mode")
        state_dict = {
            **get_lora_state_dict(self.diffusion.model),
            **get_lora_state_dict(self.diffusion.conditioner)
        }
        save_lora_safetensors(state_dict, self.lora_config, path)

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        """Arm FusionOpt's per-component telemetry for this step (if logging interval is hit).
        _telem_on=True tells FusionOpt to accumulate per-stage update norms during its
        _spectral_group_step / _scalar_group_step calls, so we can read them back in
        on_train_batch_end. Cleared after each logged step. No-op when opt is not FusionOpt."""
        opt = self._fusion_opt()
        n = self._staggered_logger.every_n_steps
        if opt is not None and (self.global_step % n == 0):
            opt._telem_on = True

        # VADD: Forward observed latent variance to ModularOptimizer if supported
        if hasattr(optimizer, "set_observed_variance") and hasattr(self, "running_latent_std"):
            optimizer.set_observed_variance(self.running_latent_std)

        # Arm per-stage component telemetry
        if hasattr(optimizer, "_telem_on"):
            optimizer._telem_on = True

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA update AFTER the optimizer step (Lightning fires this post-step). No-op unless
        # use_ema (full-finetune) built the shadow. Cheap: one lerp over the DiT params/step.
        if self.diffusion_ema is not None:
            self.diffusion_ema.update(self.diffusion.model)

        # Component telemetry (FusionOpt / ModularOptimizer): log per-stage update profile to WandB/logger
        opts = self.optimizers()
        if not isinstance(opts, (list, tuple)):
            opts = [opts] if opts is not None else []
        for opt_inst in opts:
            if hasattr(opt_inst, "_comp_telem") and opt_inst._comp_telem:
                telem = opt_inst._comp_telem
                for k, v in telem.items():
                    log_metric(self.logger, k, float(v), step=self.global_step)
                opt_inst._comp_telem = {}
                if hasattr(opt_inst, "_telem_on"):
                    opt_inst._telem_on = False

    def on_save_checkpoint(self, checkpoint):
        if self.lora_config is not None:
            # Preserve the resume-relevant keys Lightning populated (optimizer
            # state, LR schedulers, epoch/step counters) BEFORE clearing the
            # frozen 2.3B base weights out of the checkpoint. Only LoRA params
            # have optimizer state, so these stay LoRA-scale (tens of MB).
            resume_state = {
                k: checkpoint[k]
                for k in (
                    'optimizer_states',
                    'lr_schedulers',
                    'epoch',
                    'global_step',
                    # Lightning-native keys needed for trainer.fit(ckpt_path=...)
                    # to accept the checkpoint and restore loop/epoch position.
                    # Without these, ckpt_path resume raises KeyError on
                    # 'pytorch-lightning_version' (only the base weights are
                    # dropped to keep the file LoRA-scale; loop state is tiny).
                    'pytorch-lightning_version',
                    'loops',
                    'callbacks',
                    'hyper_parameters',
                    'hparams_name',
                    'MixedPrecision',
                )
                if k in checkpoint
            }
            checkpoint.clear()
            checkpoint['state_dict'] = {
                **get_lora_state_dict(self.diffusion.model),
                **get_lora_state_dict(self.diffusion.conditioner)
            }
            checkpoint['lora_config'] = self.lora_config
            checkpoint.update(resume_state)

    # ------------------------------------------------------------------
    # FusionOpt Schedule-Free lifecycle hooks
    # ------------------------------------------------------------------
    # Schedule-Free keeps a fast iterate z_t and an averaged iterate x_t. During
    # training the live params hold y = (1-beta)*z + beta*x (the eval point used
    # to compute gradients); for validation and checkpoint serialisation we want
    # the averaged x_t in the live params (that's the deployable model). The
    # pattern: switch to eval at val start, lazily switch back at the next train
    # batch — so any checkpoint Lightning saves at the end of the val epoch
    # captures the averaged x without us having to munge the state_dict keys.
    def _fusion_opt(self):
        # Returns the optimiser that needs Schedule-Free x/y iterate swapping.
        #
        # This used to isinstance-check FusionOpt ONLY, which silently excluded
        # ModularOptimizer: its train()/eval() swap methods existed but were never
        # called, so a --modular-schedule-free run stayed permanently on the training
        # iterate y and rendered/checkpointed the wrong weights (CONTINUITY 2026-09-21).
        # Duck-type on the contract instead, so any future optimiser implementing it
        # is picked up without editing this function again.
        #
        # FusionOpt is only importable where the stable-audio-tools fork is present
        # (not on LUMI containers), so its import stays guarded and non-fatal.
        try:
            from stable_audio_tools.training.fusion_opt import FusionOpt
        except ModuleNotFoundError:
            return None
        opts = self.optimizers()
        if opts is None:
            return None
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        for o in opts:
            inner = getattr(o, "optimizer", o)
            if isinstance(inner, FusionOpt):
                return inner
        return None

    def _sf_opt(self):
        """Return the optimiser needing Schedule-Free x/y iterate swapping, or None.

        Deliberately SEPARATE from _fusion_opt(): that accessor's other callers invoke
        FusionOpt-only API (set_loss, _telem_on), so widening it to ModularOptimizer
        crashes them. This one is used solely by the train()/eval() swap sites and
        duck-types the contract, so any optimiser implementing it is picked up.
        """
        opts = self.optimizers()
        if opts is None:
            return None
        if not isinstance(opts, (list, tuple)):
            opts = [opts]
        for o in opts:
            inner = getattr(o, "optimizer", o)
            if (getattr(inner, "uses_sf_averaging", False)
                    and callable(getattr(inner, "train", None))
                    and callable(getattr(inner, "eval", None))):
                return inner
        return None

    def on_validation_epoch_start(self):
        opt = self._sf_opt()
        if opt is not None:
            opt.eval()

    def on_train_batch_start(self, batch, batch_idx):
        opt = self._sf_opt()
        if opt is not None:
            opt.train()  # idempotent if already in train mode

class DiffusionCondInpaintDemoCallback(pl.Callback):
    def __init__(
        self,
        demo_every=2000,
        demo_steps=250,
        sample_size=65536,
        sample_rate=48000,
        demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
        demo_conditioning: tp.Optional[tp.List[tp.Dict[str, tp.Any]]] = None,
        inpaint_demo_config: tp.Optional[tp.Dict[str, int]] = None,
        num_demos: int = 0,
        demo_dl=None,
    ):
        super().__init__()
        self.demo_every = demo_every
        self.demo_steps = demo_steps
        self.demo_samples = sample_size
        self.sample_rate = sample_rate
        self.demo_cfg_scales = demo_cfg_scales
        self.demo_conditioning = demo_conditioning or []
        self.last_demo_step = -1

        # Map config keys to MaskType enum
        self._mask_type_map = {
            "num_random_segments": MaskType.RANDOM_SEGMENTS,
            "num_full_mask": MaskType.FULL_MASK,
            "num_causal": MaskType.CAUSAL_MASK,
            "num_random_spans": MaskType.RANDOM_SPANS,
        }

        # Legacy fallback: if no inpaint_demo_config but num_demos is set,
        # use num_demos items with random mask sampling (old behavior)
        if inpaint_demo_config is not None:
            self.inpaint_demo_config = inpaint_demo_config
            self.legacy_inpaint_demos = False
        elif num_demos > 0:
            self.inpaint_demo_config = {}
            self.legacy_inpaint_demos = True
            self.legacy_num_demos = num_demos
        else:
            self.inpaint_demo_config = {}
            self.legacy_inpaint_demos = False

        # Total inpainting demos needed from batch
        if self.legacy_inpaint_demos:
            self.num_inpaint_demos = self.legacy_num_demos
        else:
            self.num_inpaint_demos = sum(
                self.inpaint_demo_config.get(k, 0) for k in self._mask_type_map
            )

        if demo_dl is not None:
            self.demo_dl = iter(demo_dl)
        else:
            self.demo_dl = None

        self._teacher_demo_done = False

    def _generate_prompt_demos(self, module, trainer, is_rank_zero=True):
        """Generate full t2m demos from specified prompts (FULL_MASK)."""
        if not self.demo_conditioning:
            return [], []

        demo_cond = self.demo_conditioning
        num_demos = len(demo_cond)

        demo_samples = self.demo_samples
        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        # Conditioning from prompts
        conditioning = module.diffusion.conditioner(demo_cond, module.device)

        # FULL_MASK: all-zero inpaint conditioning
        io_channels = module.diffusion.io_channels
        inpaint_mask = torch.zeros(num_demos, 1, demo_samples, device=module.device)
        inpaint_masked_input = torch.zeros(num_demos, io_channels, demo_samples, device=module.device)
        conditioning['inpaint_mask'] = [inpaint_mask]
        conditioning['inpaint_masked_input'] = [inpaint_masked_input]

        cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

        noise = torch.randn(num_demos, io_channels, demo_samples, device=module.device)
        model_dtype = next(module.diffusion.parameters()).dtype
        noise = noise.to(model_dtype)

        per_elem_trim = compute_per_elem_trim(demo_cond, self.sample_rate, margin_seconds=2)

        model = module.diffusion_ema.ema_model if module.diffusion_ema is not None else module.diffusion.model

        all_audio = []
        all_context_masks = []

        for cfg_scale in self.demo_cfg_scales:
            if is_rank_zero:
                print(f"Generating prompt demos for cfg scale {cfg_scale}")

            with torch.amp.autocast("cuda"):
                fakes = sample_diffusion(
                    model=model,
                    noise=noise,
                    cond_inputs=cond_inputs,
                    diffusion_objective=module.diffusion_objective,
                    steps=self.demo_steps,
                    cfg_scale=cfg_scale,
                    conditioning=demo_cond,
                    sample_rate=self.sample_rate,
                    pretransform=module.diffusion.pretransform,
                    mask_padding_attention=module.diffusion.mask_padding_attention,
                    use_effective_length_for_schedule=module.diffusion.use_effective_length_for_schedule,
                    headroom_seconds=5.0,
                    dist_shift=module.diffusion.sampling_dist_shift,
                    batch_cfg=True,
                    disable_tqdm=not is_rank_zero,
                    decode=True
                )

            fakes = trim_and_concat(fakes, per_elem_trim)

            all_audio.append(fakes)

        # Latent-resolution all-zeros mask (no context for prompt demos),
        # trimmed to match the per-element audio durations
        ds_ratio = module.diffusion.pretransform.downsampling_ratio if module.diffusion.pretransform is not None else 1
        latent_trim = [t // ds_ratio if t is not None else None for t in per_elem_trim] if per_elem_trim is not None else None
        latent_mask = torch.zeros(num_demos, 1, demo_samples)
        context_mask = trim_and_concat(latent_mask, latent_trim).squeeze(0).cpu()
        all_context_masks = [context_mask] * len(self.demo_cfg_scales)

        del noise, conditioning, cond_inputs, inpaint_mask, inpaint_masked_input
        torch.cuda.empty_cache()

        return all_audio, all_context_masks

    def _generate_inpaint_demos(self, module, trainer, is_rank_zero=True):
        """Generate inpainting demos from batch data with forced mask types."""
        if self.num_inpaint_demos == 0 or self.demo_dl is None:
            return [], []

        demo_reals, metadata = next(self.demo_dl)

        if demo_reals.ndim == 4 and demo_reals.shape[0] == 1:
            demo_reals = demo_reals[0]

        demo_reals = demo_reals[:self.num_inpaint_demos]
        metadata = metadata[:self.num_inpaint_demos]
        model_dtype = next(module.diffusion.parameters()).dtype
        demo_reals = demo_reals.to(module.device, dtype=model_dtype)

        if not module.pre_encoded:
            if module.diffusion.pretransform is not None:
                module.diffusion.pretransform.to(module.device)
                demo_reals = module.diffusion.pretransform.encode(demo_reals)
        else:
            if hasattr(module.diffusion.pretransform, "scale") and module.diffusion.pretransform.scale != 1.0:
                demo_reals = demo_reals / module.diffusion.pretransform.scale

        padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(module.device)
        if padding_masks.shape[-1] != demo_reals.shape[-1]:
            padding_masks = resize_padding_mask(padding_masks, demo_reals.shape[-1])
        mask_padding = module.diffusion.mask_padding_attention

        if self.legacy_inpaint_demos:
            # Legacy: random mask type sampling (old behavior)
            masked_input, mask = random_inpaint_mask(
                demo_reals, padding_masks=padding_masks,
                mask_padding=mask_padding,
                **module.inpaint_mask_kwargs
            )
        else:
            # New: forced mask types per config
            all_masks = []
            all_masked_inputs = []
            idx = 0
            for config_key, mask_type in self._mask_type_map.items():
                count = self.inpaint_demo_config.get(config_key, 0)
                if count == 0:
                    continue
                subset_reals = demo_reals[idx:idx+count]
                subset_padding = padding_masks[idx:idx+count]
                mi, m = random_inpaint_mask(
                    subset_reals, padding_masks=subset_padding,
                    mask_padding=mask_padding, force_mask_type=mask_type,
                    **module.inpaint_mask_kwargs
                )
                all_masks.append(m)
                all_masked_inputs.append(mi)
                idx += count

            mask = torch.cat(all_masks, dim=0)
            masked_input = torch.cat(all_masked_inputs, dim=0)

        conditioning = module.diffusion.conditioner(metadata, module.device)
        conditioning['inpaint_mask'] = [mask]
        conditioning['inpaint_masked_input'] = [masked_input]

        cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

        demo_samples = demo_reals.shape[2]
        noise = torch.randn(demo_reals.shape[0], module.diffusion.io_channels, demo_samples, device=module.device)
        model_dtype = next(module.diffusion.parameters()).dtype
        noise = noise.to(model_dtype)

        per_elem_trim = compute_per_elem_trim(metadata, self.sample_rate, margin_seconds=2)

        # Trim and concatenate context mask at latent resolution,
        # using same trimming basis as audio (per_elem_trim // ds_ratio)
        ds_ratio = module.diffusion.pretransform.downsampling_ratio if module.diffusion.pretransform is not None else 1
        latent_trim = [t // ds_ratio if t is not None else None for t in per_elem_trim] if per_elem_trim is not None else None

        # Zero out padding region in mask for display — the mask is initialized to 1,
        # so without mask_padding the padding frames show as false context in the overlay
        display_mask = mask * padding_masks.unsqueeze(1)

        context_mask = trim_and_concat(display_mask, latent_trim).squeeze(0).cpu()

        model = module.diffusion_ema.ema_model if module.diffusion_ema is not None else module.diffusion.model

        all_audio = []
        all_context_masks = []

        for cfg_scale in self.demo_cfg_scales:
            if is_rank_zero:
                print(f"Generating inpaint demos for cfg scale {cfg_scale}")

            with torch.amp.autocast("cuda"):
                fakes = sample_diffusion(
                    model=model,
                    noise=noise,
                    cond_inputs=cond_inputs,
                    diffusion_objective=module.diffusion_objective,
                    steps=self.demo_steps,
                    cfg_scale=cfg_scale,
                    conditioning=metadata,
                    sample_rate=self.sample_rate,
                    pretransform=module.diffusion.pretransform,
                    mask_padding_attention=module.diffusion.mask_padding_attention,
                    use_effective_length_for_schedule=module.diffusion.use_effective_length_for_schedule,
                    headroom_seconds=5.0,
                    dist_shift=module.diffusion.sampling_dist_shift,
                    batch_cfg=True,
                    disable_tqdm=not is_rank_zero,
                    decode=True
                )

            fakes = trim_and_concat(fakes, per_elem_trim)

            all_audio.append(fakes)
            all_context_masks.append(context_mask)

        del noise, conditioning, cond_inputs, mask, masked_input, padding_masks, demo_reals
        torch.cuda.empty_cache()

        return all_audio, all_context_masks

    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        is_rank_zero = get_rank() == 0

        module.eval()

        self.last_demo_step = trainer.global_step

        try:
            # Generate both types of demos, freeing intermediates between phases
            prompt_audio, prompt_masks = self._generate_prompt_demos(module, trainer, is_rank_zero)
            torch.cuda.empty_cache()

            inpaint_audio, inpaint_masks = self._generate_inpaint_demos(module, trainer, is_rank_zero)
            torch.cuda.empty_cache()

            # Combine per cfg scale (prompt_audio and inpaint_audio have one entry per cfg scale)
            if is_rank_zero:
                for i, cfg_scale in enumerate(self.demo_cfg_scales):
                    parts = []
                    mask_parts = []

                    if i < len(prompt_audio):
                        parts.append(prompt_audio[i])
                        mask_parts.append(prompt_masks[i])

                    if i < len(inpaint_audio):
                        parts.append(inpaint_audio[i])
                        mask_parts.append(inpaint_masks[i])

                    if not parts:
                        continue

                    combined_audio = torch.cat(parts, dim=-1)
                    combined_mask = torch.cat(mask_parts, dim=-1) if mask_parts else None

                    filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                    combined_audio = combined_audio.to(torch.float32).div(torch.max(torch.abs(combined_audio))).mul(32767).to(torch.int16).cpu()
                    torchaudio.save(filename, combined_audio, self.sample_rate)

                    log_audio(trainer.logger, f'demo_cfg_{cfg_scale}', filename, self.sample_rate)
                    log_image(trainer.logger, f'demo_melspec_left_cfg_{cfg_scale}', audio_spectrogram_image(combined_audio, context_mask=combined_mask))
                    if isinstance(trainer.logger, (WandbLogger, CometLogger)):
                        os.remove(filename)

            # Teacher ODE warmup diagnostic: mirror the exact ODE warmup sample_diffusion call
            # and decode the target to verify teacher output quality.
            # Only runs on the first demo.
            # Generates both prompt and inpaint demos, consistent with the main callback.
            teacher_ref = getattr(module, '_teacher', None) or getattr(module, 'teacher_model', None)
            if not self._teacher_demo_done and teacher_ref is not None:
                self._teacher_demo_done = True
                if is_rank_zero:
                    print("Generating teacher ODE warmup diagnostic")
                try:
                    pretransform = module.diffusion.pretransform  # Shared pretransform (not on teacher)
                    io_channels = teacher_ref.io_channels
                    ode_warmup_config = getattr(module, 'ode_warmup_config', {})
                    teacher_cfg = getattr(module, 'ode_warmup_cfg', self.demo_cfg_scales[0])
                    ode_steps = getattr(module, 'ode_n_sampling_steps', 20)
                    mask_padding = module.diffusion.mask_padding_attention
                    ds_ratio = pretransform.downsampling_ratio if pretransform is not None else 1

                    # --- Teacher prompt demos (FULL_MASK, same as _generate_prompt_demos) ---
                    prompt_target = None
                    prompt_per_elem_trim = None
                    prompt_context_mask = None

                    demo_cond = self.demo_conditioning
                    if demo_cond:
                        num_demos = len(demo_cond)
                        demo_samples = self.demo_samples
                        if pretransform is not None:
                            demo_samples = demo_samples // ds_ratio

                        with torch.no_grad():
                            teacher_conditioning = teacher_ref.conditioner(demo_cond, module.device)
                        inpaint_mask = torch.zeros(num_demos, 1, demo_samples, device=module.device)
                        inpaint_masked_input = torch.zeros(num_demos, io_channels, demo_samples, device=module.device)
                        teacher_conditioning['inpaint_mask'] = [inpaint_mask]
                        teacher_conditioning['inpaint_masked_input'] = [inpaint_masked_input]
                        with torch.no_grad():
                            teacher_cond_inputs = teacher_ref.get_conditioning_inputs(teacher_conditioning)

                        noise = torch.randn(num_demos, io_channels, demo_samples, device=module.device)
                        noise = noise.to(next(teacher_ref.parameters()).dtype)
                        prompt_per_elem_trim = compute_per_elem_trim(demo_cond, self.sample_rate, margin_seconds=2)

                        prompt_target = sample_diffusion(
                            model=teacher_ref.model,
                            noise=noise,
                            cond_inputs=teacher_cond_inputs,
                            diffusion_objective=teacher_ref.diffusion_objective,
                            steps=ode_steps,
                            cfg_scale=teacher_cfg,
                            conditioning=demo_cond,
                            sample_rate=teacher_ref.sample_rate,
                            pretransform=pretransform,
                            mask_padding_attention=mask_padding,
                            use_effective_length_for_schedule=module.diffusion.use_effective_length_for_schedule,
                            padding_mask=None,
                            dist_shift=teacher_ref.sampling_dist_shift,
                            sampler_type=ode_warmup_config.get('sampler', 'dpmpp'),
                            batch_cfg=True,
                            disable_tqdm=not is_rank_zero,
                            decode=False,
                        )

                        prompt_latent_trim = [t // ds_ratio if t is not None else None for t in prompt_per_elem_trim] if prompt_per_elem_trim is not None else None
                        prompt_context_mask = trim_and_concat(
                            torch.zeros(num_demos, 1, demo_samples), prompt_latent_trim
                        ).squeeze(0).cpu()

                    # --- Teacher inpaint demos (same mask logic as _generate_inpaint_demos) ---
                    inpaint_target = None
                    inpaint_per_elem_trim = None
                    inpaint_context_mask = None

                    if self.num_inpaint_demos > 0 and self.demo_dl is not None:
                        try:
                            inpaint_reals, inpaint_metadata = next(self.demo_dl)
                            if inpaint_reals.ndim == 4 and inpaint_reals.shape[0] == 1:
                                inpaint_reals = inpaint_reals[0]
                            inpaint_reals = inpaint_reals[:self.num_inpaint_demos]
                            inpaint_metadata = inpaint_metadata[:self.num_inpaint_demos]
                            inpaint_reals = inpaint_reals.to(module.device)

                            if not module.pre_encoded:
                                if pretransform is not None:
                                    inpaint_reals = pretransform.encode(inpaint_reals)
                            else:
                                if hasattr(pretransform, "scale") and pretransform.scale != 1.0:
                                    inpaint_reals = inpaint_reals / pretransform.scale

                            inpaint_padding_masks = torch.stack(
                                [md["padding_mask"][0] for md in inpaint_metadata], dim=0
                            ).to(module.device)

                            if self.legacy_inpaint_demos:
                                masked_input, mask = random_inpaint_mask(
                                    inpaint_reals, padding_masks=inpaint_padding_masks,
                                    mask_padding=mask_padding, **module.inpaint_mask_kwargs
                                )
                            else:
                                all_masks = []
                                all_masked_inputs = []
                                idx = 0
                                for config_key, mask_type in self._mask_type_map.items():
                                    count = self.inpaint_demo_config.get(config_key, 0)
                                    if count == 0:
                                        continue
                                    mi, m = random_inpaint_mask(
                                        inpaint_reals[idx:idx+count],
                                        padding_masks=inpaint_padding_masks[idx:idx+count],
                                        mask_padding=mask_padding, force_mask_type=mask_type,
                                        **module.inpaint_mask_kwargs
                                    )
                                    all_masks.append(m)
                                    all_masked_inputs.append(mi)
                                    idx += count
                                mask = torch.cat(all_masks, dim=0)
                                masked_input = torch.cat(all_masked_inputs, dim=0)

                            with torch.no_grad():
                                inpaint_teacher_cond = teacher_ref.conditioner(inpaint_metadata, module.device)
                            inpaint_teacher_cond['inpaint_mask'] = [mask]
                            inpaint_teacher_cond['inpaint_masked_input'] = [masked_input]
                            with torch.no_grad():
                                inpaint_cond_inputs = teacher_ref.get_conditioning_inputs(inpaint_teacher_cond)

                            inpaint_samples = inpaint_reals.shape[2]
                            inpaint_noise = torch.randn(
                                inpaint_reals.shape[0], io_channels, inpaint_samples, device=module.device
                            ).to(next(teacher_ref.parameters()).dtype)
                            inpaint_per_elem_trim = compute_per_elem_trim(inpaint_metadata, self.sample_rate, margin_seconds=2)

                            inpaint_target = sample_diffusion(
                                model=teacher_ref.model,
                                noise=inpaint_noise,
                                cond_inputs=inpaint_cond_inputs,
                                diffusion_objective=teacher_ref.diffusion_objective,
                                steps=ode_steps,
                                cfg_scale=teacher_cfg,
                                conditioning=inpaint_metadata,
                                sample_rate=teacher_ref.sample_rate,
                                pretransform=pretransform,
                                mask_padding_attention=mask_padding,
                                use_effective_length_for_schedule=module.diffusion.use_effective_length_for_schedule,
                                padding_mask=None,
                                dist_shift=teacher_ref.sampling_dist_shift,
                                sampler_type=ode_warmup_config.get('sampler', 'dpmpp'),
                                batch_cfg=True,
                                disable_tqdm=not is_rank_zero,
                                decode=False,
                            )

                            # Context mask for overlay (same as _generate_inpaint_demos)
                            display_mask = mask * inpaint_padding_masks.unsqueeze(1)
                            inpaint_latent_trim = [t // ds_ratio if t is not None else None for t in inpaint_per_elem_trim] if inpaint_per_elem_trim is not None else None
                            inpaint_context_mask = trim_and_concat(display_mask, inpaint_latent_trim).squeeze(0).cpu()
                        except StopIteration:
                            if is_rank_zero:
                                print("Teacher diagnostic: no inpaint batch available from demo_dl")

                    # --- Combine and log (same pattern as main callback) ---
                    if is_rank_zero:
                        parts = []
                        mask_parts = []

                        if prompt_target is not None:
                            decoded_prompt = pretransform.decode(prompt_target.float())
                            decoded_prompt = trim_and_concat(decoded_prompt, prompt_per_elem_trim)
                            parts.append(decoded_prompt)
                            mask_parts.append(prompt_context_mask)

                        if inpaint_target is not None:
                            decoded_inpaint = pretransform.decode(inpaint_target.float())
                            decoded_inpaint = trim_and_concat(decoded_inpaint, inpaint_per_elem_trim)
                            parts.append(decoded_inpaint)
                            mask_parts.append(inpaint_context_mask)

                        if parts:
                            combined_audio = torch.cat(parts, dim=-1)
                            combined_mask = torch.cat(mask_parts, dim=-1) if mask_parts else None
                            filename = f'demo_teacher_target_{trainer.global_step:08}.wav'
                            combined_audio = combined_audio.to(torch.float32).div(torch.max(torch.abs(combined_audio))).mul(32767).to(torch.int16).cpu()
                            torchaudio.save(filename, combined_audio, self.sample_rate)
                            log_audio(trainer.logger, f'demo_teacher_target', filename, self.sample_rate)
                            log_image(trainer.logger, f'demo_teacher_target_melspec', audio_spectrogram_image(combined_audio, context_mask=combined_mask))
                            os.remove(filename)

                    del prompt_target, inpaint_target
                except Exception as e:
                    if is_rank_zero:
                        print(f"Teacher ODE warmup diagnostic failed: {e}")
                        import traceback
                        traceback.print_exc()

        except Exception as e:
            if is_rank_zero:
                print(f'{type(e).__name__}: {e}')
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()            