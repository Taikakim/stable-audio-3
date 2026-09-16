import json
import numpy as np
import torch
import typing as tp
from torch.nn.functional import interpolate

from stable_audio_3.inference.audio_utils import prepare_audio, numpy_audio_to_tensor
from stable_audio_3.inference.sampling import sample_diffusion, build_schedule
from stable_audio_3.inference.latch_guided import (
    resolve_guided_sampler,
    sample_flow_euler_multi_latch_guided,
    sample_flow_pingpong_multi_latch_guided,
)
from stable_audio_3.inference.latch_targets import build_target as _build_latch_target
from stable_audio_3.models.latch import load_latch_from_checkpoint
from stable_audio_3.loading_utils import load_autoencoder, load_diffusion_cond
from stable_audio_3.model_configs import ae_models, all_models

# Default generation window: 5292032 samples = 120.0 s @ 44.1 kHz (1292 latent frames).
# Kept as the auto-mode STARTING size only — generate() grows the window to fit the
# requested duration unless the caller passes an explicit sample_size cap.
DEFAULT_SAMPLE_SIZE = 5292032
from stable_audio_3.models.lora import (
    set_lora_strength as _set_lora_strength,
    load_and_apply_loras,
)


class StableAudioModel:
    def __init__(self, model, model_config, device, model_half):
        self.model = model
        self.model_config = model_config
        self.device = device
        self.model_half = model_half
        self.same = self.model.pretransform
        self.dit = self.model.model
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def from_pretrained(model_name, device=None, model_half=True):
        # Load the model and any necessary components here
        if device is None and torch.cuda.is_available():
            device = "cuda"
        elif device is None and torch.backends.mps.is_available():
            device = "mps"
        elif device is None:
            device = "cpu"

        if not torch.cuda.is_available():
            if model_name in ("medium", "medium-base"):
                print(
                    f"Warning: You are loading the {model_name} model without a GPU. This model is not designed to run on cpu"
                )
            model_half = False

        if model_name not in all_models:
            raise ValueError(
                f"Unknown model '{model_name}'. Valid models: {list(all_models)}"
            )

        model_cfg = all_models[model_name]
        local_config, local_ckpt = model_cfg.resolve()
        with open(local_config) as f:
            model_config = json.load(f)

        model = load_diffusion_cond(
            model_config, local_ckpt, device=device, model_half=model_half
        )
        model.use_lora = False
        model.lora_names = []
        return StableAudioModel(model, model_config, device, model_half)

    def load_lora(self, lora_ckpt_paths):
        """Load LoRA checkpoints onto the model after construction."""
        model_type = self.model_config["model_type"]
        svd_bases_path = self.model_config.get("svd_bases_path")
        load_and_apply_loras(
            self.model, lora_ckpt_paths, model_type, svd_bases_path=svd_bases_path
        )

    def set_lora_strength(self, strength: float, lora_index: int | None = None):
        _set_lora_strength(self.model.model, strength, lora_index=lora_index)
        _set_lora_strength(self.model.conditioner, strength, lora_index=lora_index)

    @torch.inference_mode()
    def generate(
        self,
        # Simple path: pass a prompt string and duration
        prompt: str | list = None,
        negative_prompt: str | list = None,
        duration: float | list = 120,
        # Generation parameters
        steps: int = 8,
        cfg_scale: float = 1.0,
        batch_size: int = 1,
        sample_size: tp.Optional[int] = None,
        truncate_output_to_duration: bool = True,
        # Low-level path: pass pre-built conditioning dicts
        conditioning: tp.Optional[tp.List[dict]] = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: tp.Optional[tp.List[dict]] = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        seed: int = -1,
        # Audio inputs
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        inpaint_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        inpaint_mask=None,
        inpaint_mask_start_seconds: tp.Optional[tp.Union[float, tp.List[float]]] = None,
        inpaint_mask_end_seconds: tp.Optional[tp.Union[float, tp.List[float]]] = None,
        # Schedule options
        duration_padding_sec: float = 6.0,
        apg_scale: float = 1.0,
        dist_shift=None,
        latch_configs: tp.Optional[tp.List[dict]] = None,
        latch_hparams: tp.Optional[dict] = None,
        return_latents: bool = False,
        latents_sink: tp.Optional[list] = None,
        chunked_decode: tp.Optional[bool] = None,
        **sampler_kwargs,
    ) -> torch.Tensor:
        """
        Generate audio.

        Simple path:
            model.generate(prompt="...", duration=30, steps=100)

        Low-level path (pre-built conditioning):
            model.generate(conditioning=[{"prompt": "...", "seconds_total": 30}], steps=100, ...)

        Args:
            prompt: The text prompt to condition on. Ignored if conditioning dicts are provided directly.
            negative_prompt: The negative text prompt for classifier-free guidance. Ignored if negative_conditioning dicts are provided directly.
            duration: The duration of the generated audio in seconds. Only used if conditioning dicts with "seconds_total" are not provided.
            steps: The number of diffusion steps to use.
            cfg_scale: Classifier-free guidance scale
            batch_size: The batch size to use for generation.
            sample_size: The generation window length in samples. None (default) = auto:
                start from the 120 s default window and GROW it to fit the requested
                duration (seconds_total + padding). Passing an explicit value restores the
                old hard-cap behavior: the window never exceeds it and longer requests are
                clamped with a warning. (The silent 120 s clamp cost two eval campaigns —
                2026-07-15 POOL item, closed 2026-07-22.)
            truncate_output_to_duration: If True, truncate the output audio to the specified duration.
            conditioning: A dictionary of conditioning parameters to use for generation.
            conditioning_tensors: A dictionary of precomputed conditioning tensors to use for generation.
            negative_conditioning: A dictionary of negative conditioning parameters for classifier-free guidance.
            negative_conditioning_tensors: A dictionary of precomputed negative conditioning tensors for classifier-free guidance
            seed: The random seed to use for generation, or -1 to use a random seed.
            init_audio: A tuple of (sample_rate, audio) to use as the initial audio for generation.
            init_noise_level: The noise level to use when generating from an initial audio sample.
            inpaint_audio: A tuple of (sample_rate, audio) to use as the source audio for inpainting. The inpaint region will be determined by the inpaint_mask or inpaint_mask_start_seconds/inpaint_mask_end_seconds parameters.
            inpaint_mask: A prebuilt mask tensor for inpainting. Shape should be [batch_size, sample_size].
                Ignored if inpaint_mask_start_seconds/inpaint_mask_end_seconds are provided.
            inpaint_mask_start_seconds: Start of the inpaint region in seconds. Can be a float
                for a single region, or a list of floats for multiple non-contiguous regions.
            inpaint_mask_end_seconds: End of the inpaint region in seconds. Can be a float
                for a single region, or a list of floats matching inpaint_mask_start_seconds.
            duration_padding_sec: Extra seconds to add when adapting duration (default 6.0).
            apg_scale: APG (Adaptive Projected Guidance) scale. 1.0 = full APG, 0.0 = vanilla CFG.
            dist_shift: Optional distribution shift override for sampling. If None, uses model.sampling_dist_shift.
            return_latents: Whether to return the latents used for generation instead of the decoded audio.
            latents_sink: optional list; the latents are APPENDED to it while the
                normal audio is still returned. Use this rather than a second
                return_latents=True call -- that would double the compute, and
                re-decoding outside generate() means reimplementing two different
                decode paths (latch-guided vs sample_diffusion) that then drift.
            chunked_decode: Whether to decode latents in overlapping chunks to reduce peak VRAM. True forces
                chunked decoding on, False forces it off, None (default) uses the value set in the model config.
            **sampler_kwargs: Additional keyword arguments to pass to the sampler.
        """

        device = str(self.device)

        # Build conditioning from prompt string if not provided directly
        if conditioning is None and conditioning_tensors is None:
            assert prompt is not None, "Must provide either prompt or conditioning"
            conditioning, negative_conditioning = self._build_conditioning_dicts(
                prompt, negative_prompt, duration, batch_size
            )

        # Adapt sample size based on seconds_total in conditioning.
        # sample_size=None -> auto window: default 120s, GROWN to fit the request;
        # explicit sample_size -> hard cap (old behavior, clamp + warning).
        explicit_cap = sample_size is not None
        base_sample_size = sample_size if explicit_cap else DEFAULT_SAMPLE_SIZE
        audio_sample_size = base_sample_size
        if conditioning is not None:
            audio_sample_size = self._adapt_sample_size(
                conditioning,
                base_sample_size,
                duration_padding_sec,
                allow_grow=not explicit_cap,
            )

        # Convert audio sample size to latent size
        latent_sample_size = audio_sample_size
        if self.model.pretransform is not None:
            latent_sample_size = (
                audio_sample_size // self.model.pretransform.downsampling_ratio
            )

        # Build inpaint mask from seconds if provided
        if (
            inpaint_mask_start_seconds is not None
            and inpaint_mask_end_seconds is not None
        ):
            start_is_list = isinstance(inpaint_mask_start_seconds, list)
            end_is_list = isinstance(inpaint_mask_end_seconds, list)
            if start_is_list != end_is_list:
                raise ValueError(
                    "inpaint_mask_start_seconds and inpaint_mask_end_seconds must both be "
                    "scalars or both be lists, got "
                    f"{type(inpaint_mask_start_seconds).__name__} and "
                    f"{type(inpaint_mask_end_seconds).__name__}."
                )
            starts = (
                inpaint_mask_start_seconds
                if start_is_list
                else [inpaint_mask_start_seconds]
            )
            ends = (
                inpaint_mask_end_seconds if end_is_list else [inpaint_mask_end_seconds]
            )
            if len(starts) != len(ends):
                raise ValueError(
                    f"inpaint_mask_start_seconds and inpaint_mask_end_seconds must have the same "
                    f"length, got {len(starts)} and {len(ends)}."
                )
            inpaint_mask = torch.ones(1, audio_sample_size, device=device)
            for start_sec, end_sec in zip(starts, ends):
                mask_start_samples = min(
                    int(start_sec * self.model.sample_rate),
                    audio_sample_size,
                )
                mask_end_samples = min(
                    int(end_sec * self.model.sample_rate),
                    audio_sample_size,
                )
                inpaint_mask[:, mask_start_samples:mask_end_samples] = 0

        # If the caller passed a prebuilt mask sized to the un-adapted sample_size (or
        # anything longer than audio_sample_size), truncate to audio_sample_size so the
        # downstream nearest-neighbor interpolation preserves the mask's time-domain
        # positions instead of squashing the mask region.
        if inpaint_mask is not None and inpaint_mask.shape[-1] > audio_sample_size:
            inpaint_mask = inpaint_mask[:, :audio_sample_size]

        # Match training: when mask_padding_attention is used, random_inpaint_mask
        # zeroes the mask past real_sequence_length. Apply the
        # same convention here so the mask matches the training distribution, whether
        # it was built from seconds above or passed in by the caller.
        if inpaint_mask is not None and conditioning is not None:
            max_seconds = max(
                (c.get("seconds_total", 0.0) for c in conditioning), default=0.0
            )
            if max_seconds > 0:
                effective_audio_len = int(max_seconds * self.model.sample_rate)
                mask_len = inpaint_mask.shape[-1]
                if effective_audio_len < mask_len:
                    inpaint_mask = inpaint_mask.clone()
                    inpaint_mask[:, effective_audio_len:] = 0

        if inpaint_mask is not None:
            inpaint_mask = inpaint_mask.float()

        # Seed and noise
        seed = seed if seed != -1 else np.random.randint(0, 99999)
        torch.manual_seed(seed)
        noise = torch.randn(
            [batch_size, self.model.io_channels, latent_sample_size], device=device
        )

        # Encode conditioning
        if conditioning_tensors is None:
            conditioning_tensors = self.model.conditioner(conditioning, device)
        if (
            negative_conditioning is not None
            or negative_conditioning_tensors is not None
        ):
            if negative_conditioning_tensors is None:
                negative_conditioning_tensors = self.model.conditioner(
                    negative_conditioning, device
                )
        else:
            negative_conditioning_tensors = {}

        # Process init audio
        if init_audio is not None:
            init_audio, inpaint_mask = self._encode_audio_input(
                init_audio, audio_sample_size, inpaint_mask
            )
            init_audio = init_audio.repeat(batch_size, 1, 1)

        # Process inpaint audio
        if inpaint_audio is not None:
            inpaint_audio, inpaint_mask = self._encode_audio_input(
                inpaint_audio, audio_sample_size, inpaint_mask
            )
            inpaint_audio = inpaint_audio.repeat(batch_size, 1, 1)
        else:
            if inpaint_mask is not None:
                inpaint_mask = interpolate(
                    inpaint_mask.unsqueeze(1), size=latent_sample_size, mode="nearest"
                ).squeeze(1)

        # Build inpaint mask tensor and masked input
        if inpaint_mask is None:
            mask = torch.zeros((batch_size, 1, latent_sample_size), device=device)
        else:
            mask = inpaint_mask.unsqueeze(1)
        mask = mask.to(device)

        inpaint_input = (
            inpaint_audio * mask.expand_as(inpaint_audio)
            if inpaint_audio is not None
            else torch.zeros(
                (batch_size, self.model.io_channels, latent_sample_size), device=device
            )
        )

        conditioning_tensors["inpaint_mask"] = [mask]
        conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
        conditioning_inputs = self.model.get_conditioning_inputs(conditioning_tensors)

        if negative_conditioning_tensors:
            negative_conditioning_tensors["inpaint_mask"] = [mask]
            negative_conditioning_tensors["inpaint_masked_input"] = [inpaint_input]
            negative_conditioning_tensors = self.model.get_conditioning_inputs(
                negative_conditioning_tensors, negative=True
            )

        model_dtype = next(self.model.model.parameters()).dtype
        noise = noise.type(model_dtype)
        # A modular local cond ("mir_ctrl", the pianoroll/note-roll inlet) arrives
        # here as a DICT {cond_id: tensor}, not a tensor -- diffusion.py builds it
        # that way and passes it through as one conditioning input. The old blind
        # `v.type(...)` therefore raised
        #   AttributeError: 'dict' object has no attribute 'type'
        # and generate() could never carry a modular local cond at all, which is
        # why the note-roll control had no inference path (SAO gap 3, C 2026-08-26).
        # Tensors and None behave exactly as before; only the dict case is new.
        def _cast(v):
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, dict):
                return {k2: (v2.type(model_dtype) if torch.is_tensor(v2) else v2)
                        for k2, v2 in v.items()}
            if torch.is_tensor(v):
                return v.type(model_dtype)
            return v

        conditioning_inputs = {k: _cast(v) for k, v in conditioning_inputs.items()}

        cond_inputs = {**conditioning_inputs, **negative_conditioning_tensors}

        sampler_type = sampler_kwargs.pop("sampler_type", None)

        if latch_configs:
            result = self._latch_guided_generate(
                noise=noise,
                # audio2audio under guidance: start from the (already-encoded) init
                # latents at init_noise_level instead of pure noise
                init_latents=init_audio if init_audio is not None else None,
                init_noise_level=init_noise_level,
                cond_inputs=cond_inputs,
                latch_configs=latch_configs,
                latch_hparams=latch_hparams or {},
                steps=steps,
                cfg_scale=cfg_scale,
                apg_scale=apg_scale,
                # cfg_interval rides **sampler_kwargs on the sample_diffusion path
                # below; the latch path passes explicit kwargs only, so thread it
                # through here too (same style as the callback passthrough).
                cfg_interval=sampler_kwargs.pop("cfg_interval", (0.0, 1.0)),
                batch_size=batch_size,
                latent_sample_size=latent_sample_size,
                dist_shift=dist_shift
                if dist_shift is not None
                else self.model.sampling_dist_shift,
                return_latents=return_latents,
                latents_sink=latents_sink,
                callback=sampler_kwargs.pop("callback", None),
                # generate()-level sampler_type reaches the guided path too, so a
                # caller does not need a second dialect to say the same thing.
                sampler_type=sampler_type,
            )
        else:
            result = sample_diffusion(
                model=self.model.model,
                noise=noise,
                cond_inputs=cond_inputs,
                diffusion_objective=self.model.diffusion_objective,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                sample_rate=self.model.sample_rate,
                pretransform=self.model.pretransform,
                mask_padding_attention=True,
                use_effective_length_for_schedule=True,
                headroom_seconds=duration_padding_sec,
                dist_shift=dist_shift
                if dist_shift is not None
                else self.model.sampling_dist_shift,
                sampler_type=sampler_type,
                batch_cfg=True,
                rescale_cfg=True,
                apg_scale=apg_scale,
                init_data=init_audio,
                init_noise_level=init_noise_level,
                decode=not return_latents,
                chunked_decode=chunked_decode,
                latents_sink=latents_sink,
                **sampler_kwargs,
            )

        if not return_latents:
            # Normalize DOWN instead of hard-clamping: SA3 raw output routinely peaks
            # >1.0, and clamp() flat-tops it before any writer's peak-normalize can
            # help (W's 2026-07-07 audit: 74% of 48h eval renders clipped, peak at
            # exactly 0 dBFS). Per-item scale, only when over full scale.
            result = result.to(torch.float32)
            peak = result.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1.0)
            result = result / peak

        if not return_latents and truncate_output_to_duration:
            if isinstance(duration, (int, float)):
                max_length_samples = int(duration * self.model.sample_rate)
                result = result[:, :, :max_length_samples]
            else:
                if torch.all(torch.tensor(duration) == duration[0]):
                    max_length_samples = int(duration[0] * self.model.sample_rate)
                    result = result[:, :, :max_length_samples]
                else:
                    # Warn that we can't truncate to a single duration if the durations are different, and return the full length output
                    print(
                        "Warning: Cannot truncate output to a single duration when passing a list of different durations"
                    )

        return result

    def _latch_guided_generate(
        self,
        *,
        noise,
        init_latents=None,
        init_noise_level=1.0,
        cond_inputs,
        latch_configs,
        latch_hparams,
        steps,
        cfg_scale,
        apg_scale,
        cfg_interval=(0.0, 1.0),
        batch_size,
        latent_sample_size,
        dist_shift,
        return_latents,
        latents_sink=None,
        callback=None,
        sampler_type=None,
    ):
        """Run flow-matching Euler sampling with one or more LatCH guides.

        Overrides the normal sampler: builds the schedule, loads each head,
        constructs its target on the latent frame grid, and dispatches to the
        gradient-enabled multi-guide sampler. Heads run fp32; the diffusion model
        may be fp16 (the sampler casts for the head forward).

        generate() runs under @torch.inference_mode(), but TFG needs autograd.
        We escape inference mode here and clone the inbound tensors (noise,
        conditioning) into normal tensors so the guidance gradients can flow.
        """
        device = str(self.device)
        ds_ratio = self.model.pretransform.downsampling_ratio
        latent_fps = float(self.model.sample_rate) / float(ds_ratio)

        def _clean(t):
            """Copy inference tensors (created under inference_mode) into normal ones."""
            if torch.is_tensor(t):
                return t.clone()
            if isinstance(t, list):
                return [_clean(v) for v in t]
            if isinstance(t, tuple):
                return tuple(_clean(v) for v in t)
            if isinstance(t, dict):
                return {k: _clean(v) for k, v in t.items()}
            return t

        with torch.inference_mode(False), torch.enable_grad():
            noise = _clean(noise)
            cond_inputs = _clean(cond_inputs)

            sigma_max = 1.0
            if init_latents is not None:
                # SDEdit-style start: x_t = (1-t)·z0 + t·ε at t = init_noise_level,
                # schedule truncated to [init_noise_level, 0]
                sigma_max = float(init_noise_level)
                z0 = _clean(init_latents).to(noise.dtype)
                noise = (1.0 - sigma_max) * z0 + sigma_max * noise

            sigmas = build_schedule(
                steps=steps,
                sigma_max=sigma_max,
                dist_shift=dist_shift,
                fallback_seq_len=latent_sample_size,
                include_endpoint=True,
                device=device,
            )

            guides = []
            for cfg in latch_configs:
                if cfg.get("builtin") == "recurrence":
                    # E1 anti-loop potential (validation plan 2026-07-15): parameterless
                    # head, no checkpoint; band-hinge target = corpus band UPPER edge
                    # (scalar, broadcasts against the head's (B,1,P) recurrence curve).
                    from .inference.recurrence_potential import RecurrenceHead
                    head = RecurrenceHead(
                        fps=latent_fps,
                        **{k: float(cfg[k]) for k in
                           ("patch_sec", "lookback_min_sec", "lookback_max_sec", "temp")
                           if cfg.get(k) is not None},
                    ).to(device)
                    guides.append({
                        "head": head,
                        "target": torch.full((batch_size, 1, 1),
                                             float(cfg.get("value", 0.738)), device=device),
                        "weight": float(cfg.get("weight", 1.0)),
                        "start_pct": float(cfg.get("start_pct", 0.3)),
                        "end_pct": float(cfg.get("end_pct", 0.8)),
                        "loss_type": "band_hinge",
                        "huber_beta": float(cfg.get("huber_beta", 0.05)),
                        "w_sec": None, "fps": latent_fps,
                    })
                    continue
                head = load_latch_from_checkpoint(cfg["model_path"], device=device)
                meta = getattr(head, "metadata", {}) or {}
                head_sched = meta.get("noise_schedule")
                if head_sched is not None and head_sched != self.model.diffusion_objective:
                    print(
                        f"[LatCH] WARNING: head trained for noise_schedule='{head_sched}' "
                        f"but model objective is '{self.model.diffusion_objective}'."
                    )
                if cfg.get("target_raw") is not None:
                    # Raw per-frame target array/tensor [C, T_any] (e.g. a measured
                    # chroma curve for chroma-morph transitions) — nearest-resampled
                    # to the latent frame grid, then standardized like built targets.
                    raw = torch.as_tensor(cfg["target_raw"], dtype=torch.float32)
                    if raw.dim() == 2:
                        raw = raw.unsqueeze(0)
                    if raw.shape[-1] != latent_sample_size:
                        raw = torch.nn.functional.interpolate(
                            raw, size=latent_sample_size, mode="linear",
                            align_corners=False)
                    target = raw.repeat(batch_size, 1, 1).to(device)
                else:
                    kind = cfg.get("kind") or meta.get("target_kind_default", "constant")
                    value = float(cfg.get("value", 1.0))
                    target = _build_latch_target(
                        kind, value,
                        batch_size=batch_size,
                        channels=head.out_channels,
                        frames=latent_sample_size,
                        fps=latent_fps,
                        device=device,
                        dtype=torch.float32,
                    )
                if meta.get("standardized"):
                    _m = float(meta.get("std_mean", 0.0))
                    _s = float(meta.get("std_std", 1.0)) or 1.0
                    target = (target - _m) / _s
                guides.append({
                    "head": head,
                    "target": target,
                    "weight": float(cfg.get("weight", 1.0)),
                    "start_pct": float(cfg.get("start_pct", 0.0)),
                    "end_pct": float(cfg.get("end_pct", 1.0)),
                    # cfg override wins over head metadata: lets a caller run e.g.
                    # scalar_pooled guidance on an mse-trained scalar head (the
                    # constant-target-flatness fix, 2026-07-10) without retraining.
                    "loss_type": cfg.get("loss_type") or meta.get("loss_type", "mse"),
                    "huber_beta": meta.get("huber_beta") or 1.0,
                    "w_sec": cfg.get("w_sec"),
                    "fps": cfg.get("fps"),
                })

            hp = {
                "rho": float(latch_hparams.get("rho", 1.0)),
                "mu": float(latch_hparams.get("mu", 1.0)),
                "gamma": float(latch_hparams.get("gamma", 0.3)),
                "n_iter": int(latch_hparams.get("n_iter", 4)),
                "log_norms": bool(latch_hparams.get("log_norms", False)),
            }

            # SAMPLER CHOICE MUST FOLLOW THE MODEL'S OBJECTIVE (W, 2026-09-16).
            # sampling.py:445 picks pingpong for rf_denoiser (the post-trained
            # "medium") and euler otherwise. This path used to hardcode euler, so
            # asking for ANY guidance silently swapped the post-trained model's
            # native sampler for a mismatched one -- 8 Euler steps on a model
            # distilled for 8-step pingpong -- before a head was ever consulted.
            # Keep this condition identical to sampling.py's; two places deciding
            # the same thing by different rules is how that bug arose.
            _sampler = resolve_guided_sampler(
                self.model.diffusion_objective, latch_hparams, sampler_type)
            _guided_sampler = (sample_flow_pingpong_multi_latch_guided
                               if _sampler == "pingpong"
                               else sample_flow_euler_multi_latch_guided)
            latents = _guided_sampler(
                self.model.model, noise, sigmas, guides,
                cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, apg_scale=apg_scale,
                # lands in **model_kwargs -> DiT forward, gated at dit.py
                # (cfg_interval[0] <= sigma <= cfg_interval[1]) — native sigma semantics
                cfg_interval=tuple(cfg_interval),
                callback=callback, **hp, **cond_inputs,
            )

            # Non-invasive capture: same audio out, z0 also handed to the caller.
            if latents_sink is not None:
                latents_sink.append(latents.detach())
            if return_latents:
                return latents.detach()
            decode_dtype = next(self.model.pretransform.parameters()).dtype
            with torch.no_grad():
                return self.model.pretransform.decode(latents.detach().type(decode_dtype))

    # --- generate() helpers ---

    @staticmethod
    def _build_conditioning_dicts(prompt, negative_prompt, duration, batch_size):
        """Returns (conditioning, negative_conditioning) lists of dicts."""

        def _to_list(value, name):
            """Broadcast a scalar or validate a sequence to length batch_size."""
            if isinstance(value, (list, tuple)):
                assert len(value) == batch_size, (
                    f"Length of {name} ({len(value)}) must match batch_size ({batch_size})"
                )
                return list(value)
            return [value] * batch_size

        prompts = _to_list(prompt, "prompt")
        durations = _to_list(duration, "duration")
        conditioning = [
            {"prompt": p, "seconds_total": d} for p, d in zip(prompts, durations)
        ]

        negative_conditioning = None
        if negative_prompt is not None:
            neg_prompts = _to_list(negative_prompt, "negative_prompt")
            negative_conditioning = [
                {"prompt": p, "seconds_total": d}
                for p, d in zip(neg_prompts, durations)
            ]

        return conditioning, negative_conditioning

    def _adapt_sample_size(self, conditioning, sample_size, duration_padding_sec,
                           allow_grow=False):
        """Returns audio_sample_size adapted from conditioning.

        allow_grow=False (explicit-cap mode): never exceeds sample_size; longer requests
        are clamped with a warning. allow_grow=True (auto mode): the window GROWS past
        sample_size to fit seconds_total + padding — the silent-120s-clamp fix
        (2026-07-15 POOL item; the clamp cost the LUMI native-cells campaign 103 renders)."""
        max_seconds = 0.0
        for cond_dict in conditioning:
            if "seconds_total" in cond_dict:
                max_seconds = max(max_seconds, cond_dict["seconds_total"])

        if max_seconds <= 0:
            return sample_size

        target_audio_samples = int(
            (max_seconds + duration_padding_sec) * self.model.sample_rate
        )
        if self.model.pretransform is not None:
            ds_ratio = self.model.pretransform.downsampling_ratio
            # Round up to nearest multiple of downsampling ratio
            target_audio_samples = (
                (target_audio_samples + ds_ratio - 1) // ds_ratio
            ) * ds_ratio
            encoder_config = self.model_config["model"]["pretransform"]["config"][
                "encoder"
            ]["config"]
            chunk_size = encoder_config.get("chunk_size", 32)
            stride = encoder_config["strides"][0]  # or min(strides) if multiple
            # For chunked attention with latent space, align to chunk size after downsampling
            latent_align = chunk_size // stride
            align = ds_ratio * latent_align
            target_audio_samples = ((target_audio_samples + align - 1) // align) * align

        if target_audio_samples > sample_size:
            sr = self.model.sample_rate
            if allow_grow:
                print(
                    f"[generate] window auto-grown to {target_audio_samples / sr:.2f}s "
                    f"({target_audio_samples} samples) to fit seconds_total={max_seconds:.1f} "
                    f"+ pad {duration_padding_sec:.1f} (default window is "
                    f"{sample_size / sr:.2f}s; pass sample_size explicitly to cap)."
                )
                return target_audio_samples
            print(
                f"Warning: requested duration {target_audio_samples / sr:.2f}s "
                f"(seconds_total={max_seconds:.1f} + pad {duration_padding_sec:.1f}) exceeds the "
                f"sample_size cap {sample_size / sr:.2f}s ({sample_size} samples) -- output CLAMPED "
                f"to the cap. Raise sample_size (or --duration) to render the full length."
            )
        return min(target_audio_samples, sample_size)

    def _encode_audio_input(self, audio_input, audio_sample_size, inpaint_mask=None):
        """
        Converts a (sample_rate, audio) tuple to an encoded latent tensor.
        If model has a pretransform, encodes to latent space and downsamples inpaint_mask to match.
        Returns (encoded_audio, updated_inpaint_mask). encoded_audio is not yet repeated to batch size.
        """
        device = str(self.device)
        in_sr, audio_data = audio_input
        if isinstance(audio_data, np.ndarray):
            audio_data = numpy_audio_to_tensor(audio_data)
        io_channels = (
            self.model.pretransform.io_channels
            if self.model.pretransform is not None
            else self.model.io_channels
        )
        audio = prepare_audio(
            audio_data,
            in_sr=in_sr,
            target_sr=self.model.sample_rate,
            target_length=audio_sample_size,
            target_channels=io_channels,
            device=device,
        )
        if self.model.pretransform is not None:
            audio = audio.to(next(self.model.pretransform.parameters()).dtype)
            audio = self.model.pretransform.encode(audio)
            if inpaint_mask is not None:
                inpaint_mask = interpolate(
                    inpaint_mask.unsqueeze(1),
                    size=audio.shape[-1],
                    mode="nearest",
                ).squeeze(1)
        return audio, inpaint_mask


class AutoencoderModel:
    def __init__(self, autoencoder, sample_rate, device):
        self.autoencoder = autoencoder
        self.sample_rate = sample_rate
        self.device = device

    @staticmethod
    def from_pretrained(model_name, device=None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        if not torch.cuda.is_available():
            if model_name == "same-l":
                print(
                    f"Warning: You are loading the {model_name} model without a GPU. This model is not designed to run on cpu"
                )

        if model_name not in ae_models:
            raise ValueError(
                f"Unknown autoencoder '{model_name}'. Valid models: {list(ae_models)}"
            )

        cfg = ae_models[model_name]
        local_config, local_ckpt = cfg.resolve()

        with open(local_config) as f:
            sample_rate = json.load(f)["sample_rate"]

        autoencoder = load_autoencoder(local_config, local_ckpt, device=device)
        autoencoder.eval().requires_grad_(False)

        return AutoencoderModel(autoencoder, sample_rate, device)

    @torch.inference_mode()
    def encode(self, audio, sr, chunked=False, chunk_size=128, overlap=32):
        """Encode audio to latents.

        Args:
            audio: A single waveform tensor (C, T), a list of waveform tensors,
                or a pre-batched tensor (B, C, T). Resampling, channel conversion,
                and padding are handled automatically; passing sr=ae.sample_rate
                for already-preprocessed audio skips resampling.
            sr: Sample rate of the input audio, or a list of sample rates when
                audio is a list.
            chunked: If True, encode in overlapping chunks to save memory.
            chunk_size: Chunk size in latent frames (only used when chunked=True).
            overlap: Overlap in latent frames between chunks (only used when chunked=True).

        Returns:
            Latent tensor of shape (B, latent_dim, latent_time).
        """
        if isinstance(audio, list):
            preprocessed = self.autoencoder.preprocess_audio_list_for_encoder(
                audio, in_sr_list=sr
            )
        elif isinstance(audio, torch.Tensor) and audio.dim() == 3:
            sr_list = sr if isinstance(sr, list) else [sr] * audio.shape[0]
            preprocessed = self.autoencoder.preprocess_audio_list_for_encoder(
                list(audio), in_sr_list=sr_list
            )
        else:
            preprocessed = self.autoencoder.preprocess_audio_for_encoder(
                audio, in_sr=sr
            )
        return self.autoencoder.encode_audio(
            preprocessed.to(self.device),
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    @torch.inference_mode()
    def decode(self, latents, chunked=False, chunk_size=128, overlap=32):
        """Decode latents to audio.

        Args:
            latents: Latent tensor of shape (B, latent_dim, latent_time).
            chunked: If True, decode in overlapping chunks to save memory.
            chunk_size: Chunk size in latent frames (only used when chunked=True).
            overlap: Overlap in latent frames between chunks (only used when chunked=True).

        Returns:
            Audio tensor of shape (B, channels, samples).
        """
        return self.autoencoder.decode_audio(
            latents,
            chunked=chunked,
            chunk_size=chunk_size,
            overlap=overlap,
        )
