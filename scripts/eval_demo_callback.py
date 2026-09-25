#!/usr/bin/env python3
"""Modular Demo Rendering & Epoch Loss Guard Callback for SA3 LoRA Training.

Integrates:
1. In-training milestone demo rendering (step 100 first, and between epochs)
2. Three canonical prompts: kl_2, rb_rare_7, rb_mid_4
3. Fast 20s and Native 48s clips per prompt at cfg 7 / w 1.0 (24 steps, Euler):
   - Standard 20s (T=216 frames)
   - Native 48s (T=512 frames, matches training crop)
   (Continuations can optionally be enabled via render_continuations=True)
4. Fast GPU latent offload (.z0.npy) with synchronous direct decode / CPU offload
5. Epoch Loss Guard: if mean epoch loss > 0.8, saves emergency checkpoint,
   renders demo clips, and aborts training.
6. OOM Guard: catches VRAM OOM, temporarily offloads optimizer buffers to CPU,
   and retries render cleanly.

Origin: Kim & Antigravity.Neuromancer
"""

import os
import sys
import math
import time
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import pytorch_lightning as pl

from stable_audio_3.inference.sampling import sample_diffusion

CANONICAL_DEMO_PROMPTS = [
    {
        "id": "kl_2",
        "text": (
            "This track is a high-energy Psytrance piece that blends the driving pulse of "
            "classic Goa trance with modern, polished electronic production. It sits at "
            "150 BPM in a 4/4 time signature and is rooted in F minor. Instrumentation & "
            "production: The arrangement is built around a relentless four-on-the-floor kick "
            "and a thick, side-chain-compressed synth bass that anchors the low end."
        ),
        "seed": 1002,
    },
    {
        "id": "rb_rare_7",
        "text": "2010s psy-trance, progressive trance, 136 bpm",
        "seed": 16488276,
    },
    {
        "id": "rb_mid_4",
        "text": "mid 90s psy-trance, goa trance, 142 bpm",
        "seed": 1331736365,
    },
    {
        "id": "rb_common_0",
        "text": "2020s goa trance, melodic mood, 148 bpm",
        "seed": 786795416,
    },
]


def generate_clip_latents(
    diffusion_wrapper,
    prompt_text: str,
    seed: int,
    total_frames: int,
    prefix_latents: torch.Tensor | None = None,
    steps: int = 24,
    cfg_scale: float = 7.0,
    cfg_rescale: float = 0.0,
    max_latent_std: float | None = None,
) -> torch.Tensor:
    """Generate latents for a prompt with optional inpaint-continuation conditioning and VADD guards."""
    device = next(diffusion_wrapper.parameters()).device
    model_dtype = next(diffusion_wrapper.model.parameters()).dtype
    downsampling_ratio = (
        diffusion_wrapper.pretransform.downsampling_ratio
        if diffusion_wrapper.pretransform is not None else 4096
    )
    sample_rate = diffusion_wrapper.sample_rate
    fps = sample_rate / downsampling_ratio
    dur_sec = total_frames / fps

    cond_dict = [{"prompt": prompt_text, "seconds_total": dur_sec}]
    ct = diffusion_wrapper.conditioner(cond_dict, device)

    io_channels = diffusion_wrapper.io_channels
    mask = torch.zeros((1, 1, total_frames), device=device, dtype=model_dtype)
    masked_input = torch.zeros((1, io_channels, total_frames), device=device, dtype=model_dtype)

    if prefix_latents is not None:
        prefix_len = min(prefix_latents.shape[-1], total_frames)
        mask[:, :, :prefix_len] = 1.0
        masked_input[:, :, :prefix_len] = prefix_latents[..., :prefix_len].to(device, dtype=model_dtype)

    ct["inpaint_mask"] = [mask]
    ct["inpaint_masked_input"] = [masked_input]

    cond_inputs = diffusion_wrapper.get_conditioning_inputs(ct)
    cond_inputs = {
        k: (v.type(model_dtype) if torch.is_tensor(v) else v)
        for k, v in cond_inputs.items()
    }

    torch.manual_seed(seed)
    noise = torch.randn(1, io_channels, total_frames, device=device, dtype=model_dtype)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            latents = sample_diffusion(
                model=diffusion_wrapper.model,
                noise=noise,
                cond_inputs=cond_inputs,
                diffusion_objective=diffusion_wrapper.diffusion_objective,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=cond_dict,
                sample_rate=sample_rate,
                pretransform=diffusion_wrapper.pretransform,
                mask_padding_attention=True,
                dist_shift=diffusion_wrapper.sampling_dist_shift,
                decode=False,
                scale_phi=cfg_rescale,
            )

    if not torch.all(torch.isfinite(latents)).item():
        print(f"[DEMO WARNING] Latents generated for '{prompt_text[:30]}...' contain NaN/Inf!")

    # VADD Latent Clamping Guard
    if max_latent_std is not None and max_latent_std > 0:
        curr_std = float(latents.float().std().item())
        if curr_std > max_latent_std:
            print(f"[DEMO GUARD] Latent std {curr_std:.3f} exceeded ceiling {max_latent_std:.3f}! Clamping before decode.", flush=True)
            latents = latents * (max_latent_std / (curr_std + 1e-8))

    if prefix_latents is not None:
        prefix_len = min(prefix_latents.shape[-1], total_frames)
        latents[..., :prefix_len] = prefix_latents[..., :prefix_len].to(latents.device)

    return latents.cpu()


class ModularDemoAndLossGuardCallback(pl.Callback):
    def __init__(
        self,
        save_dir: str,
        loss_guard_threshold: float = 0.8,
        loss_log_every: int = 10,
        step_milestones: tuple[int, ...] = (100,),
        demo_steps: int = 24,
        demo_cfg: float = 7.0,
        frames_native: int = 512,
        frames_ext_1: int = 512,
        frames_ext_2: int = 303,
        frames_20s: int = 216,
        auto_decode_cpu: bool = True,
        render_between_epochs: bool = False,
        render_continuations: bool = False,
        num_prompts: int = 3,
        cfg_rescale: float = 0.0,
        max_latent_std: float | None = None,
        render_inline: bool = True,
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.demos_dir = self.save_dir / "demos"
        self.demos_dir.mkdir(parents=True, exist_ok=True)
        self.loss_guard_threshold = loss_guard_threshold
        self.loss_log_every = loss_log_every
        self.step_milestones = set(step_milestones)
        self.rendered_steps = set()
        self.demo_steps = demo_steps
        self.demo_cfg = demo_cfg
        self.frames_native = frames_native
        self.frames_ext_1 = frames_ext_1
        self.frames_ext_2 = frames_ext_2
        self.frames_20s = frames_20s
        self.auto_decode_cpu = auto_decode_cpu
        self.render_between_epochs = render_between_epochs
        self.render_continuations = render_continuations
        self.num_prompts = num_prompts
        self.cfg_rescale = cfg_rescale
        self.max_latent_std = max_latent_std
        # False = milestones save checkpoints only (see --no-inline-demos, training-findings 13e).
        self.render_inline = render_inline
        self.epoch_losses: list[float] = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Lightning divides the training-step loss by accumulate_grad_batches before
        # it reaches this hook (loops/optimization/automatic.py: closure_loss / normalize),
        # so the raw value here is 1/N of the real loss. Undo it, or the monitor reports
        # a fraction of the truth and loss_guard_threshold is silently N times looser.
        accum = max(1, int(getattr(trainer, "accumulate_grad_batches", 1) or 1))
        if outputs is not None:
            if isinstance(outputs, dict) and "loss" in outputs:
                self.epoch_losses.append(outputs["loss"].item() * accum)
            elif torch.is_tensor(outputs):
                self.epoch_losses.append(outputs.item() * accum)

        step = trainer.global_step
        if step in self.step_milestones and step not in self.rendered_steps:
            self.rendered_steps.add(step)
            # Commit milestone checkpoint to disk first
            ckpt_path = self.save_dir / f"step={step}.ckpt"
            if not ckpt_path.exists():
                trainer.save_checkpoint(str(ckpt_path))
                print(f"\n[MILESTONE] Saved checkpoint to {ckpt_path.name}")
            # If demos already exist on disk, skip re-rendering
            if not self.render_inline:
                print(f"[DEMO] inline rendering off: render step {step} later with demo_cfg_sweep.py --run-dir <run> --steps-ckpt {step} --cfgs 7 --write-demos", flush=True)
                return
            out_dir = self.demos_dir / f"step{step}"
            expected_clips = self.num_prompts * (4 if self.render_continuations else 2)
            if out_dir.exists() and len(list(out_dir.glob("*.wav"))) >= expected_clips:
                print(f"[DEMO TRIGGER] Step milestone {step} already has {expected_clips} clips rendered. Skipping.")
            else:
                print(f"\n[DEMO TRIGGER] Step milestone {step} reached. Rendering canonical demos...")
                self._render_demos(trainer, pl_module, tag=f"step{step}")

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if not self.epoch_losses:
            return

        mean_loss = float(np.mean(self.epoch_losses))
        # The guard must evaluate every epoch, but it need not SAY so every epoch: at 160
        # epochs that was 320 lines burying everything else. Speak on a cadence, on the
        # first and last epoch, and always when something is wrong.
        self._epoch_losses_seen = getattr(self, "_epoch_losses_seen", [])
        self._epoch_losses_seen.append(mean_loss)
        noteworthy = (not math.isfinite(mean_loss)) or (mean_loss > self.loss_guard_threshold)
        cadence = max(1, int(getattr(self, "loss_log_every", 10)))
        last = getattr(trainer, "max_epochs", None)
        speak = noteworthy or epoch == 0 or epoch % cadence == 0 or (last and epoch == last - 1)
        if speak:
            # Window stats over FINITE values only: a single NaN epoch otherwise poisons
            # the rolling average and the trend for every later report, which is how a
            # summary stops being readable exactly when you most need it.
            fin = lambda xs: [v for v in xs if math.isfinite(v)]
            window = fin(self._epoch_losses_seen[-cadence:])
            prev = fin(self._epoch_losses_seen[-2 * cadence:-cadence])
            n_bad = len(self._epoch_losses_seen[-cadence:]) - len(window)
            parts = [f"mean {mean_loss:.4f}"]
            if window:
                parts.append(f"last {len(window)} avg {sum(window)/len(window):.4f}")
                if prev:
                    d = sum(window) / len(window) - sum(prev) / len(prev)
                    parts.append(f"{d:+.4f} vs previous {len(prev)}")
            if n_bad:
                parts.append(f"{n_bad} NON-FINITE epoch(s) in window")
            parts.append(f"guard {self.loss_guard_threshold}")
            print(f"[LOSS MONITOR] Epoch {epoch}: " + ", ".join(parts), flush=True)

        # NaN/Inf must trip the guard: `nan > threshold` is False, so a bare
        # `>` comparison routes a diverged run into the "healthy" branch.
        diverged = (not math.isfinite(mean_loss)) or (mean_loss > self.loss_guard_threshold)
        if diverged:
            print(f"\n========================================================")
            print(f"[LOSS GUARD ALERT] Epoch {epoch} mean loss {mean_loss:.4f} is non-finite or > {self.loss_guard_threshold}!")
            print(f"[LOSS GUARD ALERT] Aborting training, saving checkpoint, and rendering demos...")
            print(f"========================================================\n", flush=True)

            ckpt_path = self.save_dir / f"abort_loss_{mean_loss:.4f}_epoch{epoch}.ckpt"
            trainer.save_checkpoint(str(ckpt_path))
            print(f"[LOSS GUARD] Emergency checkpoint saved: {ckpt_path.name}")

            if self.render_inline:
                self._render_demos(trainer, pl_module, tag=f"abort_loss{mean_loss:.3f}_ep{epoch}")
            trainer.should_stop = True
        elif self.render_between_epochs and trainer.global_step >= 100:
            print(f"[DEMO TRIGGER] Epoch {epoch} complete (loss healthy). Rendering between-epoch demos...")
            self._render_demos(trainer, pl_module, tag=f"ep{epoch}")
        # No "loss healthy" line: it doubled the log volume to say what the line above
        # already says. Silence means healthy.

        self.epoch_losses.clear()

    def _render_demos(self, trainer, pl_module, tag: str):
        """Render canonical prompts at 20s and native length (with optional continuations)."""
        out_dir = self.demos_dir / tag
        out_dir.mkdir(parents=True, exist_ok=True)

        was_training = pl_module.training
        pl_module.eval()

        # Schedule-Free: training keeps params at the y iterate; the DEPLOYABLE weights
        # are the averaged x. Demos rendered without this swap judge the wrong weights.
        # pl_module.eval() does NOT do it -- the swap lives on the optimiser, and the
        # existing hook (diffusion.py on_validation_epoch_start) never fires here because
        # demos render from on_train_batch_end, not from a validation loop.
        sf_opt = None
        for _o in (getattr(trainer, "optimizers", None) or []):
            _inner = getattr(_o, "optimizer", _o)
            if getattr(_inner, "uses_sf_averaging", False) and callable(getattr(_inner, "eval", None)):
                sf_opt = _inner
                break
        if sf_opt is not None:
            print("[DEMO] Schedule-Free: swapping to averaged iterate x for rendering.", flush=True)
            sf_opt.eval()

        t0 = time.time()
        prompts = CANONICAL_DEMO_PROMPTS[:self.num_prompts]
        total_clips = len(prompts) * (4 if self.render_continuations else 2)
        print(f"[DEMO] Rendering {total_clips} clips for milestone '{tag}' at cfg={self.demo_cfg} / w=1.0...", flush=True)

        # Temporarily route attention to PyTorch SDPA during inference sampling
        # (Avoids AMD Composable Kernel FmhaFwdKernel illegal instruction on GFX1201)
        import stable_audio_3.models.transformer as T
        orig_fa = T.flash_attn_func
        orig_fa_varlen = T.flash_attn_varlen_func
        T.flash_attn_func = None
        T.flash_attn_varlen_func = None

        def _decode_and_save_wav(z, wav_path):
            if wav_path.exists():
                return
            pt = pl_module.diffusion.pretransform
            if pt is None:
                return
            dev = next(pt.parameters()).device
            pt_dtype = next(pt.parameters()).dtype
            with torch.no_grad():
                audio = pt.decode(z.to(device=dev, dtype=pt_dtype))
            audio_np = audio[0].cpu().float().numpy().T
            peak = np.abs(audio_np).max()
            if peak > 1e-6:
                audio_np = (audio_np / peak) * 0.988
            sf.write(str(wav_path), audio_np, pl_module.diffusion.sample_rate, subtype="PCM_16")

        try:
            for p in prompts:
                pid = p["id"]
                ptext = p["text"]
                pseed = p["seed"]

                def _render_all_for_prompt():
                    # 1. Standard 20s
                    f_20s = out_dir / f"{tag}_{pid}_20s.z0.npy"
                    w_20s = out_dir / f"{tag}_{pid}_20s.wav"
                    if not f_20s.exists():
                        z_20 = generate_clip_latents(
                            pl_module.diffusion, ptext, pseed,
                            total_frames=self.frames_20s,
                            steps=self.demo_steps, cfg_scale=self.demo_cfg,
                            cfg_rescale=self.cfg_rescale,
                            max_latent_std=self.max_latent_std,
                        )
                        np.save(f_20s, z_20.numpy())
                        _decode_and_save_wav(z_20, w_20s)
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        time.sleep(0.2)

                    # 2. Native 48s (T=512)
                    f_48s = out_dir / f"{tag}_{pid}_48s.z0.npy"
                    w_48s = out_dir / f"{tag}_{pid}_48s.wav"
                    if f_48s.exists():
                        z_48 = torch.from_numpy(np.load(f_48s))
                    else:
                        z_48 = generate_clip_latents(
                            pl_module.diffusion, ptext, pseed,
                            total_frames=self.frames_native,
                            steps=self.demo_steps, cfg_scale=self.demo_cfg,
                            cfg_rescale=self.cfg_rescale,
                            max_latent_std=self.max_latent_std,
                        )
                        np.save(f_48s, z_48.numpy())
                        _decode_and_save_wav(z_48, w_48s)
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        time.sleep(0.2)

                    # 3 & 4. Optional Continuations
                    if self.render_continuations:
                        # Continuation A (+512 frames -> T=1024)
                        f_ext1 = out_dir / f"{tag}_{pid}_ext512.z0.npy"
                        w_ext1 = out_dir / f"{tag}_{pid}_ext512.wav"
                        if not f_ext1.exists():
                            t_ext1 = self.frames_native + self.frames_ext_1
                            z_ext1 = generate_clip_latents(
                                pl_module.diffusion, ptext, pseed,
                                total_frames=t_ext1, prefix_latents=z_48,
                                steps=self.demo_steps, cfg_scale=self.demo_cfg,
                                cfg_rescale=self.cfg_rescale,
                                max_latent_std=self.max_latent_std,
                            )
                            np.save(f_ext1, z_ext1.numpy())
                            _decode_and_save_wav(z_ext1, w_ext1)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            time.sleep(0.2)

                        # Continuation B (+303 frames -> T=815)
                        f_ext2 = out_dir / f"{tag}_{pid}_ext303.z0.npy"
                        w_ext2 = out_dir / f"{tag}_{pid}_ext303.wav"
                        if not f_ext2.exists():
                            t_ext2 = self.frames_native + self.frames_ext_2
                            z_ext2 = generate_clip_latents(
                                pl_module.diffusion, ptext, pseed,
                                total_frames=t_ext2, prefix_latents=z_48,
                                steps=self.demo_steps, cfg_scale=self.demo_cfg,
                                cfg_rescale=self.cfg_rescale,
                                max_latent_std=self.max_latent_std,
                            )
                            np.save(f_ext2, z_ext2.numpy())
                            _decode_and_save_wav(z_ext2, w_ext2)
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            time.sleep(0.2)

                self._run_with_oom_guard(_render_all_for_prompt, trainer, pl_module)
                if self.render_continuations:
                    print(f"  [DEMO] {pid}: 20s, 48s, ext512, ext303 latents & audio saved.")
                else:
                    print(f"  [DEMO] {pid}: 20s, 48s latents & audio saved.")

            elapsed = time.time() - t0
            print(f"[DEMO] Completed latent generation & audio decode for '{tag}' ({total_clips} clips) in {elapsed:.1f}s. GPU resuming training!", flush=True)

        finally:
            T.flash_attn_func = orig_fa
            T.flash_attn_varlen_func = orig_fa_varlen
            # Swap back to y BEFORE the module returns to train mode, so training never
            # resumes from the averaged iterate. Must run even if rendering raised.
            if sf_opt is not None:
                sf_opt.train()
            if was_training:
                pl_module.train()

    def _run_with_oom_guard(self, fn, trainer, pl_module):
        """Execute fn() with automatic OOM catching and optimizer offloading."""
        torch.cuda.empty_cache()
        try:
            fn()
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower():
                raise e
            print("[OOM GUARD] Caught VRAM OOM! Offloading optimizer state buffers to CPU...")
            torch.cuda.empty_cache()

            # Offload optimizer buffers to CPU
            cpu_saved = {}
            if trainer.optimizers:
                opt = trainer.optimizers[0]
                for p, state in opt.state.items():
                    cpu_saved[p] = {}
                    for k, v in state.items():
                        if torch.is_tensor(v) and v.is_cuda:
                            cpu_saved[p][k] = v.cpu()
                            state[k] = cpu_saved[p][k]
            torch.cuda.empty_cache()

            try:
                fn()
            finally:
                # Restore optimizer buffers to GPU
                if trainer.optimizers:
                    opt = trainer.optimizers[0]
                    for p, saved in cpu_saved.items():
                        if p in opt.state:
                            for k, v in saved.items():
                                opt.state[p][k] = v.to(p.device)
            torch.cuda.empty_cache()

    def _dispatch_cpu_decoder(self, milestone_dir: Path):
        """Spawn background CPU VAE decoder subprocess."""
        decoder_script = Path(__file__).parent / "decode_latents_cpu.py"
        if not decoder_script.exists():
            print(f"[DEMO] Warning: {decoder_script} not found, skipping async CPU decode.")
            return

        cmd = [
            sys.executable,
            str(decoder_script),
            "--latent_dir", str(milestone_dir),
        ]
        log_file = milestone_dir / "cpu_decode.log"
        child_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
        with open(log_file, "a") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd="/home/kim/Projects/SAO/stable-audio-3",
                env=child_env,
                start_new_session=True
            )
        print(f"[DEMO] Spawned background CPU VAE decoder (PID {proc.pid}) -> {milestone_dir.name}/")
