"""
ZeroSep-RF: faithful flow-matching inversion separation on SA3 (RF-Solver).

The true rectified-flow analogue of ZeroSep (arXiv:2505.23625). ZeroSep itself is
edit-friendly DDPM inversion (alphas_cumprod / posterior-variance machinery) and
does NOT port to a flow-matching model. SA3's base model is rectified flow
(x_t = (1-t)*x0 + t*eps, velocity v = eps - x0), whose probability-flow ODE
dx/dt = v(x,t) is deterministic and INVERTIBLE. So we:

  1. ENCODE the mixture to its clean latent x0.
  2. INVERT x0 -> noise by integrating the ODE FORWARD in time (t: 0 -> 1) under
     the SOURCE prompt at cfg=1 (guidance during inversion destroys recoverability
     -- MusRec). RF-Solver 2nd-order Taylor correction per step keeps the round
     trip tight at fewer steps (arXiv:2411.04746):
         x <- x + dt*v + 0.5*dt^2 * v',   v' = (v(x+0.5dt*v, t+0.5dt) - v)/(0.5dt)
  3. VALIDATE the round trip: re-integrate the inverted noise back (t: 1 -> 0)
     under the SAME source prompt and compare to x0. This is the gate -- if
     reconstruction is poor, separation cannot be good; raise --steps first.
  4. SEPARATE: re-integrate the inverted noise (t: 1 -> 0) under each TARGET-source
     prompt at the (higher) target cfg. The inverted trajectory anchors the output
     to the mixture; the prompt selects the source.

Contrast with sa3_flowsep.py (FlowEdit): that is inversion-FREE (difference field,
never visits noise) and is the published SOTA on SA3. This script recovers the
actual noise state, at the cost of inversion error on SAME-L's 256-d latent.

Use a -base checkpoint (deterministic Euler velocity, live cfg). All inversion
math runs in fp32; x is cast to the model dtype only for the velocity forward.

The faithfulness controller (--eta) is what makes the re-denoise stay tied to the
input: eta=0 = clean but unrelated, eta~=0.3-0.5 = the separation sweet spot
(anchored to the actual part, still prompt-shaped), eta~=0.7 over-anchors and just
rebuilds the mix. See rf_integrate() for the mechanism.

Examples:
    python sa3_zerosep_rf.py -i mix.wav --start 400 --max-seconds 10 \
        --source-prompt "psychedelic goa trance full mix: drums, bass, acid lead" \
        --prompts "solo drum kit" "isolated bass synth" "lead synth alone" \
        --steps 28 --cfg-tar 8.0 --eta 0 0.3 0.5 0.7   # sweep the fidelity dial
"""

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def build_cond_inputs(sam, prompt, duration, batch_size, latent_sample_size, device):
    """generate()'s conditioning assembly (model.py:252-318) for one prompt."""
    import torch
    conditioning, _ = sam._build_conditioning_dicts(prompt, None, duration, batch_size)
    ct = sam.model.conditioner(conditioning, device)
    mask = torch.zeros((batch_size, 1, latent_sample_size), device=device)
    io_ch = sam.model.io_channels
    ct["inpaint_mask"] = [mask]
    ct["inpaint_masked_input"] = [
        torch.zeros((batch_size, io_ch, latent_sample_size), device=device)
    ]
    ci = sam.model.get_conditioning_inputs(ct)
    md = next(sam.model.model.parameters()).dtype
    return {k: (v.type(md) if torch.is_tensor(v) else v) for k, v in ci.items()}


def make_model_kwargs(cond, cfg_scale):
    return {**cond, "cfg_scale": cfg_scale, "batch_cfg": True,
            "rescale_cfg": True, "apg_scale": 1.0, "padding_mask": None}


def _vel(dit, x, t_scalar, mk, model_dtype):
    """Velocity v(x,t) from the SA3 DiT. x stays fp32; cast only for the forward."""
    import torch
    b = x.shape[0]
    t_clamped = float(min(max(float(t_scalar), 1e-4), 1.0 - 1e-4))
    t_ten = torch.full((b,), t_clamped, device=x.device, dtype=model_dtype)
    return dit(x.to(model_dtype), t_ten, **mk).float()


def rf_integrate(dit, x, t_grid, mk, model_dtype, taylor, tqdm, desc,
                 anchor=None, eta=0.0, tau=0.0):
    """Integrate dx/dt = v(x,t) along t_grid (ascending => invert, descending =>
    denoise). RF-Solver 2nd-order Taylor step when taylor=True. fp32 throughout.

    Faithfulness controller (anchor/eta/tau, re-denoise only): on each high-noise
    step (t >= tau) pull the predicted-clean latent z0 = x - t*v toward the source
    latent `anchor`, then rebuild the velocity:
        z0 <- (1-eta)*z0 + eta*anchor ;  v <- (x - z0)/t
    `eta` is a continuous fidelity dial (0 = free prompt edit -> clean but far from
    input; ->1 = predicted-clean forced to the source -> faithful). Releasing below
    `tau` lets the prompt shape fine detail. This z0-anchor is the proven SA3
    mean-guidance form (latch_guided / steer_chroma) and is stable, unlike the
    RF-Inversion (anchor-x)/(1-t) field which has the wrong sign and blows up at
    t->1 under SA3's descending-t Euler."""
    x = x.float().clone()
    n = t_grid.shape[-1] - 1
    use_ctrl = anchor is not None and eta > 0.0
    for i in tqdm(range(n), desc=desc):
        t0 = float(t_grid[i])
        t1 = float(t_grid[i + 1])
        dt = t1 - t0
        v = _vel(dit, x, t0, mk, model_dtype)
        if use_ctrl and t0 >= tau:
            t_c = max(t0, 1e-3)
            z0 = x - t_c * v                      # predicted-clean under target prompt
            z0 = (1.0 - eta) * z0 + eta * anchor  # pull toward the source mixture
            v = (x - z0) / t_c
            x = x + dt * v                        # plain Euler under the controller
        elif taylor:
            v_half = _vel(dit, x + 0.5 * dt * v, t0 + 0.5 * dt, mk, model_dtype)
            vp = (v_half - v) / (0.5 * dt)
            x = x + dt * v + 0.5 * dt * dt * vp
        else:
            x = x + dt * v
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--prompts", "-p", type=str, nargs="+",
                        default=["solo drum kit", "isolated bass synth",
                                 "lead synth melody alone"],
                        help="TARGET source prompts")
    parser.add_argument("--source-prompt", type=str,
                        default="full mix with drums, bass and lead")
    parser.add_argument("--model", type=str, default="medium-base")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=28,
                        help="steps per pass; Taylor doubles NFE (28 ~ 50 plain Euler)")
    parser.add_argument("--no-taylor", action="store_true",
                        help="plain reverse-Euler inversion (cheaper, less accurate)")
    parser.add_argument("--cfg-invert", type=float, default=1.0,
                        help="MUST be ~1.0; >1 ruins recoverability (MusRec)")
    parser.add_argument("--cfg-tar", type=float, default=8.0,
                        help="target cfg on the re-denoise pass (fixed; the controller "
                             "now provides anchoring, so cfg can stay high for prompt "
                             "selectivity)")
    parser.add_argument("--eta", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.7],
                        help="RF-Inversion controller strength sweep: 0 = free prompt "
                             "edit (clean but far from input), higher = pulled toward the "
                             "mixture. Swept cheaply off the shared inversion.")
    parser.add_argument("--tau", type=float, default=0.3,
                        help="controller hand-off: active for t>=tau (high noise), "
                             "released below so the prompt shapes fine detail")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out_dir", type=str, default="zerosep_rf_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if "base" not in args.model:
        logger.warning(f"'{args.model}' is not a -base checkpoint - inversion needs "
                       f"the deterministic Euler velocity and live cfg.")
    if args.cfg_invert != 1.0:
        logger.warning(f"--cfg-invert={args.cfg_invert} != 1.0: guidance during "
                       f"inversion pushes the latent where it can't be guided back.")

    import torch
    import torchaudio
    from stable_audio_3 import StableAudioModel
    from stable_audio_3.inference.sampling import build_schedule
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    taylor = not args.no_taylor

    wav, sr = torchaudio.load(args.input)
    s0 = int(args.start * sr)
    wav = wav[..., s0:s0 + int(args.max_seconds * sr)]
    duration = wav.shape[-1] / sr
    logger.info(f"input window [{args.start:.1f}, {args.start + duration:.1f}]s "
                f"-> {wav.shape} @ {sr}Hz")

    sam = StableAudioModel.from_pretrained(args.model, device=device)
    dit = sam.model.model
    model_dtype = next(dit.parameters()).dtype
    out_sr = sam.model.sample_rate
    ds = sam.model.pretransform.downsampling_ratio

    out_dir = Path(args.out_dir) / f"{Path(args.input).stem}_{int(time.time())}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out_dir / "original.wav"), wav.float().cpu(), sr)

    def decode_save(latent, name):
        pt_dtype = next(sam.model.pretransform.parameters()).dtype
        audio = sam.model.pretransform.decode(latent.to(pt_dtype))
        audio = audio.to(torch.float32).clamp(-1, 1)[0, :, :int(duration * out_sr)]
        torchaudio.save(str(out_dir / name), audio.cpu(), out_sr)

    with torch.inference_mode():
        # Encode mixture at a window-length latent grid (no silence padding).
        conditioning = [{"prompt": args.source_prompt, "seconds_total": duration}]
        audio_sample_size = sam._adapt_sample_size(conditioning, args.__dict__.get(
            "sample_size", 5292032), 0.0)
        x0, _ = sam._encode_audio_input((sr, wav), audio_sample_size, None)
        x0 = x0.to(device=device).float()
        L = x0.shape[-1]
        logger.info(f"x0 latent: {tuple(x0.shape)} ({L} frames @ {out_sr/ds:.2f}Hz)")

        # Shared schedule (descending 1->0); ascending flip for inversion.
        sig_desc = build_schedule(steps=args.steps, sigma_max=1.0,
                                  dist_shift=sam.model.sampling_dist_shift,
                                  fallback_seq_len=L, include_endpoint=True,
                                  device=device)
        if sig_desc.dim() == 2:
            sig_desc = sig_desc[0]
        sig_asc = torch.flip(sig_desc, dims=[-1])

        src_mk = make_model_kwargs(
            build_cond_inputs(sam, args.source_prompt, duration, 1, L, device),
            args.cfg_invert)

        # 1+2. Invert x0 -> noise under the source prompt.
        eps_inv = rf_integrate(dit, x0, sig_asc, src_mk, model_dtype, taylor,
                               tqdm, "invert 0->1")
        logger.info(f"inverted noise: mean={eps_inv.mean():.3f} std={eps_inv.std():.3f} "
                    f"(target ~N(0,1))")

        # 3. Round-trip gate: re-denoise under the SAME source prompt.
        x0_rec = rf_integrate(dit, eps_inv, sig_desc, src_mk, model_dtype, taylor,
                              tqdm, "recon 1->0")
        rel = float((x0_rec - x0).norm() / x0.norm())
        logger.info(f"ROUND-TRIP rel error ||x0_rec - x0|| / ||x0|| = {rel:.3f} "
                    f"({'OK' if rel < 0.5 else 'POOR - raise --steps'})")
        decode_save(x0_rec, "reconstruction.wav")

        # 4. Separate: re-denoise under each target prompt.
        # The inversion (eps_inv) is shared; only the re-denoise repeats per
        # (prompt, eta). cond is independent of eta -> build it once per prompt.
        results = [{"reconstruction_rel_error": rel}]
        for prompt in args.prompts:
            base = prompt.replace(" ", "_").replace(",", "")[:36]
            cond = build_cond_inputs(sam, prompt, duration, 1, L, device)
            tgt_mk = make_model_kwargs(cond, args.cfg_tar)
            for eta in args.eta:
                tag = f"{base}_eta{eta:g}"
                logger.info(f"--- separating: '{prompt}' (cfg_tar={args.cfg_tar}, "
                            f"eta={eta}, tau={args.tau}) ---")
                x0_edit = rf_integrate(dit, eps_inv, sig_desc, tgt_mk, model_dtype,
                                       taylor, tqdm, f"denoise {tag[:16]}",
                                       anchor=x0, eta=eta, tau=args.tau)
                decode_save(x0_edit, f"{tag}.wav")
                results.append({"prompt": prompt, "cfg_tar": args.cfg_tar,
                                "eta": eta, "tau": args.tau, "file": f"{tag}.wav"})

    (out_dir / "params.json").write_text(json.dumps(
        {"args": vars(args), "results": results}, indent=2))
    logger.info(f"done -> {out_dir}")
    logger.info("LISTEN: reconstruction.wav (inversion ceiling) first, then each "
                "<target>_eta*.wav in order of eta. eta=0 is the free prompt edit "
                "(clean but far from input); faithfulness to the actual source should "
                "increase with eta. Pick the eta that keeps the source's notes/timing "
                "while staying clean.")


if __name__ == "__main__":
    main()
