#!/usr/bin/env python
"""sa3_control_onnx.py — shared, torch-free ONNX generation core for the control-DiT.

numpy + stdlib ONLY. The ORT InferenceSession is passed in (no onnxruntime import here),
so this module is importable from any venv that has numpy. It holds the scalar control
conditioner numpy port, the schedule, the cond/uncond loader, the host-side fractional
PE, and the CFG generation loop — all extracted verbatim from dit_control_onnx_infer.py
so z0 is bit-exact for the same seed.
"""
import time

import numpy as np

SR = 44100
LATENT_DIM = 256
DS = 4096
LOCAL_ADD_DIM = 257


def schedule(steps: int, T: int, alpha_min: float = 1.0, alpha_max: float = 1.0,
             min_len: int = 256, max_len: int = 4096) -> np.ndarray:
    import math
    t = np.linspace(1.0, 0.0, steps + 1).astype(np.float64)
    log_amin, log_amax = math.log(max(alpha_min, 1e-8)), math.log(max(alpha_max, 1e-8))
    log_lo, log_hi = math.log(min_len), math.log(max(max_len, min_len + 1))
    seqc = max(min(T, max_len), min_len)
    frac = (math.log(seqc) - log_lo) / (log_hi - log_lo)
    alpha = math.exp(log_amin + frac * (log_amax - log_amin))
    if abs(alpha - 1.0) > 1e-9:
        t = alpha * t / (1.0 + (alpha - 1.0) * t)
        t[0] = 1.0
    return t


def load_cond(path):
    z = np.load(path)
    return (z["cross_attn_cond"].astype(np.float32), z["cross_attn_mask"],
            z["global_embed"].astype(np.float32))


def control_tokens_from_npz(z, raw_value: float) -> np.ndarray:
    """ScalarAttributeEncoder forward in numpy: scalar -> (1, n_tokens, control_dim)."""
    s = np.float32((raw_value - float(z["mean"])) / float(z["std"]))
    x = np.array([[s]], np.float32)
    h = x @ z["film0_w"].T + z["film0_b"]
    h = h * (1.0 / (1.0 + np.exp(-h)))                                  # SiLU
    nt, cd = int(z["n_tokens"]), int(z["control_dim"])
    gb = (h @ z["film2_w"].T + z["film2_b"]).reshape(1, nt, cd, 2)
    return (z["tokens"][None] * (1.0 + gb[..., 0]) + gb[..., 1]).astype(np.float32)


def add_fractional_positions_np(tokens: np.ndarray) -> np.ndarray:
    """numpy port of sa3_control.adapters.add_fractional_positions. Identical sin/cos
    layout. Applied HOST-SIDE when the control-DiT was exported with the adapter PE
    disabled (so the fp16 graph has no ConstantOfShape) — the result is the same as the
    in-adapter PE because PE is added to the tokens before the K/V projection."""
    b, s, d = tokens.shape
    if s <= 1:
        return tokens
    half = d // 2
    if half == 0:
        return tokens
    pos = np.linspace(0.0, 1.0, s, dtype=np.float32)
    freq = np.exp(np.linspace(0.0, 8.0, half, dtype=np.float32))
    ang = pos[:, None] * freq[None, :] * np.float64(np.pi)   # fp64 pi (matches torch's pi)
    pe = np.zeros((s, d), np.float32)
    pe[:, :half] = np.sin(ang)
    pe[:, half:half + half] = np.cos(ang)
    return (tokens + pe[None]).astype(np.float32)


def resolve_host_pe(dit_session, film_z) -> bool:
    """Decide whether the fractional PE is applied host-side, and assert the .cond.npz
    agrees with the onnx metadata stamp (catches a mismatched .cond.npz/.onnx pair that
    would double-apply or drop the PE → wrong CFG, silently)."""
    host_pe = bool(film_z["host_pe"]) if "host_pe" in film_z.files else False
    onnx_meta = dit_session.get_modelmeta().custom_metadata_map
    if "host_pe" in onnx_meta:
        onnx_host_pe = str(onnx_meta["host_pe"]).lower() == "true"
        if onnx_host_pe != host_pe:
            raise SystemExit(
                f"[fatal] PE mismatch — onnx host_pe={onnx_host_pe} but .cond.npz host_pe={host_pe}: "
                f"wrong .cond.npz/.onnx pair (PE would be applied "
                f"{'twice' if host_pe else 'zero times'}). Use the matching pair.")
    return host_pe


def make_control_tokens(film_z, onset_density: float, host_pe: bool):
    """Build (cond_tok, zero_tok). cond_tok = control ON (requested onset); zero_tok =
    control OFF (the trained null). If host_pe, apply the fractional PE to BOTH passes
    (uncond's zeros become PE, not 0) — exactly as the in-adapter PE did."""
    cond_tok = control_tokens_from_npz(film_z, onset_density)
    zero_tok = np.zeros_like(cond_tok)
    if host_pe:
        cond_tok = add_fractional_positions_np(cond_tok)
        zero_tok = add_fractional_positions_np(zero_tok)
    return cond_tok, zero_tok


def generate_z0(dit_session, *, cond, uncond, cond_tok, zero_tok, frames, steps,
                cfg_scale: float = 6.0, seed: int = 42, gain: float = 1.0,
                alpha_min: float = 1.0, alpha_max: float = 1.0):
    """CFG generation loop — byte-identical math to dit_control_onnx_infer.py lines
    163-186 so z0 is bit-exact for the same seed. cond/uncond are (cross,mask,glob)
    tuples. Returns {'z0': (1,256,frames) fp32, 'dit_loop_s': float}."""
    c_cross, c_mask, c_glob = cond
    u_cross, u_mask, u_glob = uncond
    T = frames
    local_add = np.zeros((1, LOCAL_ADD_DIM, T), np.float32)
    gain_arr = np.array([gain], np.float32)

    def dit_v(cross1, mask1, glob1, t_cur, ctrl_tok):
        return dit_session.run(None, {
            "x": x, "t": np.full((1,), t_cur, np.float32),
            "cross_attn_cond": cross1[None], "cross_attn_cond_mask": mask1[None],
            "global_embed": glob1[None], "local_add_cond": local_add,
            "control_tokens": ctrl_tok, "gain": gain_arr})[0]

    rng = np.random.default_rng(seed)
    x = rng.standard_normal((1, LATENT_DIM, T)).astype(np.float32)
    sig = schedule(steps, T, alpha_min, alpha_max)

    t0 = time.time()
    for i in range(steps):
        t_cur, dt = float(sig[i]), float(sig[i + 1] - sig[i])
        v_cond = dit_v(c_cross, c_mask, c_glob, t_cur, cond_tok)        # cond + control
        if cfg_scale == 1.0:
            v = v_cond
        else:
            v_unc = dit_v(u_cross, u_mask, u_glob, t_cur, zero_tok)     # uncond + null control
            v = v_unc + cfg_scale * (v_cond - v_unc)
        x = x + dt * v
    dit_loop_s = time.time() - t0
    return {"z0": x, "dit_loop_s": dit_loop_s}
