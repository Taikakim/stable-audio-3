"""Multi-Head RoPE Jitter — training-free loop-attractor mitigation for long-form
sliding-window continuation (LoL, arXiv 2601.16914; see the long-form-coherence
deep-research triage, papers/deep-research/2026-07-11-long-form-coherence.md).

SA3's DiT uses a HEAD-SHARED rotary embedding: `RotaryEmbedding.inv_freq` is
(dim/2,), broadcast identically across every attention head. Over a long
sliding-window rollout, RoPE's periodicity makes distant frames alias onto the
clamped prefix's positions, and because all heads share the same phase they
synchronise onto copying the prefix -> the loop attractor.

The fix: give each head its OWN base frequency, base_h = base * (1 + s * eps_h),
so the heads can't concurrently reach phase concentration on the prefix. This is
a pure inference-time perturbation; NO retraining, NO weight change.

Usage (non-invasive, reversible):
    from stable_audio_3.inference.rope_jitter import rope_jitter
    with rope_jitter(dit_model, scale=0.08, seed=0):
        latents = renderer.render_latents(...)      # jittered rollout
    # outside the context, the model is byte-identical to baseline

CAVEATS
* The model was TRAINED with shared RoPE, so any jitter is out-of-distribution.
  Keep `scale` small (~0.05-0.10); large jitter perturbs short-range positioning
  the model relies on and degrades quality even within the native window. ALWAYS
  run the quality gates (CE/PQ/zero-cross) on the in-window region, not only the
  extension — a loop that's "broken" into noise is not a win.
* The shared rope also feeds cross-attention's query rotary, so jitter touches
  that too. The loop attractor is a self-attention phenomenon; the cross-attn
  side effect is second-order. Scope to self-attn only if it matters later.
"""
from __future__ import annotations

import contextlib

import torch

import stable_audio_3.models.transformer as _T

# module-level activation flag: the patched apply only takes the per-head path
# while a jitter context is active, and only for freqs whose leading dim equals
# the head count we recorded at install (disambiguates from a batched (b,n,dim)
# freqs, since num_heads >> inference batch).
_STATE: dict = {"active": False, "num_heads": None, "orig_apply": None}


def _multihead_apply(t, freqs, scale=1):
    """apply_rotary_pos_emb that also accepts per-head freqs of shape (h, n, dim).

    Falls back to the ORIGINAL function for every other shape, so the 2D shared
    path and the batched (b,n,dim) path are byte-for-byte unchanged.
    """
    orig = _STATE["orig_apply"]
    nh = _STATE["num_heads"]
    is_per_head = (
        _STATE["active"] and freqs.ndim == 3 and t.ndim == 4
        and nh is not None and freqs.shape[0] == nh and t.shape[1] == nh
    )
    if not is_per_head:
        return orig(t, freqs, scale)

    out_dtype = t.dtype
    dtype = torch.promote_types(torch.promote_types(t.dtype, freqs.dtype), torch.float32)
    freqs, t = freqs.to(dtype), t.to(dtype)
    rot_dim, seq_len = freqs.shape[-1], t.shape[-2]
    freqs = freqs[:, -seq_len:, :].unsqueeze(0)          # (1, h, n, dim) -> broadcasts over batch
    t_rot, t_unrot = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos() * scale) + (_T.rotate_half(t_rot) * freqs.sin() * scale)
    return torch.cat((t_rot.to(out_dtype), t_unrot.to(out_dtype)), dim=-1)


def _per_head_inv_freq(inv_freq, num_heads, scale, seed):
    """(num_heads, dim/2) per-head inv_freq that PERTURBS the module's real buffer.

    A per-head base base_h = base*(1+scale*eps_h) is equivalent to scaling the
    original inv_freq by (1+scale*eps_h)**(-exps), exps = arange(0,dim,2)/dim.
    This reuses the exact `inv_freq` buffer, so scale=0 gives inv_freq_h == inv_freq
    BIT-EXACTLY (factor = 1**(-exps) = 1) — a genuine no-op, no base recovery.
    """
    dim = inv_freq.shape[0] * 2
    exps = (torch.arange(0, dim, 2, dtype=torch.float32) / dim).to(inv_freq.device)  # (dim/2,)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    eps = torch.randn(num_heads, generator=g).to(inv_freq.device)                    # (h,)
    factor = (1.0 + scale * eps)[:, None] ** (-exps[None, :])                         # (h, dim/2)
    return inv_freq[None, :] * factor                                                # (h, dim/2)


def find_rope(model):
    """Return the first RotaryEmbedding module in `model` (the DiT's self.rope)."""
    for _, m in model.named_modules():
        if isinstance(m, _T.RotaryEmbedding):
            return m
    raise RuntimeError("no RotaryEmbedding found in model — is this the DiT?")


def infer_num_heads(model):
    """Best-effort head count from the first attention module."""
    for _, m in model.named_modules():
        nh = getattr(m, "num_heads", None)
        if isinstance(nh, int) and nh > 0:
            return nh
    raise RuntimeError("could not infer num_heads from model")


@contextlib.contextmanager
def rope_jitter(model, scale=0.08, seed=0, num_heads=None):
    """Context manager: install per-head RoPE jitter on `model`'s DiT rope, restore on exit.

    `scale`=0 is a verified no-op (bit-identical to baseline). Reversible and
    thread-unsafe (patches a module global) — run one jittered rollout at a time.
    """
    rope = find_rope(model)
    nh = int(num_heads) if num_heads else infer_num_heads(model)
    dim = rope.inv_freq.shape[0] * 2
    # recovered only for the info dict / logging; the jitter perturbs inv_freq directly
    base = float(rope.inv_freq[1].pow(-dim / 2.0)) if rope.inv_freq.shape[0] > 1 else 10000.0

    per_head = _per_head_inv_freq(rope.inv_freq, nh, scale, seed)
    orig_forward = rope.forward
    orig_from_len = rope.forward_from_seq_len

    if rope.scale is not None:
        raise NotImplementedError("RoPE jitter does not support xpos (rope.scale set); "
                                  "SA3's DiT uses use_xpos=False so this should not fire.")

    def jittered_forward(t):
        t = t.to(torch.float32) / rope.interpolation_factor
        # (h, n, dim/2) -> cat -> (h, n, dim); scale is None on SA3 -> return 1.0
        freqs = torch.einsum("n, h j -> h n j", t, per_head.to(t.device))
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs, 1.0

    def jittered_from_len(seq_len):
        tt = torch.arange(seq_len, device=rope.inv_freq.device)
        return jittered_forward(tt)

    _STATE["orig_apply"] = _T.apply_rotary_pos_emb
    _STATE["num_heads"] = nh
    _STATE["active"] = True
    _T.apply_rotary_pos_emb = _multihead_apply
    rope.forward = jittered_forward
    rope.forward_from_seq_len = jittered_from_len
    try:
        yield {"scale": scale, "seed": seed, "num_heads": nh, "dim": dim, "base": base}
    finally:
        rope.forward = orig_forward
        rope.forward_from_seq_len = orig_from_len
        _T.apply_rotary_pos_emb = _STATE["orig_apply"]
        _STATE.update(active=False, num_heads=None, orig_apply=None)
