"""Incantation mask — bar the time-varying text prompt from the CLAMPED HISTORY frames
in sliding-window inpaint-continuation (Incantation, matrixteam-ai; see the long-form
coherence deep-research triage, papers/deep-research/2026-07-11-long-form-coherence.md).

The loop attractor in `longform.py` is conditioning-driven: each new window is clamped to
the previous window's tail (the "committed history") and generated under a prompt. When the
prompt is TIME-VARYING (a prompt arc), letting the new prompt's cross-attention reach the
clamped-history frames causes *temporal cross-contamination* — the new prompt rewrites the
past's relationship and the output echoes it forward instead of moving on. The fix: restrict
text cross-attention to the NOISY TARGET frames only; the clamped-history frames get NO text
conditioning (they are fixed anyway), so the model must generate novel material for the new
section.

Implementation: a forward hook on every block's `cross_attn_scale` (the exact per-position
contribution `cross_attn_scale(cross_attn(x))` added to the residual) zeros the first
`prefix` positions — the clamped history sits at positions [0, prefix). The per-window prefix
is threaded automatically by wrapping the generator's `generate()`. Reversible, no weight
change, no retrain.

Usage:
    from stable_audio_3.inference.incantation_mask import incantation_mask
    with incantation_mask(dit, generator):
        latents = renderer.render_latents(...)      # masked rollout
    # outside the context the model + generator are byte-identical to baseline

CAVEAT: this only bites when prefix > 0 (continuation windows). The first window and hard
transitions (prefix=0 in longform's Approach A) are unaffected. It is a no-op on same-prompt
rollouts where the point is continuity, and is aimed squarely at the prompt-arc case.
"""
from __future__ import annotations

import contextlib

_STATE = {"active": False, "prefix": 0}


def _hook(_module, _inp, out):
    """Zero the clamped-history query positions of a cross-attn contribution.

    out is the residual contribution, shape (batch, T, dim). Only acts while a mask
    context is active and there is a clamped prefix; otherwise returns None (unchanged).
    """
    if not _STATE["active"] or _STATE["prefix"] <= 0:
        return None
    p = _STATE["prefix"]
    if out.dim() == 3 and out.shape[1] > p:
        out = out.clone()
        out[:, :p, :] = 0
        return out
    return None


def install_hooks(dit):
    """Hook every `*cross_attn_scale` module (the contribution added to the residual).

    Falls back to `*cross_attn` if no scale modules are found. Returns handles + the
    matched-module count so the caller can assert it found the cross-attn stack.
    """
    handles, n = [], 0
    for name, mod in dit.named_modules():
        if name.endswith("cross_attn_scale"):
            handles.append(mod.register_forward_hook(_hook)); n += 1
    if n == 0:  # fallback: hook the cross-attn module directly (scale may be identity)
        for name, mod in dit.named_modules():
            if name.endswith("cross_attn"):
                handles.append(mod.register_forward_hook(_hook)); n += 1
    return handles, n


@contextlib.contextmanager
def incantation_mask(dit, generator, require_hooks=True):
    """Install the Incantation mask on `dit`, threading `generator.generate`'s prefix.

    `dit` = the DiT module whose blocks carry the cross-attn (e.g. generator.inner.model).
    The generator's `generate(prompt, prefix_latents, prefix_frames, n_frames, seed)` is
    wrapped so each window's `prefix_frames` drives the mask. Restores everything on exit.
    """
    handles, n = install_hooks(dit)
    if require_hooks and n == 0:
        raise RuntimeError("incantation_mask: found no cross_attn(_scale) modules to hook")
    orig_generate = generator.generate

    def wrapped(prompt, prefix_latents, prefix_frames, n_frames, seed):
        _STATE["prefix"] = int(prefix_frames or 0)
        return orig_generate(prompt, prefix_latents, prefix_frames, n_frames, seed)

    _STATE["active"] = True
    generator.generate = wrapped
    try:
        yield {"n_hooks": n}
    finally:
        for h in handles:
            h.remove()
        generator.generate = orig_generate
        _STATE.update(active=False, prefix=0)
