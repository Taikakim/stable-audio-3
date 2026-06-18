# stable_audio_3/inference/fifo_infinite.py
"""InfiniteAudio-style FIFO ("diagonal denoising") generation for SA3.

PROTOTYPE / UNTESTED. See docs/INFINITE_AUDIO_FIFO.md for the full spec, the
architectural rationale, and the validation plan.

Core idea (Jung et al. 2025, arXiv:2506.03020; FIFO-Diffusion lineage): hold a
fixed-size window of `window` latent frames where each frame sits at a *different*
noise level along the time axis. One model forward per step advances every frame
one notch toward clean; the fully-denoised front frame is popped (emitted) and a
fresh pure-noise frame is pushed at the back. Unbounded length; the sampling loop
runs in constant memory (fixed window). Collecting + decoding the emitted latents is
linear in output length — for true streaming, pop frames to disk or decode in
overlapping crossfaded chunks.

SA3's DiT only accepts a *scalar-per-batch* timestep (models/dit.py:239), so this
module installs a guarded monkey-patch that makes the timestep conditioning
*per-token* when (and only when) a 2-D timestep ``(B, T)`` is passed. The original
scalar path is preserved exactly, so installing the patch is safe for normal
generation. CFG is done manually here (two unit-scale passes) because the DiT's
internal CFG block assumes a scalar sigma (dit.py:466/473/479/577/581).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# --------------------------------------------------------------------------- #
# Per-frame timestep monkey-patch (guarded, idempotent)
# --------------------------------------------------------------------------- #

_ORIG: dict = {}
_INSTALLED = False


def install_per_frame_timestep_patch() -> None:
    """Patch DiT/transformer to accept a per-frame timestep ``(B, T)``.

    Idempotent. The scalar-``(B,)`` path is untouched: every patched method
    falls straight through to the original unless a 2-D timestep / 3-D global
    cond is in play.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from stable_audio_3.models.dit import DiffusionTransformer
    from stable_audio_3.models.transformer import ContinuousTransformer, TransformerBlock

    _ORIG["dit_forward"] = DiffusionTransformer._forward
    _ORIG["ct_forward"] = ContinuousTransformer.forward
    _ORIG["blk_forward"] = TransformerBlock.forward

    DiffusionTransformer._forward = _dit_forward_patched
    ContinuousTransformer.forward = _ct_forward_patched
    TransformerBlock.forward = _blk_forward_patched

    _INSTALLED = True


def uninstall_per_frame_timestep_patch() -> None:
    """Restore the original methods (mainly for tests)."""
    global _INSTALLED
    if not _INSTALLED:
        return
    from stable_audio_3.models.dit import DiffusionTransformer
    from stable_audio_3.models.transformer import ContinuousTransformer, TransformerBlock

    DiffusionTransformer._forward = _ORIG["dit_forward"]
    ContinuousTransformer.forward = _ORIG["ct_forward"]
    TransformerBlock.forward = _ORIG["blk_forward"]
    _INSTALLED = False


def _compute_pertoken_gcond(dit, t, global_embed):
    """Per-frame global conditioning (pre-embedder), mirroring dit._forward.

    Returns ``(B, T, embed_dim)`` = projected text/seconds global cond (broadcast
    over T) + per-frame timestep embedding. This is the *input* to the
    transformer's ``global_cond_embedder`` (which maps embed_dim -> 6*dim).
    """
    # Mirror dit._forward's timestep embedding, but per-frame.
    t_cond = dit._t_to_logsnr_cond(t) if dit.timestep_features_logsnr else t  # (B, T)
    model_dtype = next(dit.parameters()).dtype
    t_cond = t_cond.to(model_dtype)
    # FourierFeatures does `input @ weight.T` (blocks.py:46), weight.T is (1, F//2):
    # (B,T,1)@(1,F//2) -> (B,T,F//2), then cat(cos,sin) -> (B,T,F).
    ts_embed = dit.to_timestep_embed(dit.timestep_features(t_cond[..., None]))  # (B, T, embed)

    if dit.timestep_cond_type != "global":
        raise NotImplementedError(
            f"per-frame FIFO only supports timestep_cond_type='global', "
            f"got {dit.timestep_cond_type!r}"
        )

    if global_embed is not None:
        ge = dit.to_global_embed(global_embed.to(model_dtype))  # (B, embed)
        gtok = ge[:, None, :] + ts_embed                        # (B, T, embed)
    else:
        gtok = ts_embed
    return gtok


def _dit_forward_patched(self, x, t, *args, **kwargs):
    """DiffusionTransformer._forward with per-frame timestep support.

    For scalar ``t`` (dim 1) this is a straight pass-through to the original. For
    per-frame ``t`` (dim 2, shape ``(B, T)``) we precompute a per-token global
    cond, stash it on the transformer, and delegate to the original with a
    representative scalar timestep — the transformer wrapper substitutes the
    stashed per-token cond, so the scalar path's global cond is discarded.
    """
    if t.dim() != 2:
        return _ORIG["dit_forward"](self, x, t, *args, **kwargs)

    global_embed = kwargs.get("global_embed", None)
    gtok = _compute_pertoken_gcond(self, t, global_embed)  # (B, T, embed)

    self.transformer._fifo_gtok = gtok
    try:
        # t_rep exists ONLY to drive the x-side paths inside ORIG _forward
        # (preprocess_conv, patchify, RoPE seq-len). The scalar global cond ORIG
        # computes from t_rep (to_global_embed @ dit.py:202 + to_timestep_embed(t_rep)
        # @ 239 + the add @ 245) is intentionally DEAD work: _ct_forward_patched
        # overwrites global_cond with the per-token gtok. Verified: conditioning is
        # not dropped (the gtok carries text+seconds + per-frame timestep).
        t_rep = t.float().mean(dim=1)  # (B,)
        return _ORIG["dit_forward"](self, x, t_rep, *args, **kwargs)
    finally:
        self.transformer._fifo_gtok = None


def _ct_forward_patched(self, x, *args, **kwargs):
    """ContinuousTransformer.forward: swap in the stashed per-token global cond.

    When a per-frame FIFO cond is stashed, replace the (scalar) ``global_cond``
    with the per-token one ``(B, T, embed)``. The original forward then runs the
    embedder on it (-> ``(B, T, 6*dim)``, since nn.Linear acts on the last dim)
    and hands 3-D cond to the blocks, which the block patch handles.
    """
    gtok = getattr(self, "_fifo_gtok", None)
    if gtok is not None:
        kwargs["global_cond"] = gtok
    return _ORIG["ct_forward"](self, x, *args, **kwargs)


def _blk_forward_patched(self, x, *args, **kwargs):
    """TransformerBlock.forward with per-token adaLN.

    The original assumes a 2-D ``global_cond`` ``(B, 6*dim)`` and broadcasts it
    over tokens via ``.unsqueeze(1)``. For a 3-D per-token cond ``(B, T, 6*dim)``
    we instead modulate each token by its own row. The cond covers only the
    ``T`` latent tokens, which are the *last* T tokens of x (memory + prepend are
    prepended at the front, see transformer.py:1205-1214), so we left-pad the
    cond to x's token length, padding front rows (memory/prepend) with the mean
    modulation. Falls through to the original for the scalar path.
    """
    global_cond = kwargs.get("global_cond", None)
    if not (self.global_cond_dim is not None and self.global_cond_dim > 0
            and global_cond is not None and global_cond.dim() == 3):
        return _ORIG["blk_forward"](self, x, *args, **kwargs)

    rotary_pos_emb = kwargs.get("rotary_pos_emb", None)
    if rotary_pos_emb is None and self.add_rope:
        rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

    seq = x.shape[-2]
    T = global_cond.shape[1]
    if T < seq:
        pad = global_cond.mean(dim=1, keepdim=True).expand(x.shape[0], seq - T, global_cond.shape[-1])
        gc = torch.cat([pad, global_cond], dim=1)  # (B, seq, 6*dim)
    elif T == seq:
        gc = global_cond
    else:
        raise ValueError(f"per-token global_cond length {T} exceeds sequence length {seq}")

    # Per-token adaLN: NO unsqueeze (the 3-D cond already has the token axis).
    scale_self, shift_self, gate_self, scale_ff, shift_ff, gate_ff = (
        (self.to_scale_shift_gate + gc).chunk(6, dim=-1)
    )

    context = kwargs.get("context", None)
    cross_attn_rotary_pos_emb = kwargs.get("cross_attn_rotary_pos_emb", None)
    self_attention_block_mask = kwargs.get("self_attention_block_mask", None)
    self_attention_score_mod = kwargs.get("self_attention_score_mod", None)
    cross_attention_block_mask = kwargs.get("cross_attention_block_mask", None)
    cross_attention_score_mod = kwargs.get("cross_attention_score_mod", None)
    self_attention_flash_sliding_window = kwargs.get("self_attention_flash_sliding_window", None)
    cross_attention_flash_sliding_window = kwargs.get("cross_attention_flash_sliding_window", None)
    local_add_cond = kwargs.get("local_add_cond", None)
    modular_local_cond = kwargs.get("modular_local_cond", None)
    padding_mask = kwargs.get("padding_mask", None)
    varlen_metadata = kwargs.get("varlen_metadata", None)

    # self-attention with adaLN
    residual = x
    x = self.pre_norm(x)
    x = x * (1 + scale_self) + shift_self
    x = self.self_attn(
        x, rotary_pos_emb=rotary_pos_emb, flex_attention_block_mask=self_attention_block_mask,
        flex_attention_score_mod=self_attention_score_mod,
        flash_attn_sliding_window=self_attention_flash_sliding_window,
        padding_mask=padding_mask, varlen_metadata=varlen_metadata,
    )
    x = x * torch.sigmoid(1 - gate_self)
    x = self.self_attn_scale(x)
    x = x + residual

    if context is not None and self.cross_attend:
        if cross_attn_rotary_pos_emb is not None:
            x = x + self.cross_attn_scale(self.cross_attn(
                self.cross_attend_norm(x), rotary_pos_emb=rotary_pos_emb,
                rotary_pos_emb_k=cross_attn_rotary_pos_emb, context=context,
                flex_attention_block_mask=cross_attention_block_mask,
                flex_attention_score_mod=cross_attention_score_mod,
                flash_attn_sliding_window=cross_attention_flash_sliding_window))
        else:
            x = x + self.cross_attn_scale(self.cross_attn(
                self.cross_attend_norm(x), context=context,
                flex_attention_block_mask=cross_attention_block_mask,
                flex_attention_score_mod=cross_attention_score_mod,
                flash_attn_sliding_window=cross_attention_flash_sliding_window))

    if self.conformer is not None:
        x = x + self.conformer_scale(self.conformer(x))

    x = self._apply_local_conditioning(x, local_add_cond, modular_local_cond)

    # feedforward with adaLN
    residual = x
    x = self.ff_norm(x)
    x = x * (1 + scale_ff) + shift_ff
    x = self.ff(x, varlen_metadata=varlen_metadata)
    x = x * torch.sigmoid(1 - gate_ff)
    x = self.ff_scale(x)
    x = x + residual
    return x


# --------------------------------------------------------------------------- #
# FIFO sampler
# --------------------------------------------------------------------------- #

@dataclass
class FIFOConfig:
    window: int = 256          # n latent frames held in the buffer
    n_buffer: int = 0          # frozen-context slots at the front (see build_slot_sigmas NOTE)
    emit_frames: int = 512     # latent frames to emit (post-warmup)
    warmup_frames: int | None = None  # discarded leading pops; default = window
    cfg_scale: float = 6.0
    sigma_min: float = 1e-3    # σ of the most-denoised non-zero slot
    seed: int = 0


def build_slot_sigmas(cfg: FIFOConfig, device, dtype=torch.float32, dist_shift=None) -> torch.Tensor:
    """Fixed per-slot σ: slot 0 = most denoised, slot window-1 = pure noise (σ=1).

    The ramp slots ``[n_buffer, window)`` follow SA3's *native* schedule shape
    (``build_schedule`` warped by the model's ``dist_shift``; reversed to ascending
    σ), so each frame's per-step Δσ as it migrates down the buffer corresponds to a
    real native sampling step rather than an ad-hoc ramp.

    NOTE on the buffer zone: slots ``[0, n_buffer)`` are pinned at σ_min, giving them
    Δσ≈0 (frozen) — they are *held* model outputs, NOT clean ground-truth context.
    This does **not** replicate InfiniteAudio's buffer zone (clean known-good frames
    that close the train/inference gap). The SA3-native anchor is the inpaint clamp
    (inpaint_mask=1 + inpaint_masked_input on those slots), which needs per-step
    bookkeeping — deferred to v2 (see docs/INFINITE_AUDIO_FIFO.md). Default n_buffer=0.
    """
    from stable_audio_3.inference.sampling import build_schedule

    n = cfg.window
    f = max(0, min(cfg.n_buffer, n - 1))
    sig = torch.empty(n, device=device, dtype=dtype)
    sig[:f] = cfg.sigma_min
    ramp_len = n - f
    # SA3-native schedule: descending σ_max->0 of length ramp_len, reversed to ascending.
    sched = build_schedule(
        steps=ramp_len - 1, sigma_max=1.0, dist_shift=dist_shift,
        effective_seq_len=n, include_endpoint=True, device=device,
    )  # (ramp_len,) descending 1.0 -> 0
    ramp = sched.flip(0).to(dtype).clamp(min=cfg.sigma_min)  # ascending σ_min -> 1.0
    sig[f:] = ramp
    sig[-1] = 1.0  # ensure the enqueue slot is exactly pure-noise level
    return sig


@torch.no_grad()
def sample_fifo_infinite(
    wrapper,                # DiTWrapper, i.e. model.model.model
    cond_pos: dict,         # DiTWrapper kwargs for the prompt (window-sized input_concat)
    cond_neg: dict,         # DiTWrapper kwargs for the unconditional/negative pass
    *,
    io_channels: int,
    cfg: FIFOConfig,
    device,
    dtype=torch.float32,
    dist_shift=None,
    progress: bool = True,
):
    """Run FIFO diagonal denoising. Returns emitted latents ``(B, C, emit_frames)``.

    All guidance/sampler math is fp32; the wrapper casts to model dtype internally
    for the velocity forward (see DiffusionTransformer.forward). ``dist_shift`` is
    the model's ``sampling_dist_shift`` (``model.model.sampling_dist_shift``) used to
    shape the per-slot σ schedule; None falls back to a linear ramp.
    """
    install_per_frame_timestep_patch()

    # Per-frame (2-D) t makes sigma a (B,n) tensor; the DiT's LoRA-interval guards
    # (dit.py:466/473) do `interval[0] <= sigma[0] <= interval[1]`, which raises an
    # ambiguous-truth-value error on a length-n sigma[0]. Fail fast with a clear msg.
    try:
        from stable_audio_3.models.lora.utils import has_lora
        if has_lora(getattr(wrapper, "model", wrapper)):
            raise ValueError(
                "FIFO per-frame sampling is incompatible with a loaded LoRA: the DiT's "
                "sigma[0] interval checks (dit.py:466/473) assume a scalar timestep and "
                "raise on a (B,n) sigma. Unload the LoRA first."
            )
    except ImportError:
        pass

    n = cfg.window
    B = 1
    warmup = cfg.warmup_frames if cfg.warmup_frames is not None else n

    sig = build_slot_sigmas(cfg, device, dtype, dist_shift=dist_shift)  # (n,)
    # Per-slot Euler dt = σ_target - σ_curr, where slot k advances toward slot k-1.
    dt = torch.empty(n, device=device, dtype=dtype)
    dt[1:] = sig[:-1] - sig[1:]                           # (k>=1) negative
    dt[0] = -sig[0]                                       # slot 0 -> σ=0
    dt = dt.view(1, 1, n)

    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    x = torch.randn(B, io_channels, n, generator=gen, device=device, dtype=dtype)

    tau = sig.view(1, n).expand(B, n).contiguous()       # (B, n) per-frame timestep

    outputs = []
    total = warmup + cfg.emit_frames
    cfg_s = float(cfg.cfg_scale)
    fwd_per_iter = 1 if cfg_s == 1.0 else 2
    print(f"[fifo] window={n} warmup={warmup} emit={cfg.emit_frames} -> "
          f"{total} iters x {fwd_per_iter} fwd = {total * fwd_per_iter} model forwards")

    rng = range(total)
    if progress:
        from tqdm import tqdm
        rng = tqdm(rng, desc="FIFO")

    for it in rng:
        v_c = wrapper(x, tau, cfg_scale=1.0, **cond_pos).float()
        if cfg_s != 1.0:
            v_u = wrapper(x, tau, cfg_scale=1.0, **cond_neg).float()
            v = v_u + cfg_s * (v_c - v_u)
        else:
            v = v_c

        x = x + dt * v                                   # per-frame Euler

        popped = x[..., :1].clone()                      # slot 0 now at σ=0
        if it >= warmup:
            outputs.append(popped)

        # shift toward the front, enqueue fresh noise at the back (σ=1)
        new_noise = torch.randn(B, io_channels, 1, generator=gen, device=device, dtype=dtype)
        x = torch.cat([x[..., 1:], new_noise], dim=-1)

    return torch.cat(outputs, dim=-1)                    # (B, C, emit_frames)


# --------------------------------------------------------------------------- #
# Window conditioning builder (mirrors StableAudioModel.generate, window-sized)
# --------------------------------------------------------------------------- #

def build_window_conditioning(model, prompt: str, window: int, device, *, batch_size: int = 1):
    """Build (cond_pos, cond_neg) DiTWrapper kwargs for a FIFO window.

    Mirrors generate()'s conditioning path (model.py:161/254/302) but sizes the
    inpaint conditioning to ``window`` frames and sets seconds_total to the window
    duration. cond_neg uses an empty prompt.

    For this model config the inpaint mask/masked-input enter as ``local_add_cond``
    (verified live: ``local_add_cond_ids=['inpaint_mask','inpaint_masked_input']``,
    ``input_concat_ids=[]``) — additive, projected per-frame via ``to_local_embed``
    and left-padded to the working sequence length. The block patch forwards
    ``local_add_cond`` correctly.
    """
    inner = model.model  # ConditionedDiffusionModelWrapper
    fps = inner.sample_rate / inner.pretransform.downsampling_ratio
    win_seconds = window / fps
    io_channels = inner.io_channels
    model_dtype = next(inner.model.parameters()).dtype

    def _cond_inputs(text):
        conditioning, _ = model._build_conditioning_dicts(text, None, win_seconds, batch_size)
        ctensors = inner.conditioner(conditioning, device)
        # window-sized inpaint conditioning (pure generation -> all zeros; rides in
        # as local_add_cond for this config)
        mask = torch.zeros((batch_size, 1, window), device=device)
        masked_input = torch.zeros((batch_size, io_channels, window), device=device)
        ctensors["inpaint_mask"] = [mask]
        ctensors["inpaint_masked_input"] = [masked_input]
        ci = inner.get_conditioning_inputs(ctensors)
        return {k: (v.type(model_dtype) if torch.is_tensor(v) else v) for k, v in ci.items()}

    cond_pos = _cond_inputs(prompt)
    cond_neg = _cond_inputs("")
    return cond_pos, cond_neg, io_channels
