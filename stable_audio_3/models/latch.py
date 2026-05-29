"""Latent-Control Head (LatCH) for Stable Audio 3.

A lightweight bidirectional transformer (RoPE) that predicts an MIR control
time-series from noisy SAME latents. Used as Training-Free Guidance in the
gradient-enabled Euler sampler (see ``inference/latch_guided.py``).

The architecture is identical to the stable-audio-tools head; only the default
``in_channels`` differs (256 for the SA3 ``same-s``/``same-l`` latent space vs.
64 for SAO). ``load_latch_from_checkpoint`` infers all shapes from the saved
state dict, so it round-trips either repo's checkpoints.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.cos(emb), torch.sin(emb)


def apply_rotary_emb(x, cos, sin):
    b, s, h, d = x.shape
    d_half = d // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    x_rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return (x * cos) + (x_rotated * sin)


class LatCHAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class LatCHBlock(nn.Module):
    """Plain pre-norm transformer block, used for t_injection='concat' (legacy)
    and 't_injection=film' (where t is injected once after latent_proj)."""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LatCHAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class LatCHBlockAdaLN(nn.Module):
    """DiT-style adaLN-zero block. Each block receives t_emb and produces 6
    modulators (γ1,β1,α1 for attention path; γ2,β2,α2 for MLP path). The final
    linear is zero-initialised so all modulators are 0 at step 0 → residual
    contributions vanish (block is identity) and the model warms up safely.

    Mirrors stable_audio_tools.models.latch.LatCHBlockAdaLN — by design the
    state-dict keys are bit-identical so SAT and SA3 LatCH checkpoints
    round-trip through either repo's loader.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = LatCHAttention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, dim),
        )
        # adaLN_mod[0] = SiLU, [1] = Linear(dim, 6*dim). Loader keys off
        # ".adaLN_mod.1.weight" — keep this structure stable.
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_mod[-1].weight)
        nn.init.zeros_(self.adaLN_mod[-1].bias)

    def forward(self, x, t_emb, cos, sin, mods=None):
        """Optionally take precomputed modulators from a TimeConditioningCache.

        mods, if given, is a 6-tuple of tensors (g1, b1, a1, g2, b2, a2). Each
        tensor's leading dim broadcasts over x's batch — the cache stores them
        as (1, dim).
        """
        if mods is None:
            g1, b1, a1, g2, b2, a2 = self.adaLN_mod(t_emb).chunk(6, dim=-1)
        else:
            g1, b1, a1, g2, b2, a2 = mods
        n1 = self.norm1(x) * (1 + g1.unsqueeze(1)) + b1.unsqueeze(1)
        x = x + a1.unsqueeze(1) * self.attn(n1, cos, sin)
        n2 = self.norm2(x) * (1 + g2.unsqueeze(1)) + b2.unsqueeze(1)
        x = x + a2.unsqueeze(1) * self.mlp(n2)
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        half_dim = self.frequency_embedding_size // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)
        return self.mlp(emb)


class LatCH(nn.Module):
    """Bidirectional transformer over latents predicting a control feature.

    For dim=256, depth=6, num_heads=8 this is ~5–7 M params depending on
    t_injection.

    t_injection ∈ {'concat', 'film', 'adaln_zero'}:
      - 'concat':     legacy — prepend t_emb as an extra sequence token (T=257).
                      Default for back-compat with existing SA3 Phase-1 checkpoints.
      - 'film':       t_film(t_emb).chunk(2) → (scale, shift) once after latent_proj.
                      T stays at 256 (FA-aligned: ~17% faster attention than T=257).
      - 'adaln_zero': DiT-style per-block (γ,β,α) modulators. T=256. Per LATCH_RESULTS
                      §17, the chosen winner on quality (val 0.1682). Unlocks the
                      TimeConditioningCache for inference (attach via
                      attach_time_cache(); the forward path picks it up).
    """

    def __init__(
        self,
        in_channels=256,
        out_channels=1,
        dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        t_injection="concat",
    ):
        super().__init__()
        assert t_injection in ("concat", "film", "adaln_zero"), \
            f"unknown t_injection={t_injection!r}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.depth = depth
        self.t_injection = t_injection

        self.latent_proj = nn.Linear(in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        # FiLM head: outputs (scale, shift) applied once after latent_proj.
        # Identity-init so x is untouched at step 0.
        if t_injection == "film":
            self.t_film = nn.Linear(dim, 2 * dim)
            nn.init.zeros_(self.t_film.weight)
            nn.init.zeros_(self.t_film.bias)
        block_cls = LatCHBlockAdaLN if t_injection == "adaln_zero" else LatCHBlock
        self.blocks = nn.ModuleList(
            [block_cls(dim, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm_final = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, out_channels)
        self.rotary_emb = RotaryEmbedding(dim // num_heads)

    def forward(self, x, t):
        """x: [B, in_channels, T] noisy latents; t: [B] timesteps. -> [B, out_channels, T].

        For t_injection='adaln_zero': if self._time_cache is set (a
        TimeConditioningCache) AND t is uniform across the batch AND the value
        is cached, t_emb and per-block modulators are looked up rather than
        recomputed. Falls back to live calc otherwise — no quality regression.
        """
        B, C, T_seq = x.shape
        x = x.transpose(1, 2)
        x = self.latent_proj(x)

        # ── Try the time-conditioning cache, if attached. The sampler always
        # passes a uniform t (torch.full((B,), value)) so we can cache by
        # scalar value. Non-uniform t falls through to the live path.
        cache_entry = None
        cache = getattr(self, "_time_cache", None)
        if cache is not None and t.numel() > 0:
            t_first = t.flatten()[0]
            if t.numel() == 1 or torch.all(t == t_first):
                cache_entry = cache.get(float(t_first.item()))

        if cache_entry is not None:
            t_emb = cache_entry["t_emb"]
            block_mods_cached = cache_entry["modulators"]
        else:
            t_emb = self.t_embedder(t)
            block_mods_cached = [None] * len(self.blocks)

        if self.t_injection == "film":
            scale, shift = self.t_film(t_emb).chunk(2, dim=-1)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
            pos = torch.arange(T_seq, device=x.device).float()
            cos, sin = self.rotary_emb(pos)
            for block in self.blocks:
                x = block(x, cos, sin)
            x = self.norm_final(x)
        elif self.t_injection == "adaln_zero":
            pos = torch.arange(T_seq, device=x.device).float()
            cos, sin = self.rotary_emb(pos)
            for i, block in enumerate(self.blocks):
                x = block(x, t_emb, cos, sin, mods=block_mods_cached[i])
            x = self.norm_final(x)
        else:  # concat (legacy)
            x = torch.cat([t_emb.unsqueeze(1), x], dim=1)
            pos = torch.arange(T_seq + 1, device=x.device).float()
            cos, sin = self.rotary_emb(pos)
            for block in self.blocks:
                x = block(x, cos, sin)
            x = self.norm_final(x)
            x = x[:, 1:, :]  # strip t-token

        out = self.out_proj(x)
        return out.transpose(1, 2)

    # ── Time-conditioning cache: lazy wiring ────────────────────────────────
    def attach_time_cache(self, cache) -> None:
        """Attach a TimeConditioningCache. None or absent => live calc only."""
        self._time_cache = cache

    def detach_time_cache(self) -> None:
        """Remove the attached cache and revert to live calc."""
        if hasattr(self, "_time_cache"):
            self._time_cache = None


def load_latch_from_checkpoint(path: str, device="cpu") -> LatCH:
    """Load a LatCH head, inferring architecture from the saved state dict.

    Supports three formats:
      - Legacy bare state_dict (flat dict of name -> tensor).
      - SA3 trainer dict: ``{"state_dict": ..., "feature_name": ..., ...}``.
      - FusionOpt dict: as above, plus ``"averaged_state_dict"`` — the
        Schedule-Free averaged iterate x_t (the deployable model per the SF
        contract). Preferred over the live z_t in ``state_dict``.

    t_injection is taken from metadata if present, else inferred from state-dict
    keys (per-block ``.adaLN_mod.1.weight`` → adaln_zero; top-level ``t_film.*``
    → film; otherwise concat). This makes SA3 and SAT LatCH checkpoints
    round-trip through either repo's loader.

    The returned model carries a ``.metadata`` attribute with everything in
    the checkpoint other than the state-dict tensors.
    """
    raw = torch.load(path, map_location=device, weights_only=True)

    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        # FusionOpt heads: prefer averaged x_t over the live z_t for inference, but
        # MERGE with state_dict so non-parameter buffers (rotary_emb.inv_freq, etc.)
        # that FusionOpt's average_state_dict doesn't track come along automatically.
        if "averaged_state_dict" in raw and isinstance(raw["averaged_state_dict"], dict):
            state = {**raw["state_dict"], **raw["averaged_state_dict"]}
        else:
            state = raw["state_dict"]
        metadata = {
            k: v for k, v in raw.items()
            if k not in ("state_dict", "averaged_state_dict")
        }
    else:
        state = raw
        metadata = {}

    in_channels = state["latent_proj.weight"].shape[1]
    dim = state["latent_proj.weight"].shape[0]
    out_channels = state["out_proj.weight"].shape[0]
    depth = sum(1 for k in state if k.endswith(".attn.qkv.weight"))
    num_heads_dim = state["rotary_emb.inv_freq"].shape[0] * 2
    num_heads = dim // num_heads_dim

    if metadata.get("t_injection"):
        t_injection = metadata["t_injection"]
    elif any(k.endswith(".adaLN_mod.1.weight") for k in state):
        t_injection = "adaln_zero"
    elif "t_film.weight" in state:
        t_injection = "film"
    else:
        t_injection = "concat"

    model = LatCH(
        in_channels=in_channels,
        out_channels=out_channels,
        dim=dim,
        depth=depth,
        num_heads=num_heads,
        t_injection=t_injection,
    )
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)
    model.metadata = metadata
    return model
