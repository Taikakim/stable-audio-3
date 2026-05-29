# scripts/latch/latch_model.py
"""LatCH head for SA3. Canonical implementation lives in the package; re-exported
here so the training/verify scripts and their tests keep a stable import path."""

from stable_audio_3.models.latch import (
    LatCH,
    LatCHAttention,
    LatCHBlock,
    LatCHBlockAdaLN,
    RotaryEmbedding,
    TimestepEmbedder,
    apply_rotary_emb,
    load_latch_from_checkpoint,
)

__all__ = [
    "LatCH",
    "LatCHAttention",
    "LatCHBlock",
    "LatCHBlockAdaLN",
    "RotaryEmbedding",
    "TimestepEmbedder",
    "apply_rotary_emb",
    "load_latch_from_checkpoint",
]
