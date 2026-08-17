"""Per-source --caption_probs parsing (train_lora._parse_caption_probs).

WHY THIS IS TESTED AT ALL: a caption-tier mix that lands on the wrong corpus is INVISIBLE — no
error, no crash, just a model that learned the wrong prompts. The goa bigset's granite tier
measured NOT GROUNDED (rare-term recall 1.24x vs chance; generated from folder names) while
suomisoundi's is 16.2x, so the whole point of per-source probs is to keep a bad tier out of the mix
without holding the good corpora back. Silent recycling or a miscount here would defeat that.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from train_lora import _parse_caption_probs  # noqa: E402


def test_single_tuple_applies_to_every_source():
    # the historical behaviour — existing commands must keep working unchanged
    assert _parse_caption_probs("0.6,0.3,0.1", 3) == [(0.6, 0.3, 0.1)] * 3


def test_single_tuple_with_one_source():
    assert _parse_caption_probs("0,0,1", 1) == [(0.0, 0.0, 1.0)]


def test_per_source_tuples_map_in_order():
    # avp ; suomisoundi ; goa -- goa held to its MF tier because its granite is contaminated
    got = _parse_caption_probs("0,0.9,0.1;0,0.9,0.1;0,0,1", 3)
    assert got == [(0.0, 0.9, 0.1), (0.0, 0.9, 0.1), (0.0, 0.0, 1.0)]
    assert got[2] != got[0], "the goa source must NOT inherit the granite-led mix"


def test_whitespace_and_trailing_semicolon_tolerated():
    assert _parse_caption_probs(" 0,0.9,0.1 ; 0,0,1 ;", 2) == [(0.0, 0.9, 0.1), (0.0, 0.0, 1.0)]


def test_count_mismatch_raises_rather_than_recycling():
    # 2 tuples for 3 sources must NOT silently reuse the first
    with pytest.raises(ValueError, match="per-source tuples"):
        _parse_caption_probs("0,0.9,0.1;0,0,1", 3)


def test_too_many_tuples_raises():
    with pytest.raises(ValueError, match="per-source tuples"):
        _parse_caption_probs("0,0.9,0.1;0,0,1;0,0,1", 2)


def test_wrong_arity_raises():
    with pytest.raises(ValueError, match="3 values"):
        _parse_caption_probs("0.9,0.1", 1)


def test_all_zero_raises():
    # sum 0 would make the sampler's `rng.random() * sum` degenerate -- no tier ever selected
    with pytest.raises(ValueError, match="sums to 0"):
        _parse_caption_probs("0,0,0", 1)


def test_negative_raises():
    with pytest.raises(ValueError, match="negative"):
        _parse_caption_probs("0,-0.5,1", 1)


def test_unnormalised_weights_are_preserved_not_rescaled():
    # make_caption_sampler normalises internally (rng.random() * (p1+p2+p3)), so passing 9:1
    # instead of 0.9:0.1 must mean the same thing rather than being rejected or altered here
    assert _parse_caption_probs("0,9,1", 1) == [(0.0, 9.0, 1.0)]
