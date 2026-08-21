"""mir_control.py — MIR-timeseries conditioning for SA3 training (EXPERIMENTS B7,
Kim direct 2026-08-21: "train some really traditional models, using our mir data as
conditioners, probably rank 32 DoRAs").

WHAT THIS IS: classic MIR curves (band RMS, stem RMS/onset envelopes, beat/downbeat/onset
activations, HPCP chroma, spectral shape, f0) — already latent-rate-aligned (T=4096,
10.767 Hz) in each crop's .TIMESERIES.npz — fed into the DiT's NATIVE modular local
conditioning inlet (`modular_local_cond_configs`: per-id zero-init projection, applied
additively per frame; transformer.py:1024). A rank-32 DoRA on the backbone (installed by
train_lora's normal --adapter_type path) learns to USE the stream; the zero-init projection
makes step 0 a provable no-op. This executes the parked 2026-06-19 milestone
control/sa3_control/ATTRIBUTE_BRANCHES.md through the fleet-proven train_lora stack instead
of the frozen-base sa3_control adapter (that remains the Arm-B comparison, run locally).

PIECES (each unit-tested in tests/test_mir_control.py):
  build_ctrl_array / pack_channel_index  — .TIMESERIES.npz -> normalized (C,4096) fp16;
                                           packs are channel subsets of ONE superset array
  make_ctrl_metadata_wrapper             — custom_metadata_fn wrapper: slices the control
                                           to the EXACT latent crop window (PreEncodedDataset
                                           records latent_crop_start/length) + chains the
                                           caption fn
  MirCtrlConditioner                     — SAT Conditioner: list of (C,Tcrop) -> (B,C,T)
                                           + per-item CFG dropout (train mode only)
  install_mir_control                    — adds the zero-init modular projection to a loaded
                                           model + registers the cond id; returns new params
  ControlAblationCallback                — in-training "is the inlet used" meter: fixed val
                                           batch loss with true vs SHUFFLED vs ZERO control
  write_report                           — report.md + report.json per arm (runs on LUMI in
                                           the sbatch epilogue; eval-after-allocation still
                                           gets loss curves + ablation trajectory now)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Field registry — the 21-field goa∩avp intersection + goa-only f0 (zero-filled
# where absent, so one channel layout serves both corpora).
# ---------------------------------------------------------------------------
CTRL_FIELDS = [
    # dynamics (perceptual bands)
    "rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts", "rms_energy_air_ts",
    # stems (demucs-separated RMS + onset envelopes)
    "rms_bass_ts", "rms_drums_ts", "rms_other_ts", "rms_vocals_ts",
    "onset_envelope_bass_ts", "onset_envelope_drums_ts",
    "onset_envelope_other_ts", "onset_envelope_vocals_ts",
    # rhythm (madmom activations + global onset envelope)
    "beat_activation_ts", "downbeat_activation_ts", "onset_envelope_ts",
    # melody / harmony
    "hpcp_ts",                                        # 12 channels
    # spectral shape
    "spectral_flatness_ts", "spectral_flux_ts",
    "spectral_kurtosis_ts", "spectral_skewness_ts",
    # structure
    "relative_position_ts",
    # melody-height (goa-only today; zero-filled on avp) — mask with _voiced, never raw 0 Hz
    "f0_bass_ts", "f0_bass_voiced_ts", "f0_other_ts", "f0_other_voiced_ts",
]

_FIELD_WIDTH = {f: (12 if f == "hpcp_ts" else 1) for f in CTRL_FIELDS}
N_CTRL_CHANNELS = sum(_FIELD_WIDTH.values())          # 24 scalar + 12 hpcp = 36

PACKS = {
    "dynamics": ["rms_energy_bass_ts", "rms_energy_body_ts", "rms_energy_mid_ts", "rms_energy_air_ts"],
    "stems": ["rms_bass_ts", "rms_drums_ts", "rms_other_ts", "rms_vocals_ts",
              "onset_envelope_bass_ts", "onset_envelope_drums_ts",
              "onset_envelope_other_ts", "onset_envelope_vocals_ts"],
    "rhythm": ["beat_activation_ts", "downbeat_activation_ts", "onset_envelope_ts"],
    "melody": ["hpcp_ts"],
    "spectral": ["spectral_flatness_ts", "spectral_flux_ts",
                 "spectral_kurtosis_ts", "spectral_skewness_ts"],
    "f0": ["f0_bass_ts", "f0_bass_voiced_ts", "f0_other_ts", "f0_other_voiced_ts"],
    "structure": ["relative_position_ts"],
    "all": list(CTRL_FIELDS),
    # SPECIAL pack (2026-08-21, Kim's piano-roll lane): 128-ch MuScriptor note roll from the
    # SIBLING dir <latent_dir>_proll/<stem>.ctrl.npy (build_pianoroll_ctrl.py) — full-file
    # load, no channel indexing into the 36-ch superset.
    "pianoroll": None,
}


def _field_channel_offsets():
    off, out = 0, {}
    for f in CTRL_FIELDS:
        out[f] = (off, off + _FIELD_WIDTH[f])
        off += _FIELD_WIDTH[f]
    return out


_OFFSETS = _field_channel_offsets()


def pack_channel_index(pack: str):
    """Channel indices (into the superset ctrl array) for a named pack."""
    if pack == "pianoroll":
        return list(range(128))          # its own 128-ch sidecar, not the 36-ch superset
    idx = []
    for f in PACKS[pack]:
        lo, hi = _OFFSETS[f]
        idx.extend(range(lo, hi))
    return idx


# ---------------------------------------------------------------------------
# Control array construction + normalization
# ---------------------------------------------------------------------------
def build_ctrl_array(npz_path, stats, T: int = 4096) -> np.ndarray:
    """(N_CTRL_CHANNELS, T) fp16 from a .TIMESERIES.npz. Missing fields zero-fill
    (e.g. f0 on avp). stats = {field: {center, scale}} robust normalization; a
    degenerate scale (<=0) falls back to 1. NaN/inf -> 0 after normalization."""
    z = np.load(npz_path)
    out = np.zeros((N_CTRL_CHANNELS, T), dtype=np.float32)
    for f in CTRL_FIELDS:
        if f not in getattr(z, "files", []):
            continue
        a = np.asarray(z[f], dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        else:                                       # (T, 12) hpcp -> (12, T)
            a = a.T
        tlen = min(T, a.shape[1])
        st = (stats or {}).get(f, {"center": 0.0, "scale": 1.0})
        scale = float(st.get("scale", 1.0)) or 1.0
        if scale <= 0:
            scale = 1.0
        lo, hi = _OFFSETS[f]
        out[lo:lo + a.shape[0], :tlen] = (a[:, :tlen] - float(st.get("center", 0.0))) / scale
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float16)


# ---------------------------------------------------------------------------
# Dataset-side: metadata wrapper (slices control to the exact latent crop)
# ---------------------------------------------------------------------------
def make_ctrl_metadata_wrapper(base_fn, ctrl_source: str, pack: str, stats,
                               ctrl_dir: str | None = None):
    """Wrap a custom_metadata_fn so every item also carries info["mir_ctrl"] =
    (C_pack, crop_len) fp16, sliced with the SAME window PreEncodedDataset used
    (info["latent_crop_start"/"latent_crop_length"], recorded at crop time).

    ctrl_source: "ctrl"       -> pre-built superset .ctrl.npy in the SIBLING dir
                                 <latent_dir>_ctrl/ (LUMI path — zero mir deps).
                                 NEVER co-located with the latents: PreEncodedDataset
                                 recursively globs *.npy, so a co-located .ctrl.npy
                                 would be loaded as a latent (caught 2026-08-21).
                 "timeseries" -> <stem>.TIMESERIES.npz next to the latent (local path)
    ctrl_dir: for "ctrl", an explicit override directory holding <basename>.ctrl.npy.
    """
    idx = pack_channel_index(pack)

    def fn(info, audio):
        out = dict(base_fn(info, audio)) if base_fn is not None else {}
        stem = info["latent_filename"]
        stem = stem[: stem.rfind(".")]
        if ctrl_source == "ctrl":
            base = os.path.basename(stem) + ".ctrl.npy"
            suffix = "_proll" if pack == "pianoroll" else "_ctrl"
            d = ctrl_dir or (os.path.dirname(stem).rstrip("/") + suffix)
            full = np.load(os.path.join(d, base))
        else:
            full = build_ctrl_array(stem + ".TIMESERIES.npz", stats)
        s = int(info.get("latent_crop_start", 0))
        L = int(info.get("latent_crop_length", full.shape[1]))
        out["mir_ctrl"] = (full[:, s:s + L] if pack == "pianoroll" else full[idx][:, s:s + L])
        return out

    return fn


# ---------------------------------------------------------------------------
# Conditioner (SAT MultiConditioner slot "mir_ctrl")
# ---------------------------------------------------------------------------
class MirCtrlConditioner(nn.Module):
    """List of per-item (C, Tcrop) arrays -> ((B, C, T) float32, (B,) mask).
    Per-ITEM dropout (train mode only) = the control's CFG null: zeros through the
    zero-init projection are exactly the unconditioned model."""

    def __init__(self, n_channels: int, dropout_prob: float = 0.2):
        super().__init__()
        self.n_channels = int(n_channels)
        self.dropout_prob = float(dropout_prob)

    def forward(self, items, device):
        ts = [torch.as_tensor(np.asarray(x), dtype=torch.float32) for x in items]
        t = torch.stack(ts, 0).to(device)
        assert t.shape[1] == self.n_channels, (t.shape, self.n_channels)
        if self.training and self.dropout_prob > 0:
            keep = (torch.rand(t.shape[0], device=t.device) >= self.dropout_prob)
            t = t * keep[:, None, None]
        mask = torch.ones(t.shape[0], 1, dtype=torch.bool, device=t.device)
        return t, mask


# ---------------------------------------------------------------------------
# Installer — post-hoc modular projection on a LOADED model
# ---------------------------------------------------------------------------
def _make_proj(n_channels, dim, ref):
    """The exact module TransformerBlock builds for a modular config
    (transformer.py:1029): Linear->SiLU->Linear with zero-init output."""
    proj = nn.Sequential(nn.Linear(n_channels, dim), nn.SiLU(), nn.Linear(dim, dim))
    nn.init.zeros_(proj[-1].weight)
    nn.init.zeros_(proj[-1].bias)
    if ref is not None:
        proj = proj.to(device=ref.device, dtype=ref.dtype)
    return proj


def install_mir_control(diffusion_wrapper, transformer, n_channels: int,
                        cond_id: str = "mir_ctrl", dim: int | None = None,
                        blocks: str = "12-23"):
    """Register zero-init modular local projections on an already-loaded model and
    route the cond id through the diffusion wrapper. Returns the new trainable params.

    The modular inlet is PER TransformerBlock (each block owns a modular_local_embeds
    dict and adds its projection's output to its own input — transformer.py:1042).
    A transformer that carries blocks in .layers gets per-block installs on the
    `blocks` range ("lo-hi" inclusive, or "all") — default 12-23 = the union of W's
    layer-map localizations (rhythm 12-19, acoustic 16-23; 2026-07-10). A bare module
    exposing modular_local_embeds directly (tests) gets a single install."""
    dim = dim or transformer.dim
    ref = next(transformer.parameters(), None)
    params = []
    layers = getattr(transformer, "layers", None)
    if layers is not None and len(layers) > 0 and hasattr(layers[0], "modular_local_embeds"):
        n = len(layers)
        if blocks == "all":
            sel = range(n)
        else:
            lo, hi = (int(x) for x in blocks.split("-"))
            sel = range(max(0, lo), min(n - 1, hi) + 1)
        for i in sel:
            blk = layers[i]
            proj = _make_proj(n_channels, dim, ref)
            blk.modular_local_embeds[cond_id] = proj
            if getattr(blk, "modular_local_cond_configs", None) is None:
                blk.modular_local_cond_configs = []
            blk.modular_local_cond_configs.append({"id": cond_id, "dim": n_channels})
            params += list(proj.parameters())
        print(f"[mir_ctrl] installed on blocks {list(sel)[0]}..{list(sel)[-1]} of {n}")
    else:
        proj = _make_proj(n_channels, dim, ref)
        transformer.modular_local_embeds[cond_id] = proj
        if getattr(transformer, "modular_local_cond_configs", None) is None:
            transformer.modular_local_cond_configs = []
        transformer.modular_local_cond_configs.append({"id": cond_id, "dim": n_channels})
        params = list(proj.parameters())
    if cond_id not in diffusion_wrapper.modular_local_cond_ids:
        diffusion_wrapper.modular_local_cond_ids.append(cond_id)
    for p in params:
        p.requires_grad_(True)
    return params


# ---------------------------------------------------------------------------
# Control-ablation meter (the in-training "is the inlet used" report source)
# ---------------------------------------------------------------------------
class ControlAblationCallback:
    """Every N steps, measure a fixed val batch's loss under {true, shuffled, zero}
    control. control_gain = loss_shuffled - loss_true (>0 <=> the model exploits the
    ALIGNED control, not just its distribution). Records to control_ablation.jsonl.

    Framework-agnostic core (measure/append_record) so it is unit-testable without
    Lightning; the Lightning adapter in train_lora calls exactly these."""

    def __init__(self, out_dir: str, every_n_steps: int = 500):
        self.out_dir = out_dir
        self.every_n_steps = int(every_n_steps)

    def measure(self, loss_fn, ctrl: torch.Tensor):
        perm = torch.randperm(ctrl.shape[0])
        if ctrl.shape[0] > 1 and bool((perm == torch.arange(ctrl.shape[0])).all()):
            perm = perm.roll(1)
        return {
            "loss_true": float(loss_fn(ctrl)),
            "loss_shuffled": float(loss_fn(ctrl[perm])),
            "loss_zero": float(loss_fn(torch.zeros_like(ctrl))),
        }

    def append_record(self, rec: dict, step: int):
        rec = {"step": int(step), **rec}
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "control_ablation.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec


# ---------------------------------------------------------------------------
# Report writer (runs in the sbatch epilogue; CPU, stdlib+numpy only)
# ---------------------------------------------------------------------------
def write_report(save_dir: str, arm_meta: dict | None = None) -> str:
    """report.json + report.md from control_ablation.jsonl + Lightning CSV logs.
    Robust to partial runs (deadline truncation): reports whatever exists."""
    save = Path(save_dir)
    abl_rows = []
    ablf = save / "control_ablation.jsonl"
    if ablf.exists():
        for line in ablf.read_text().splitlines():
            if line.strip():
                abl_rows.append(json.loads(line))
    loss_rows = []
    for csv in sorted(save.glob("**/metrics.csv")):
        header = None
        for line in csv.read_text().splitlines():
            cells = line.split(",")
            if header is None:
                header = cells
                continue
            row = dict(zip(header, cells))
            lt = row.get("train/loss") or row.get("train_loss") or ""
            if lt:
                try:
                    loss_rows.append({"step": int(float(row.get("step", 0))), "loss": float(lt)})
                except ValueError:
                    pass

    def _gain(r):
        return r["loss_shuffled"] - r["loss_true"]

    rep = {
        "arm": arm_meta or {},
        "n_loss_points": len(loss_rows),
        "loss": {"first": loss_rows[0] if loss_rows else None,
                 "last": loss_rows[-1] if loss_rows else None},
        "ablation": {
            "n_points": len(abl_rows),
            "trajectory": abl_rows,
            "final": ({**abl_rows[-1], "control_gain": _gain(abl_rows[-1]),
                       "zero_gain": abl_rows[-1]["loss_zero"] - abl_rows[-1]["loss_true"]}
                      if abl_rows else None),
        },
        "how_to_read": ("control_gain = loss(shuffled ctrl) - loss(true ctrl) on a fixed val "
                        "batch: >0 means the model exploits the time-ALIGNED control; ~0 means "
                        "the inlet is unused (B7 kill-criterion input). zero_gain compares "
                        "against the CFG-null (zeros)."),
    }
    (save / "report.json").write_text(json.dumps(rep, indent=1))

    md = ["# B7 conditioner arm report", ""]
    for k, v in (arm_meta or {}).items():
        md.append(f"- **{k}**: {v}")
    if loss_rows:
        md.append(f"- **train loss**: {loss_rows[0]['loss']:.4f} (step {loss_rows[0]['step']}) "
                  f"-> {loss_rows[-1]['loss']:.4f} (step {loss_rows[-1]['step']})")
    md += ["", "## Control ablation (is the inlet used?)", "",
           "| step | true | shuffled | zero | control_gain |", "|---|---|---|---|---|"]
    for r in abl_rows:
        md.append(f"| {r['step']} | {r['loss_true']:.4f} | {r['loss_shuffled']:.4f} "
                  f"| {r['loss_zero']:.4f} | {_gain(r):+.4f} |")
    if abl_rows:
        md += ["", f"**Final control_gain: {_gain(abl_rows[-1]):+.4f}** "
                   "(>0 = aligned control exploited; ~0 = inlet unused)."]
    else:
        md += ["", "_No ablation records (run truncated before first measurement)._"]
    out = save / "report.md"
    out.write_text("\n".join(md) + "\n")
    return str(out)
