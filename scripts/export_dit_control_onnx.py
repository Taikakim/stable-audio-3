#!/usr/bin/env python
"""export_dit_control_onnx.py — export the SA3 DiT with a trained **scalar control
adapter baked in** (e.g. onset-density steering), to ONNX for low-VRAM AMD inference.

This is the control sibling of export_dit_onnx.py. The control adapter (avp_sa3
`sa3_control`) is a decoupled cross-attention branch added to every DiT block — a
pure *forward* modification (no autograd/guidance), so it folds straight into the
ONNX graph. The control signal enters as two explicit graph inputs:

    control_tokens [1, n_tokens, control_dim]   # from the scalar conditioner (host)
    gain           [1]                           # control strength (1.0 = as trained)

At generation the host feeds, per CFG pass (exactly as sa3_control/onset_eval.py):
    cond pass   -> control_tokens = enc((target - mean)/std)   # the requested onset
    uncond pass -> control_tokens = zeros                      # the trained null
    v = v_uncond + cfg * (v_cond - v_uncond)
so the control rides the CFG axis. `gain` dials strength without re-exporting.

The scalar->tokens conditioner (a 2-layer FiLM over n_tokens learned tokens) is tiny;
its weights + the (mean,std) normalization are saved next to the .onnx as a .cond.npz
so the host can compute control_tokens in pure numpy (no torch at runtime).

Export + validate run on CPU. Validation is THREE-way to prove the control actually
landed in the graph: ONNX==torch(with control), and both differ from torch(no control).

    python scripts/export_dit_control_onnx.py \
        --ckpt /run/media/kim/Mantu/sa3_control_runs/onset_FUSION_lr2e5_40epoch/soup_exppeak.pt \
        --model medium-base --frames 256 --text-seq 128 --validate
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from export_dit_onnx import force_exportable_attention, load_dit_only, LATENT_DIM  # noqa: E402

# sa3_control lives in the avp_sa3 package (stable-audio-tools); its adapter/conditioner
# core only needs torch, so it imports fine in the SA3 venv with the path added.
AVP = "/home/kim/Projects/SAO/stable-audio-tools/avp_sa3"
if AVP not in sys.path:
    sys.path.insert(0, AVP)
from sa3_control.adapters import ControlledCrossAttention, _ACTIVE  # noqa: E402
from sa3_control.inject import find_cross_attn  # noqa: E402
from sa3_control.conditioner import ScalarAttributeEncoder  # noqa: E402
from dit_control_onnx_infer import add_fractional_positions_np  # noqa: E402


def install_adapters_on(dit, control_dim, position_encoding=True):
    """sa3_control.inject.install_adapters, but on a bare DiffusionTransformer
    (not a StableAudioModel). Same walk order, so adapter.{i} ↔ the i-th cross-attn."""
    targets = find_cross_attn(dit)
    if not targets:
        raise RuntimeError("no SA3 cross-attention modules found on the DiT")
    wrappers = []
    for parent, attr, base in targets:
        w = ControlledCrossAttention(base, control_dim, position_encoding)
        setattr(parent, attr, w)
        wrappers.append(w)
    return wrappers


class ControlledDiTCore(nn.Module):
    """DiT `_forward` with the control adapter active. control_tokens + gain are
    explicit forward args; we stash them in the adapter's module-global right before
    `_forward` so each ControlledCrossAttention reads them. The tracer follows them
    because they are forward inputs (the global is just a pass-through reference)."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, t, cross_attn_cond, cross_attn_cond_mask, global_embed,
                local_add_cond, control_tokens, gain):
        _ACTIVE["tokens"] = control_tokens
        _ACTIVE["gain"] = gain
        try:
            return self.dit._forward(
                x, t,
                cross_attn_cond=cross_attn_cond,
                cross_attn_cond_mask=cross_attn_cond_mask,
                global_embed=global_embed,
                local_add_cond=local_add_cond,
            )
        finally:
            _ACTIVE["tokens"] = None
            _ACTIVE["gain"] = 1.0


def _cos(a, b):
    r, o = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float((r * o).sum() / ((np.linalg.norm(r) * np.linalg.norm(o)) or 1.0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path, help="trained control adapter .pt")
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--frames", type=int, required=True, help="fixed latent length T")
    ap.add_argument("--text-seq", type=int, default=128)
    ap.add_argument("--cond-dim", type=int, default=768, help="cross_attn cond_token_dim")
    ap.add_argument("--global-dim", type=int, default=768, help="global_cond_dim")
    ap.add_argument("--gain", type=float, default=2.0, help="dummy/validate gain (runtime is an input)")
    ap.add_argument("--in-graph-pe", action="store_true",
                    help="keep the adapter's fractional-position PE inside the graph (default: move it "
                         "host-side so the fp16 export has no ConstantOfShape and converts cleanly)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--fp16", action="store_true",
                    help="also write a stamped fp16 copy (~3.1 GB; the host-PE graph converts cleanly)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    force_exportable_attention()
    dev = args.device
    print(f"[load] DiffusionTransformer ({args.model}) ...")
    dit, diff_cfg = load_dit_only(args.model, dev)
    dit = dit.float()

    print(f"[load] control adapter {args.ckpt.name} ...")
    ck = torch.load(str(args.ckpt), map_location=dev, weights_only=False)
    state = ck["state"]
    control_dim = int(ck["args"].get("control_dim", args.cond_dim))
    n_tokens = min(int(ck["args"].get("n_tokens", 16)), 16)
    field = ck.get("scalar_field")
    mean, std = ck.get("scalar_norm", [0.0, 1.0])
    n_blocks = len({int(k.split(".")[1]) for k in state if k.startswith("adapter.")})
    print(f"[ctrl] field={field}  norm(mean={mean:.3f},std={std:.3f})  "
          f"control_dim={control_dim}  n_tokens={n_tokens}  adapter_blocks={n_blocks}")

    wrappers = install_adapters_on(dit, control_dim)
    if len(wrappers) != n_blocks:
        print(f"[warn] wrapped {len(wrappers)} cross-attns but ckpt has {n_blocks} adapter blocks")
    for i, w in enumerate(wrappers):
        asd = {k[len(f"adapter.{i}."):]: v for k, v in state.items() if k.startswith(f"adapter.{i}.")}
        w.adapter.load_state_dict(asd)
        w.adapter.float()
    enc = ScalarAttributeEncoder(control_dim=control_dim, n_tokens=n_tokens)
    csd = {k[len("conditioner."):]: v for k, v in state.items() if k.startswith("conditioner.")}
    miss, unexp = enc.load_state_dict(csd, strict=False)
    if miss or unexp:
        print(f"[warn] conditioner load miss={miss} unexp={unexp}")
    enc.float().eval()
    dit.eval()

    local_add_dim = int(diff_cfg.get("local_add_cond_dim", 0))
    T, seq = args.frames, args.text_seq
    print(f"[info] export shapes: x[1,{LATENT_DIM},{T}] t[1] cross[1,{seq},{args.cond_dim}] "
          f"mask[1,{seq}] global[1,{args.global_dim}] local_add[1,{local_add_dim},{T}] "
          f"control_tokens[1,{n_tokens},{control_dim}] gain[1]")

    host_pe = not args.in_graph_pe                          # default: PE host-side (fp16-friendly)
    print(f"[pe] adapter positional encoding: {'HOST-side (fp16-friendly)' if host_pe else 'in-graph'}")

    g = torch.Generator().manual_seed(0)
    raw_ct = torch.randn(1, n_tokens, control_dim, generator=g)        # raw conditioner-style tokens
    base = [
        torch.randn(1, LATENT_DIM, T, generator=g),
        torch.rand(1, generator=g),
        torch.randn(1, seq, args.cond_dim, generator=g),
        torch.ones(1, seq, dtype=torch.bool),
        torch.randn(1, args.global_dim, generator=g),
        torch.randn(1, local_add_dim, T, generator=g),
    ]
    gain_t = torch.tensor([float(args.gain)])
    in_names = ["x", "t", "cross_attn_cond", "cross_attn_cond_mask", "global_embed",
                "local_add_cond", "control_tokens", "gain"]

    wrap = ControlledDiTCore(dit).eval()
    # GOLD reference = the TRAINED behavior: adapter applies its PE in-forward, raw tokens in.
    with torch.no_grad():
        ref_gold = wrap(*base, raw_ct, gain_t).cpu().numpy().astype(np.float32)

    if host_pe:
        for w in wrappers:
            w.adapter.position_encoding = False                       # PE now applied host-side
        ct_in = torch.from_numpy(add_fractional_positions_np(raw_ct.numpy()))
    else:
        ct_in = raw_ct
    dummy = tuple(base) + (ct_in, gain_t)                              # what the graph is traced with

    with torch.no_grad():
        ref = wrap(*dummy).cpu().numpy().astype(np.float32)           # the EXPORTED behavior
        off_ct = np.zeros((1, n_tokens, control_dim), np.float32)     # the uncond null...
        if host_pe:
            off_ct = add_fractional_positions_np(off_ct)              # ...+PE, exactly as in-adapter did
        d0 = list(dummy); d0[6] = torch.from_numpy(off_ct)
        ref_off = wrap(*d0).cpu().numpy().astype(np.float32)

    equiv = _cos(ref_gold, ref)
    print(f"[torch] host-PE refactor vs trained in-graph PE: cos={equiv:.6f}  "
          + ("✔ equivalent" if equiv > 0.99999 else "✗ MISMATCH — PE port differs"))
    eff_rel = float(np.linalg.norm(ref - ref_off) / (np.linalg.norm(ref_off) or 1.0))
    print(f"[torch] control effect: rel-L2={eff_rel:.2%}  max|Δ|={np.abs(ref - ref_off).max():.3e}  "
          + ("(control changes the velocity ✔)" if eff_rel > 0.002 else
             "(WARNING: control had ~no effect — gain/tokens?)"))

    out = args.out or Path(f"dit_{args.model}_L{T}_ctrl_{field}.onnx")
    print(f"[onnx] exporting opset {args.opset} -> {out}")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(wrap, dummy, str(out), input_names=in_names, output_names=["v"],
                          opset_version=args.opset, do_constant_folding=True, dynamic_axes=None)
    print(f"[onnx] exported in {time.time() - t0:.1f}s  ({out.stat().st_size / 1e6:.1f} MB)")

    # conditioner weights -> npz so the host computes control_tokens in numpy
    cond_npz = out.with_suffix(".cond.npz")
    np.savez(cond_npz,
             tokens=enc.tokens.detach().cpu().numpy().astype(np.float32),
             film0_w=enc.film[0].weight.detach().cpu().numpy().astype(np.float32),
             film0_b=enc.film[0].bias.detach().cpu().numpy().astype(np.float32),
             film2_w=enc.film[2].weight.detach().cpu().numpy().astype(np.float32),
             film2_b=enc.film[2].bias.detach().cpu().numpy().astype(np.float32),
             mean=np.float32(mean), std=np.float32(std),
             n_tokens=np.int64(n_tokens), control_dim=np.int64(control_dim),
             field=str(field), host_pe=np.bool_(host_pe))
    print(f"[cond] wrote conditioner -> {cond_npz}  (host_pe={host_pe})")

    # Stamp the onnx so the runner can detect a mismatched .cond.npz/.onnx pair (host_pe
    # applied twice or zero times = silent CFG corruption). load_external_data=False rewrites
    # only the small graph proto; the multi-GB .data weights are untouched.
    import onnx

    def _stamp(path):
        m = onnx.load(str(path), load_external_data=False)
        keys = {p.key for p in m.metadata_props}
        for k, v in [("host_pe", str(bool(host_pe))), ("control_field", str(field))]:
            if k not in keys:
                m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
        onnx.save(m, str(path))

    _stamp(out)
    print(f"[meta] stamped {out.name}: host_pe={host_pe}, control_field={field}")

    if args.fp16:
        from onnxconverter_common import float16
        fp16_out = out.with_name(out.stem + "_fp16.onnx")
        for p in (fp16_out, Path(str(fp16_out) + ".data")):
            if p.exists():
                p.unlink()
        mf = float16.convert_float_to_float16_model_path(str(out), keep_io_types=True)
        onnx.save_model(mf, str(fp16_out), save_as_external_data=True,
                        all_tensors_to_one_file=True, location=fp16_out.name + ".data", size_threshold=1024)
        _stamp(fp16_out)
        sz = fp16_out.stat().st_size + Path(str(fp16_out) + ".data").stat().st_size
        print(f"[fp16] wrote {fp16_out.name} (~{sz / 1e9:.1f} GB, stamped) — host-PE graph converts cleanly "
              f"(no ConstantOfShape). Low-VRAM: ~3.1 GB DiT + fp16 decoder.")

    if args.validate:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[validate] onnxruntime not installed — skipping")
            return
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        feeds = {n: dummy[i].cpu().numpy() for i, n in enumerate(in_names)}
        o = sess.run(None, feeds)[0].astype(np.float32)
        cos_on = _cos(ref, o)             # must be ~1.0 (graph reproduces controlled torch)
        cos_off = _cos(ref_off, o)        # must be < 1 (graph is NOT the plain DiT)
        rel = float(np.abs(ref - o).max() / (np.abs(ref).max() or 1.0))
        off_rel = float(np.linalg.norm(o - ref_off) / (np.linalg.norm(ref_off) or 1.0))
        print(f"[validate] ONNX vs torch(control ON):  cos={cos_on:.6f}  rel={rel:.2%}  (want ~1.0)")
        print(f"[validate] ONNX vs plain-DiT(ctrl OFF): rel-L2={off_rel:.2%}  cos={cos_off:.6f}  (want > 0)")
        # `equiv` (host-PE graph vs trained in-graph PE) is gated too: cos_on alone CANNOT catch a
        # PE-port bug — it compares ONNX to the host-PE torch ref, so a bad PE cancels out. equiv is
        # the only check that compares against the trained gold.
        ok = cos_on > 0.9999 and off_rel > 0.002 and eff_rel > 0.002 and equiv > 0.99999
        print(f"[validate] host-PE equivalence (vs trained gold): cos={equiv:.6f}  (want > 0.99999)")
        print("[validate] ✔ control correctly baked into the ONNX graph" if ok else
              "[validate] ✗ FAILED — control not in the graph / PE-port diverged")


if __name__ == "__main__":
    main()
