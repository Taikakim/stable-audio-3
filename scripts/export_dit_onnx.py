#!/usr/bin/env python
"""export_dit_onnx.py — export the SA3 DiT (DiffusionTransformer) to ONNX at a
FIXED latent length, for low-VRAM AMD inference via ORT + MIGraphX.

Companion to export_same_onnx.py (the autoencoder). The DiT is the per-step hot
loop of generation; with text precached (T5-Gemma run offline) the ONNX graph is
just the DiT's CFG-free core `_forward` — conditioning assembly, CFG combine, and
the sampling loop all stay on the host (see dit_onnx_infer.py).

Why fixed length (and a *ladder* of lengths)
--------------------------------------------
Unlike the autoencoder, the DiT does full-sequence self-attention every step, so
it can't be chunked — the whole T must be one graph. The DiT has NO length-
dependent control flow (its rearranges are plain transpose + patchify), so a
dynamic-axis export would trace fine — but MIGraphX compiles per static shape and
this ORT build can't cache compiled programs, so a dynamic graph would recompile
(~min, 1.4B params) for every distinct T at runtime. Hence: export one fixed T,
build a LADDER (256/512/1024/2048/4096 = 23.8/47.5/95/190/380 s), compile each
once, and at runtime pad up to the smallest that fits.

Export target: `DiffusionTransformer._forward` (the CFG-free core).
Inputs (B=1): x[1,256,T], t[1], cross_attn_cond[1,seq,768],
cross_attn_cond_mask[1,seq], global_embed[1,768].
Output: velocity [1,256,T] (rectified-flow objective).

Two fixed dims: T (--frames, the ladder rung) and seq (--text-seq, the padded
text length your precache emits). Runs on CPU — no GPU needed for export/validate.

Usage
-----
    python scripts/export_dit_onnx.py --model medium-base --frames 256 --text-seq 128 --validate
    for L in 256 512 1024 2048 4096; do
      python scripts/export_dit_onnx.py --model medium-base --frames $L --text-seq 128
    done
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("SA3_DISABLE_FLASH_ATTN", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

import numpy as np
import torch
import torch.nn as nn

LATENT_DIM = 256


def force_exportable_attention() -> None:
    """Same fix as the autoencoder export: route attention off FlexAttention (a
    torch.compile HOP the ONNX exporter can't translate) onto plain masked SDPA.
    The DiT self-attn is global so it likely takes the plain-SDPA branch anyway,
    but disable Flex defensively (cross-attn / any masked path)."""
    import stable_audio_3.models.transformer as T
    T.flex_attention_available = False
    T.flex_attention_compiled = None


def load_dit_only(model_name: str, device: str):
    """Build ONLY the DiffusionTransformer from the cached config + weights —
    NO T5-Gemma / conditioners (text is precached, so they're not in the runtime
    path and t5gemma may not even be downloaded). Returns (DiffusionTransformer,
    diffusion_config_dict)."""
    import glob
    import json
    from safetensors.torch import load_file
    from stable_audio_3.models.diffusion import DiTWrapper

    snap = sorted(glob.glob(
        f"{os.path.expanduser('~')}/.cache/huggingface/hub/"
        f"models--stabilityai--stable-audio-3-{model_name}/snapshots/*"))
    if not snap:
        raise RuntimeError(f"no cached snapshot for {model_name}; check the model id")
    cfg_path = Path(snap[-1]) / "model_config.json"
    wts_path = Path(snap[-1]) / "model.safetensors"
    cfg = json.loads(cfg_path.read_text())
    dcfg = cfg["model"]["diffusion"]
    diff_cfg = dcfg["config"]
    obj = dcfg.get("diffusion_objective", "v")

    wrapper = DiTWrapper(diffusion_objective=obj, **diff_cfg)
    full = load_file(str(wts_path))
    # safetensors keys are 'model.model.<...>'; DiTWrapper's state_dict is
    # 'model.<...>' (DiTWrapper.model = DiffusionTransformer). Strip one 'model.'.
    state = {k[len("model."):]: v for k, v in full.items() if k.startswith("model.model.")}
    missing, unexpected = wrapper.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] {len(missing)} missing keys (e.g. {missing[:2]}) — check config match")
    wrapper.eval().to(device)
    return wrapper.model, diff_cfg          # .model is the DiffusionTransformer


class DiTCore(nn.Module):
    """CFG-free DiT forward: (x, t, cross_attn_cond, cross_attn_cond_mask,
    global_embed, local_add_cond) -> velocity. CFG batching/combine + sampling
    stay on the host. local_add_cond [B,257,T] = cat(inpaint_mask, inpaint_masked
    _input); for text-to-audio (no inpaint) it's all-zeros, but the DiT projects it
    with a bias so it is NOT a no-op — must be fed (zeros), not omitted."""
    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, x, t, cross_attn_cond, cross_attn_cond_mask, global_embed, local_add_cond):
        return self.dit._forward(
            x, t,
            cross_attn_cond=cross_attn_cond,
            cross_attn_cond_mask=cross_attn_cond_mask,
            global_embed=global_embed,
            local_add_cond=local_add_cond,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medium-base")
    ap.add_argument("--frames", type=int, required=True,
                    help="fixed latent length T (ladder rung: 256/512/1024/2048/4096)")
    ap.add_argument("--text-seq", type=int, default=128,
                    help="fixed text sequence length (your precache pads to this)")
    ap.add_argument("--cond-dim", type=int, default=768, help="cross_attn cond_token_dim")
    ap.add_argument("--global-dim", type=int, default=768, help="global_cond_dim")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--batch", type=int, default=1,
                    help="export batch size; 2 = stack cond+uncond so CFG is one DiT call/step")
    ap.add_argument("--fp16", action="store_true",
                    help="also write an fp16 copy (halves the ~5.8GB/rung on disk; needs onnxconverter-common)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    force_exportable_attention()
    print(f"[load] DiffusionTransformer only (no T5-Gemma) for {args.model!r} on {args.device} ...")
    dit, diff_cfg = load_dit_only(args.model, args.device)
    dit = dit.eval().float().requires_grad_(False)
    force_exportable_attention()
    for m in dit.modules():
        if hasattr(m, "checkpointing"):
            m.checkpointing = False
    # Pull conditioning dims from the config so the dummy inputs match exactly.
    args.cond_dim = int(diff_cfg.get("cond_token_dim", args.cond_dim))
    args.global_dim = int(diff_cfg.get("global_cond_dim", args.global_dim))
    args.local_add_dim = int(diff_cfg.get("local_add_cond_dim", 0))

    T, seq, B = args.frames, args.text_seq, args.batch
    print(f"[info] export shapes (batch={B}): x[{B},{LATENT_DIM},{T}] t[{B}] "
          f"cross_attn[{B},{seq},{args.cond_dim}] mask[{B},{seq}] global[{B},{args.global_dim}] "
          f"local_add[{B},{args.local_add_dim},{T}]")

    wrap = DiTCore(dit).eval()
    dev = args.device
    dummy = (
        torch.randn(B, LATENT_DIM, T, device=dev),
        torch.rand(B, device=dev),                                   # t in [0,1]
        torch.randn(B, seq, args.cond_dim, device=dev),              # cross_attn_cond
        torch.ones(B, seq, dtype=torch.bool, device=dev),            # cross_attn_cond_mask
        torch.randn(B, args.global_dim, device=dev),                 # global_embed
        torch.randn(B, args.local_add_dim, T, device=dev),           # local_add_cond (zeros at runtime)
    )
    in_names = ["x", "t", "cross_attn_cond", "cross_attn_cond_mask", "global_embed", "local_add_cond"]

    print(f"[torch] reference forward at T={T} ...")
    t0 = time.time()
    with torch.no_grad():
        ref = wrap(*dummy)
    print(f"[torch] -> {tuple(ref.shape)} in {time.time() - t0:.1f}s")

    bsuf = f"_b{B}" if B != 1 else ""
    out = args.out or Path(f"dit_{args.model}_L{T}{bsuf}.onnx")
    print(f"[onnx] exporting opset {args.opset} -> {out}")
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(
            wrap, dummy, str(out),
            input_names=in_names, output_names=["v"],
            opset_version=args.opset, do_constant_folding=True,
            dynamic_axes=None,   # fully static: batch=1, fixed T, fixed seq
        )
    print(f"[onnx] exported in {time.time() - t0:.1f}s  ({out.stat().st_size / 1e6:.1f} MB)")

    if args.validate:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[validate] onnxruntime not installed — skipping")
        else:
            sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
            feeds = {n: dummy[i].cpu().numpy() for i, n in enumerate(in_names)}
            o = sess.run(None, feeds)[0].astype(np.float32)
            r = ref.cpu().numpy().astype(np.float32)
            rel = float(np.abs(r - o).max() / (np.abs(r).max() or 1.0))
            denom = (np.linalg.norm(r) * np.linalg.norm(o)) or 1.0
            cos = float((r.ravel() * o.ravel()).sum() / denom)
            print(f"[validate] max|Δ|={np.abs(r - o).max():.3e} (rel {rel:.2%})  cos={cos:.6f}")
            print("[validate] ok" if cos > 0.9999 and rel < 0.05 else
                  "[validate] WARNING: larger diff than expected")

    if args.fp16:
        try:
            from onnxconverter_common import float16
            import onnx
        except ImportError:
            print("[fp16] onnxconverter-common/onnx not installed — skipping "
                  "(uv pip install onnxconverter-common onnx)")
        else:
            fp16_out = out.with_name(out.stem + "_fp16.onnx")
            m = float16.convert_float_to_float16(onnx.load(str(out)), keep_io_types=True)
            onnx.save(m, str(fp16_out))
            sz = fp16_out.stat().st_size + Path(str(fp16_out) + ".data").stat().st_size \
                if Path(str(fp16_out) + ".data").exists() else fp16_out.stat().st_size
            print(f"[fp16] wrote {fp16_out} (~{sz / 1e9:.1f} GB; delete the fp32 to reclaim disk)")

    print("\n[next] one .onnx per ladder rung (256/512/1024/2048/4096); compile each once on\n"
          "MIGraphX (long-lived server at boot). Host loop = dit_onnx_infer.py:\n"
          "  per step: stack [cond;uncond] (batch=2) -> DiT-onnx -> v = v_unc + cfg*(v_cond-v_unc)\n"
          "  x += dt*v ; final z0 -> same_decoder ONNX -> wav.")


if __name__ == "__main__":
    main()
