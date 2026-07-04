"""Game of Life style weight mutation script for SA3 base model.

Applies drift, shuffle, blur, and contrast mutations to the DiT block weights 
with an exponential decay from the highest layers (block 23) to the lowest (block 0).
Bounds are strictly clamped to a +10% ecosystem based on the base model's variance.

Usage:
    python mutate_weights.py --out-dir /path/to/renders --drift 0.05 --shuffle 0.01 --blur 0.1
"""

import argparse
import os
import sys

SA3_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAO_ROOT = os.path.dirname(SA3_ROOT)
sys.path.append(SA3_ROOT)
sys.path.append(os.path.join(SAO_ROOT, "control"))

import torch
import torch.nn as nn
from stable_audio_3 import StableAudioModel
from sa3_control.audio_io import save_audio

def compute_ecosystem_bounds(sam, expand_pct=0.10):
    """
    Since merging all DoRAs is slow, we use the base model weights' variance
    to define the ecosystem bounds per layer.
    """
    bounds = {}
    for name, module in sam.model.named_modules():
        if isinstance(module, nn.Linear) and "blocks." in name:
            w = module.weight.detach()
            w_min, w_max = w.min().item(), w.max().item()
            rng = w_max - w_min
            if rng == 0:
                rng = 1e-5
            bounds[name] = (w_min - expand_pct * rng, w_max + expand_pct * rng)
    return bounds

def apply_mutations(sam, bounds, args):
    """
    Applies the specified mutations to the DiT blocks with exponential decay.
    """
    # 24 blocks in medium-base
    max_block = 23
    
    with torch.no_grad():
        for name, module in sam.model.named_modules():
            if isinstance(module, nn.Linear) and "blocks." in name:
                # Extract block index
                parts = name.split(".")
                idx = parts.index("blocks")
                block_idx = int(parts[idx + 1])
                
                # Calculate decay multiplier: 1.0 at max_block, approaches 0 at block 0
                decay = torch.exp(torch.tensor(args.decay_rate * (block_idx - max_block))).item()
                
                # Apply mutations if decay is significant enough
                if decay < 1e-4:
                    continue
                    
                b_min, b_max = bounds[name]
                w = module.weight
                w_mut = w.clone()
                
                # 1. Drift
                if args.drift > 0:
                    sigma = (b_max - b_min) * args.drift * decay
                    w_mut += torch.randn_like(w_mut) * sigma
                    
                # 2. Shuffle
                if args.shuffle > 0 and decay > 0.05:
                    flat = w_mut.view(-1)
                    n = flat.size(0)
                    num_swaps = int(n * args.shuffle * decay)
                    if num_swaps > 0:
                        idx_swap = torch.randint(0, n - 1, (num_swaps,), device=w.device)
                        temp = flat[idx_swap].clone()
                        flat[idx_swap] = flat[idx_swap + 1]
                        flat[idx_swap + 1] = temp
                        
                # 3. Blur / Pixelate (average adjacent)
                if args.blur > 0 and decay > 0.05:
                    # 1D blur along the inner dimension
                    # w_mut is (out_features, in_features)
                    blur_factor = args.blur * decay
                    blurred = (w_mut[:, :-1] + w_mut[:, 1:]) / 2.0
                    w_mut[:, :-1] = w_mut[:, :-1] * (1 - blur_factor) + blurred * blur_factor
                    
                # 4. Contrast
                if args.contrast > 0:
                    mean = w_mut.mean()
                    factor = 1.0 + (args.contrast * decay)
                    w_mut = mean + (w_mut - mean) * factor
                    
                # Clamp strictly to ecosystem bounds
                w_mut = torch.clamp(w_mut, b_min, b_max)
                w.copy_(w_mut)


def main():
    ap = argparse.ArgumentParser(description="Mutate SA3 weights using biological/image processing principles.")
    ap.add_argument("--out-dir", type=str, required=True, help="Directory to save generated audio")
    
    # Mutations
    ap.add_argument("--drift", type=float, default=0.0, help="Magnitude of gaussian noise added (e.g. 0.01)")
    ap.add_argument("--shuffle", type=float, default=0.0, help="Fraction of weights to swap with neighbors (e.g. 0.05)")
    ap.add_argument("--blur", type=float, default=0.0, help="Magnitude of adjacent weight averaging (e.g. 0.2)")
    ap.add_argument("--contrast", type=float, default=0.0, help="Multiplier for stretching weights from their mean (e.g. 0.1)")
    
    # Decay
    ap.add_argument("--decay-rate", type=float, default=0.5, help="Rate of exponential decay from block 23 down to 0")
    
    # Generation
    ap.add_argument("--prompt", type=str, default="psychedelic experimental glitch texture", help="Prompt to render")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-ckpt", action="store_true", help="Save the mutated model weights to a .pt file")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[init] Loading base SA3 model...", flush=True)
    sam = StableAudioModel.from_pretrained("medium-base", device=device)
    sr = sam.model.sample_rate

    print("[mutate] Calculating ecosystem bounds...", flush=True)
    bounds = compute_ecosystem_bounds(sam, expand_pct=0.10)

    print(f"[mutate] Applying mutations with decay={args.decay_rate}...", flush=True)
    apply_mutations(sam, bounds, args)
    
    fn_base = f"mutated_drift{args.drift}_shuf{args.shuffle}_blur{args.blur}_cont{args.contrast}"

    if args.save_ckpt:
        ckpt_path = os.path.join(args.out_dir, f"{fn_base}.pt")
        print(f"[mutate] Saving mutated model weights to {ckpt_path}...", flush=True)
        torch.save(sam.model.state_dict(), ckpt_path)

    print("[run] Generating mutated audio clip...", flush=True)
    with torch.inference_mode():
        audio = sam.generate(prompt=args.prompt, duration=args.duration, steps=50,
                             cfg_scale=args.cfg, seed=args.seed, sampler_type="euler")

    save_path = os.path.join(args.out_dir, f"{fn_base}.wav")
    save_audio(save_path, audio[0], sr)
    print(f"[done] Saved mutated texture to {save_path}", flush=True)


if __name__ == "__main__":
    main()
