#!/bin/bash
# spectral_skewness LatCH head — LR/batch sweep (6 runs, sequential). Resumable.
cd /home/kim/Projects/SAO/stable-audio-3 || exit 1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE PYTORCH_TUNABLEOP_ENABLED=0 MIOPEN_FIND_MODE=2
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
SWEEP=/home/kim/Projects/SAO/stable-audio-3/latch_weights_sweep
mkdir -p "$SWEEP"

run() {
  local lr=$1 bs=$2 tag=$3
  local out="$SWEEP/skew_${tag}"
  if [ -f "$out/latch_sa3_spectral_skewness_best.pt" ]; then echo "[skip] $tag (exists)"; return; fi
  echo "=== TRAIN skew lr=$lr bs=$bs tag=$tag  start $(date +%H:%M:%S) ==="
  .venv/bin/python scripts/latch/train_latch.py \
    --feature spectral_skewness --target-source npz \
    --latent-dir /home/kim/Projects/latents_sa3 \
    --optimizer adamw --lr "$lr" --batch-size "$bs" \
    --loss smooth_l1 --standardize --precision bf16 \
    --t-injection adaln_zero --dim 256 --depth 4 --num-heads 8 \
    --epochs 20 --seed 1 --save-best-only --num-workers 8 \
    --run-name "skew_${tag}" --wandb --wandb-project sa3-latch --save-dir "$out"
  echo "=== DONE $tag  end $(date +%H:%M:%S)  rc=$? ==="
}

# LR sweep at original batch (32):  2x / 4x / 8x  (orig lr 3e-4)
run 6e-4   32 lr2x_bs32
run 1.2e-3 32 lr4x_bs32
run 2.4e-3 32 lr8x_bs32
# batch sweep at 2x LR (6e-4):  0.5x / 0.25x / single
run 6e-4   16 lr2x_bs16
run 6e-4    8 lr2x_bs8
run 6e-4    1 lr2x_bs1
echo "SWEEP_TRAIN_DONE $(date +%H:%M:%S)"
