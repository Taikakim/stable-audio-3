#!/bin/bash
# spectral_skewness EMA experiments at the best settings (adamw, lr 3e-4, bs 32).
# EMA on all; vary duration (2x/4x) + effective batch (grad-accum 2). Resumable.
cd /home/kim/Projects/SAO/stable-audio-3 || exit 1
export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MIOPEN_FIND_MODE=2 PYTORCH_TUNABLEOP_ENABLED=0
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
SW=/home/kim/Projects/SAO/stable-audio-3/latch_weights_ema_sweep
mkdir -p "$SW"

run() {  # epochs grad_accum tag
  local ep=$1 ga=$2 tag=$3 out="$SW/$3"
  if [ -f "$out/latch_sa3_spectral_skewness_best.pt" ]; then echo "[skip] $tag"; return; fi
  echo "=== TRAIN $tag (ema0.999 ep=$ep ga=$ga) start $(date +%H:%M:%S) ==="
  ./.venv/bin/python scripts/latch/train_latch.py \
    --feature spectral_skewness --optimizer adamw --lr 3e-4 --batch-size 32 \
    --ema 0.999 --grad-accum "$ga" --epochs "$ep" \
    --target-source npz --latent-dir /home/kim/Projects/latents_sa3 \
    --standardize --precision bf16 --t-injection adaln_zero --dim 256 --depth 4 --num-heads 8 \
    --seed 1 --save-best-only --wandb --wandb-project sa3-latch --run-name "skew_$tag" --save-dir "$out"
  echo "=== DONE $tag end $(date +%H:%M:%S) rc=$? ==="
}

run 20 1 ema20       # EMA at base duration
run 20 2 ema_ga2     # EMA + 2x effective batch (grad-accum 2)
run 40 1 ema40       # EMA + 2x training time
run 80 1 ema80       # EMA + 4x training time
echo "EMA_SWEEP_DONE $(date +%H:%M:%S)"
