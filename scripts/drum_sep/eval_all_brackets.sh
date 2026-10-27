#!/bin/bash
set -euo pipefail

SAO_DIR="/home/kim/Projects/SAO"
STABLE_AUDIO_DIR="$SAO_DIR/stable-audio-3"
PYTHON="$SAO_DIR/.venv-rocm10-amd/bin/python3"
DATA_DIR="/run/media/kim/Mantu/drum_sep_data_g3"
BASE_CKPT="/run/media/kim/Mantu/drum_sep_g3_multistem/epoch=1-step=6000.ckpt"
CKPT_LOW="/run/media/kim/Mantu/drum_sep_g3_lr3e5_anneal/epoch=0-step=3000.ckpt"
CKPT_HIGH="/run/media/kim/Mantu/drum_sep_g3_lr2e4_boost/epoch=0-step=3000.ckpt"

EVAL_BASE="/home/kim/sao_runs/drum_sep_lr_brackets_eval"
mkdir -p "$EVAL_BASE/baseline_step6000" "$EVAL_BASE/lr3e5_anneal" "$EVAL_BASE/lr2e4_boost"

echo "=== Acquiring GPU Guard for ANTIGRAVITY ==="
trap 'bash "$SAO_DIR/Misc/gpu_guard.sh" release ANTIGRAVITY' EXIT
bash "$SAO_DIR/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$

echo "================================================================================"
echo "STARTING FULL EVALUATION SHOOTOUT (3 CHECKPOINTS)"
echo "Start Time: $(date)"
echo "================================================================================"

# 1. Baseline (step 6000, lr 1e-4)
echo ""
echo "=== [1/3] Evaluating Baseline (step 6000) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/eval_separation.py" \
    --checkpoint "$BASE_CKPT" \
    --val_dir "$DATA_DIR/val" \
    --steps 25 \
    --cfg 2.5 \
    --wiener_alpha 1.5 \
    --max_samples 16 \
    --out_dir "$EVAL_BASE/baseline_step6000"

# 2. Low-LR Anneal (step 3000, lr 3e-5)
echo ""
echo "=== [2/3] Evaluating Low-LR Anneal (3e-5, step 3000) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/eval_separation.py" \
    --checkpoint "$CKPT_LOW" \
    --compare_checkpoint "$BASE_CKPT" \
    --val_dir "$DATA_DIR/val" \
    --steps 25 \
    --cfg 2.5 \
    --wiener_alpha 1.5 \
    --max_samples 16 \
    --out_dir "$EVAL_BASE/lr3e5_anneal"

# 3. High-LR Boost (step 3000, lr 2e-4)
echo ""
echo "=== [3/3] Evaluating High-LR Boost (2e-4, step 3000) ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/eval_separation.py" \
    --checkpoint "$CKPT_HIGH" \
    --compare_checkpoint "$BASE_CKPT" \
    --val_dir "$DATA_DIR/val" \
    --steps 25 \
    --cfg 2.5 \
    --wiener_alpha 1.5 \
    --max_samples 16 \
    --out_dir "$EVAL_BASE/lr2e4_boost"

echo "================================================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "End Time: $(date)"
echo "================================================================================"
