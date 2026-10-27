#!/bin/bash
set -euo pipefail

SAO_DIR="/home/kim/Projects/SAO"
STABLE_AUDIO_DIR="$SAO_DIR/stable-audio-3"
PYTHON="$SAO_DIR/.venv-rocm10-amd/bin/python3"
DATA_DIR="/run/media/kim/Mantu/drum_sep_data_g3"
BASE_CKPT="/run/media/kim/Mantu/drum_sep_g3_multistem/epoch=1-step=6000.ckpt"

SAVE_DIR_LOW="/run/media/kim/Mantu/drum_sep_g3_lr3e5_anneal"
SAVE_DIR_HIGH="/run/media/kim/Mantu/drum_sep_g3_lr2e4_boost"
EVAL_DIR="/home/kim/sao_runs/drum_sep_lr_brackets_eval"

mkdir -p "$SAVE_DIR_LOW" "$SAVE_DIR_HIGH" "$EVAL_DIR"

echo "=== [1/4] Acquiring GPU Guard for ANTIGRAVITY ==="
trap 'bash "$SAO_DIR/Misc/gpu_guard.sh" release ANTIGRAVITY' EXIT
bash "$SAO_DIR/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$

echo "================================================================================"
echo "STARTING OVERNIGHT GATE G3 DRUM SEPARATION LR BRACKETS"
echo "Base Checkpoint: $BASE_CKPT"
echo "Start Time: $(date)"
echo "================================================================================"

# -----------------------------------------------------------------------------
# 1. Bracket 1: Low-LR Annealing (lr = 3e-5, 3000 steps)
# -----------------------------------------------------------------------------
echo ""
echo "=== [2/4] Running Bracket 1: Low-LR Annealing (lr=3e-5, 3000 steps) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/train_lora.py" \
    --devices 1 \
    --num_workers 0 \
    --model medium-base \
    --sep-data "$DATA_DIR/train" \
    --lora_checkpoint "$BASE_CKPT" \
    --rank 32 \
    --batch_size 4 \
    --steps 3000 \
    --checkpoint_every 1000 \
    --no_demos \
    --log_every 50 \
    --lr 3e-5 \
    --lr_schedule wsd \
    --warmup_steps 300 \
    --save_dir "$SAVE_DIR_LOW" \
    --name drum_sep_g3_lr3e5

echo "Bracket 1 Finished: $(date)"

# -----------------------------------------------------------------------------
# 2. Bracket 2: High-LR Exploration (lr = 2e-4, 3000 steps)
# -----------------------------------------------------------------------------
echo ""
echo "=== [3/4] Running Bracket 2: High-LR Boost (lr=2e-4, 3000 steps) ==="
echo "Start: $(date)"

CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/train_lora.py" \
    --devices 1 \
    --num_workers 0 \
    --model medium-base \
    --sep-data "$DATA_DIR/train" \
    --lora_checkpoint "$BASE_CKPT" \
    --rank 32 \
    --batch_size 4 \
    --steps 3000 \
    --checkpoint_every 1000 \
    --no_demos \
    --log_every 50 \
    --lr 2e-4 \
    --lr_schedule wsd \
    --warmup_steps 300 \
    --save_dir "$SAVE_DIR_HIGH" \
    --name drum_sep_g3_lr2e4

echo "Bracket 2 Finished: $(date)"

# -----------------------------------------------------------------------------
# 3. Bracket Evaluation Shootout & Audition Console
# -----------------------------------------------------------------------------
echo ""
echo "=== [4/4] Evaluating Checkpoints & Building Audition Console ==="
echo "Start: $(date)"

CKPT_LOW="$SAVE_DIR_LOW/epoch=0-step=3000.ckpt"
CKPT_HIGH="$SAVE_DIR_HIGH/epoch=0-step=3000.ckpt"

# Run evaluation on held-out tracks comparing baseline vs best new checkpoints
CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/eval_separation.py" \
    --checkpoint "$CKPT_LOW" \
    --compare_checkpoint "$BASE_CKPT" \
    --steps 25 \
    --cfg 2.5 \
    --wiener_alpha 1.5 \
    --val_samples 20 \
    --out_dir "$EVAL_DIR"

echo "================================================================================"
echo "OVERNIGHT DRUM SEPARATION RUNS COMPLETE"
echo "End Time: $(date)"
echo "Audition Console: file://$EVAL_DIR/index.html"
echo "================================================================================"
