#!/bin/bash
set -euo pipefail

SAO_DIR="/home/kim/Projects/SAO"
STABLE_AUDIO_DIR="$SAO_DIR/stable-audio-3"
PYTHON="$SAO_DIR/.venv-rocm10-amd/bin/python3"
DATA_DIR="/run/media/kim/Mantu/drum_sep_data_g3"
BASE_CKPT="/run/media/kim/Mantu/drum_sep_g3_multistem/epoch=1-step=6000.ckpt"

SAVE_DIR_ARM1="/run/media/kim/Mantu/drum_sep_scratch_r64_normuon_sf"
SAVE_DIR_ARM0="/run/media/kim/Mantu/drum_sep_scratch_r64_adamw"
SAVE_DIR_ARM2="/run/media/kim/Mantu/drum_sep_scratch_r64_muon_baseline"
SAVE_DIR_ARM3="/run/media/kim/Mantu/drum_sep_scratch_r128_normuon_sf"

EVAL_ROOT="/home/kim/sao_runs/drum_sep_scratch_ranks_eval"
mkdir -p "$SAVE_DIR_ARM1" "$SAVE_DIR_ARM0" "$SAVE_DIR_ARM2" "$SAVE_DIR_ARM3" "$EVAL_ROOT"

TARGET_ARM="${1:-all}"

echo "=== Acquiring GPU Guard for ANTIGRAVITY ==="
trap 'bash "$SAO_DIR/Misc/gpu_guard.sh" release ANTIGRAVITY' EXIT
bash "$SAO_DIR/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$

echo "================================================================================"
echo "STARTING FROM-SCRATCH DRUM SEPARATION SHOOTOUT"
echo "Target Arm: $TARGET_ARM"
echo "Data: $DATA_DIR/train (19,278 paired crops)"
echo "Steps per arm: 4000"
echo "Start Time: $(date)"
echo "================================================================================"

# Update dashboard initially
$PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/compile_ranks_shootout.py" || true

# -----------------------------------------------------------------------------
# Helper: Run evaluation for a finished arm
# -----------------------------------------------------------------------------
evaluate_arm() {
    local arm_id="$1"
    local ckpt_path="$2"
    local arm_eval_dir="$EVAL_ROOT/$arm_id"

    echo ""
    echo ">>> Running evaluation for $arm_id on $ckpt_path..."
    mkdir -p "$arm_eval_dir"

    CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/eval_separation.py" \
        --checkpoint "$ckpt_path" \
        --compare_checkpoint "$BASE_CKPT" \
        --val_dir "$DATA_DIR/val" \
        --steps 25 \
        --cfg 2.5 \
        --wiener_alpha 1.5 \
        --max_samples 16 \
        --out_dir "$arm_eval_dir"

    echo ">>> Updating master shootout comparison dashboard..."
    $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/compile_ranks_shootout.py" || true
    echo ">>> Console ready at: file://$arm_eval_dir/index.html"
}

# -----------------------------------------------------------------------------
# Arm 1: Rank 64 NorMuon + Warmup + Schedule-Free
# -----------------------------------------------------------------------------
run_arm1() {
    local ckpt="$SAVE_DIR_ARM1/epoch=0-step=4000.ckpt"
    echo ""
    echo "================================================================================"
    echo "=== ARM 1: Rank 64 NorMuon + Warmup + Schedule-Free (4000 steps) ==="
    echo "================================================================================"
    if [ -f "$ckpt" ]; then
        echo "Checkpoint $ckpt already exists. Skipping training."
    else
        CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/train_lora.py" \
            --devices 1 \
            --num_workers 0 \
            --model medium-base \
            --sep-data "$DATA_DIR/train" \
            --rank 64 \
            --batch_size 4 \
            --steps 4000 \
            --checkpoint_every 1000 \
            --no_demos \
            --log_every 50 \
            --optimizer fusion \
            --fusion-components "ns5,normuon,sf" \
            --spectral-lr 1e-3 \
            --scalar-lr 2e-4 \
            --warmup_steps 500 \
            --weight_decay 0.01 \
            --save_dir "$SAVE_DIR_ARM1" \
            --name drum_sep_r64_normuon_sf
    fi
    evaluate_arm "arm1_r64_normuon_sf" "$ckpt"
}

# -----------------------------------------------------------------------------
# Arm 0: Rank 64 AdamW Control Baseline (WSD Schedule)
# -----------------------------------------------------------------------------
run_arm0() {
    local ckpt="$SAVE_DIR_ARM0/epoch=0-step=4000.ckpt"
    echo ""
    echo "================================================================================"
    echo "=== ARM 0: Rank 64 AdamW Control Baseline (4000 steps) ==="
    echo "================================================================================"
    if [ -f "$ckpt" ]; then
        echo "Checkpoint $ckpt already exists. Skipping training."
    else
        CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/train_lora.py" \
            --devices 1 \
            --num_workers 0 \
            --model medium-base \
            --sep-data "$DATA_DIR/train" \
            --rank 64 \
            --batch_size 4 \
            --steps 4000 \
            --checkpoint_every 1000 \
            --no_demos \
            --log_every 50 \
            --optimizer adamw \
            --lr 2e-4 \
            --lr_schedule wsd \
            --warmup_steps 500 \
            --weight_decay 0.01 \
            --save_dir "$SAVE_DIR_ARM0" \
            --name drum_sep_r64_adamw
    fi
    evaluate_arm "arm0_r64_adamw" "$ckpt"
}

# -----------------------------------------------------------------------------
# Arm 2: Rank 64 Baseline Muon (Pure NS5 + WSD Decay)
# -----------------------------------------------------------------------------
run_arm2() {
    local ckpt="$SAVE_DIR_ARM2/epoch=0-step=4000.ckpt"
    echo ""
    echo "================================================================================"
    echo "=== ARM 2: Rank 64 Baseline Muon (Pure NS5 + WSD, 4000 steps) ==="
    echo "================================================================================"
    if [ -f "$ckpt" ]; then
        echo "Checkpoint $ckpt already exists. Skipping training."
    else
        CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/train_lora.py" \
            --devices 1 \
            --num_workers 0 \
            --model medium-base \
            --sep-data "$DATA_DIR/train" \
            --rank 64 \
            --batch_size 4 \
            --steps 4000 \
            --checkpoint_every 1000 \
            --no_demos \
            --log_every 50 \
            --optimizer fusion \
            --fusion-components "ns5" \
            --spectral-lr 3e-4 \
            --scalar-lr 2e-4 \
            --warmup_steps 500 \
            --fusion-decay wsd \
            --fusion-decay-start-frac 0.4 \
            --fusion-decay-min 0.1 \
            --weight_decay 0.01 \
            --save_dir "$SAVE_DIR_ARM2" \
            --name drum_sep_r64_muon_baseline
    fi
    evaluate_arm "arm2_r64_muon_baseline" "$ckpt"
}

# -----------------------------------------------------------------------------
# Arm 3: Rank 128 NorMuon + Warmup + Schedule-Free
# -----------------------------------------------------------------------------
run_arm3() {
    local ckpt="$SAVE_DIR_ARM3/epoch=0-step=4000.ckpt"
    echo ""
    echo "================================================================================"
    echo "=== ARM 3: Rank 128 NorMuon + Warmup + Schedule-Free (4000 steps) ==="
    echo "================================================================================"
    if [ -f "$ckpt" ]; then
        echo "Checkpoint $ckpt already exists. Skipping training."
    else
        CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/train_lora.py" \
            --devices 1 \
            --num_workers 0 \
            --model medium-base \
            --sep-data "$DATA_DIR/train" \
            --rank 128 \
            --batch_size 4 \
            --steps 4000 \
            --checkpoint_every 1000 \
            --no_demos \
            --log_every 50 \
            --optimizer fusion \
            --fusion-components "ns5,normuon,sf" \
            --force-scalar "lora_" \
            --spectral-lr 1e-3 \
            --scalar-lr 2e-4 \
            --warmup_steps 500 \
            --weight_decay 0.01 \
            --save_dir "$SAVE_DIR_ARM3" \
            --name drum_sep_r128_normuon_sf
    fi
    evaluate_arm "arm3_r128_normuon_sf" "$ckpt"
}

# -----------------------------------------------------------------------------
# Execution Dispatch
# -----------------------------------------------------------------------------
case "$TARGET_ARM" in
    all)
        run_arm1
        run_arm0
        run_arm2
        run_arm3
        ;;
    arm1)
        run_arm1
        ;;
    arm0)
        run_arm0
        ;;
    arm2)
        run_arm2
        ;;
    arm3)
        run_arm3
        ;;
    *)
        echo "Unknown target arm: $TARGET_ARM. Valid: all, arm1, arm0, arm2, arm3"
        exit 1
        ;;
esac

echo ""
echo "================================================================================"
echo "SHOOTOUT EXECUTION COMPLETE"
echo "Master Dashboard: file://$EVAL_ROOT/index.html"
echo "Web URL:          http://localhost:8088/drum_sep_scratch_ranks_eval/index.html"
echo "End Time:         $(date)"
echo "================================================================================"
