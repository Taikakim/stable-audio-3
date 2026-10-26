#!/bin/bash
set -euo pipefail

SAO_DIR="/home/kim/Projects/SAO"
STABLE_AUDIO_DIR="$SAO_DIR/stable-audio-3"
PYTHON="$SAO_DIR/.venv-rocm10-amd/bin/python3"
DATA_DIR="/run/media/kim/Mantu/drum_sep_data_g3"
SAVE_DIR="/run/media/kim/Mantu/drum_sep_g3_multistem"

echo "=== [1/3] Acquiring GPU Guard for ANTIGRAVITY ==="
trap 'bash "$SAO_DIR/Misc/gpu_guard.sh" release ANTIGRAVITY' EXIT
bash "$SAO_DIR/Misc/gpu_guard.sh" acquire ANTIGRAVITY $$

mkdir -p "$DATA_DIR/val" "$DATA_DIR/train" "$SAVE_DIR"

echo "=== [2/3] Encoding Multi-Stem Dataset (Val & Train) ==="
# Encode Val Split (10 held-out tracks)
if [ ! -f "$DATA_DIR/val/manifest.json" ]; then
    echo "Encoding Val Split (10 tracks, 4 stems)..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/build_pairs.py" \
        --stems_dir "/run/media/kim/Mantu/avp-stems-original-classified" \
        --out_dir "$DATA_DIR" \
        --split val \
        --remixes \
        --stems "kick,snare,hihat,percussion"
else
    echo "Val split manifest exists. Skipping encoding."
fi

# Encode Train Split (30 tracks)
if [ ! -f "$DATA_DIR/train/manifest.json" ]; then
    echo "Encoding Train Split (30 tracks, 4 stems)..."
    CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/drum_sep/build_pairs.py" \
        --stems_dir "/run/media/kim/Mantu/avp-stems-original-classified" \
        --out_dir "$DATA_DIR" \
        --split train \
        --remixes \
        --stems "kick,snare,hihat,percussion"
else
    echo "Train split manifest exists. Skipping encoding."
fi

echo "=== [3/3] Launching Gate G3 Multi-Stem LoRA Training ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON "$STABLE_AUDIO_DIR/scripts/train_lora.py" \
    --devices 1 \
    --num_workers 0 \
    --model medium-base \
    --sep-data "$DATA_DIR/train" \
    --rank 32 \
    --batch_size 4 \
    --steps 6000 \
    --checkpoint_every 1000 \
    --no_demos \
    --log_every 50 \
    --lr 1e-4 \
    --lr_schedule wsd \
    --warmup_steps 500 \
    --save_dir "$SAVE_DIR" \
    --name drum_sep_g3_multistem

echo "=== Multi-Stem LoRA Training Complete! ==="
