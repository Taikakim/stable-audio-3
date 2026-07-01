#!/usr/bin/env bash
# DoRA dim-128 finetune of SA3 medium-base on the 300-track subset (607 crops).
#
# Run after a GPU reset / reboot (a wedged HIP runtime from repeated GPU-process kills
# makes this hang at 100% GPU with no optimizer step — see WORKLOG 2026-06-01):
#     bash launch_dora.sh                      # full 380 s crops, ~9 h for 30 epochs
#     DORA_DURATION=100 bash launch_dora.sh    # ~100 s crops -> 30 epochs fits a session
#
# MINIMAL env on purpose — exactly the config that progressed. Do NOT add an `export`
# block, PYTHONUNBUFFERED, or MIOPEN_FIND_MODE (each coincided with a freeze).
# Progress signal = newest lightning_logs/version_*/metrics.csv (tqdm is silent in non-TTY).
# Checkpoints land every 5 epochs in the save_dir below. DO NOT kill/relaunch repeatedly.
set -u
cd /home/kim/Projects/SAO/stable-audio-3 || exit 1
PYTORCH_TUNABLEOP_ENABLED=0 .venv/bin/python scripts/train_lora.py \
  --model medium-base \
  --encoded_dir /run/media/kim/Lehto/latents_sa3_lora300 \
  --adapter_type dora-rows --rank 16 --lora_alpha 16 \
  --epochs 30 --batch_size 8 --accumulate_grad_batches 16 \
  --checkpoint_every_epochs 1 --gradient_clip_val 1.0 \
  --lr 1e-4 --base_precision bf16 --no_demos \
  --duration "${DORA_DURATION:-120}" \
  --num_workers 6 --seed 42 \
  --exclude seconds_total \
  --save_dir /run/media/kim/Lehto/sa3_lora_runs/dora128_300trk \
  --name dora128_300trk --logger "${DORA_LOGGER:-wandb}"
