#!/usr/bin/env bash
# One-variable probe: LR 2e-5 (vs current 1e-4), everything else identical to the winning
# SF-NorMuon run.
S=/run/media/kim/Lehto/sa3_control_runs
A=/home/kim/Projects/SAO/stable-audio-tools/avp_sa3
cd "$A" || exit 1
D=$S/onset_FUSION_lr2e5_40epoch
mkdir -p "$D"
echo "=== onset head: SF-NorMuon, lr2e-5, crop1024 no-ckpt, full data $(date) ==="
STACK=714 CROP=512 CHECKPOINT=0 STEPS=216000 LR=2e-5 OPTIMIZER=fusion WARMUP=300 NO_PREENCODE=1 \
  CONTROL_MODE=scalar SCALAR_FIELD=onset_density \
  SAVE_DIR="$D" SAVE_EVERY=5400 RUN_NAME=onset_FUSION_lr2e5 bash launch_riffer.sh > "$D.log" 2>&1
echo "TRAIN DONE $(date) (last: $(grep -E '\[step ' "$D.log" | tail -1))"
echo "=== ELBOW MAP gains 2,3 (compare vs lr1e-4: held +0.90-0.95, widened to 6.3-9.2 @ step5000) $(date) ==="
for ((ck=5400; ck<=216000; ck+=5400)); do
  [ -f "$D/riffer_step${ck}.pt" ] || continue
  FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE PYTORCH_TUNABLEOP_ENABLED=1 \
    /home/kim/Projects/SAO/sa3-rocm7.13-test/.venv/bin/python sa3_control/onset_eval.py \
    "$D/riffer_step${ck}.pt" --densities "2,4,6,8,10,15,20" --gains "0.5,1,2,3,6,8,12" --out "$S/onset_eval_lr2e5_${ck}" >> "$D.log" 2>&1
done
echo "ALL DONE $(date)"
