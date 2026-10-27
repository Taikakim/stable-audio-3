#!/usr/bin/env bash
# publish_boards.sh — publish staged drum separation boards from ~/evals_aac/ to aavepyora.online
set -euo pipefail

CONF="$HOME/.config/aavepyora.conf"
if [ ! -f "$CONF" ]; then
    echo "Error: missing $CONF"
    exit 1
fi

KEY=$(grep -E '^KEY=' "$CONF" | cut -d= -f2-)
HOST=$(grep -E '^HOST=' "$CONF" | cut -d= -f2-)
DEST_DIR=$(grep -E '^DEST_DIR=' "$CONF" | cut -d= -f2-)
DEST="${DEST_DIR}/evals"

echo "=== Syncing Drum Separation Boards to https://aavepyora.online/files/evals/ ==="
rsync -avz --chmod=F644,D755 -e "ssh -i $KEY -o BatchMode=yes" \
    /home/kim/evals_aac/drum_sep_scratch_ranks.html \
    /home/kim/evals_aac/drum_sep_lr_brackets.html \
    /home/kim/evals_aac/index.html \
    "$HOST:$DEST/"

echo "=== Syncing audio assets & consoles (drum_sep_scratch_ranks) ==="
rsync -avz --chmod=F644,D755 -e "ssh -i $KEY -o BatchMode=yes" \
    /home/kim/evals_aac/drum_sep_scratch_ranks/ \
    "$HOST:$DEST/drum_sep_scratch_ranks/"

echo "=== Syncing audio assets & consoles (drum_sep_lr_brackets) ==="
rsync -avz --chmod=F644,D755 -e "ssh -i $KEY -o BatchMode=yes" \
    /home/kim/evals_aac/drum_sep_lr_brackets/ \
    "$HOST:$DEST/drum_sep_lr_brackets/"

echo "=== PUBLISH COMPLETE ==="
echo "Live URLs:"
echo "  - https://aavepyora.online/files/evals/drum_sep_scratch_ranks/index.html"
echo "  - https://aavepyora.online/files/evals/drum_sep_lr_brackets/index.html"
echo "  - https://aavepyora.online/files/evals/index.html"
