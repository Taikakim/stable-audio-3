#!/usr/bin/env bash
# Arch / EndeavourOS port of AMD's install_kdb_files_for_pytorch_wheels.sh
#
# MIOpen .kdb files are SQLite databases of pre-compiled GPU kernels — they're
# pure data with no Linux distro dependency, so we can grab AMD's Ubuntu .deb
# packages and extract them with ar + tar (no dpkg-deb needed).
#
# Usage:
#   ROCM_VERSION=7.0 GFX_ARCH="gfx1201;gfx1030" ./install_kdb_files_for_pytorch_wheels_arch.sh
#
# Env vars:
#   ROCM_VERSION    e.g. 7.2.2, 7.0, 6.4   (required)
#   GFX_ARCH        semicolon-separated, e.g. "gfx1201" or "gfx1100;gfx1030"
#                   default: probes for everything available in the repo
#   UBUNTU_RELEASE  default 24.04 — picks which Ubuntu build of the .deb to grab.
#                   contents are data, so this doesn't affect compatibility.
#   DEST            override install dest. default = active venv's
#                   <torch>/share/miopen/

set -euo pipefail

ROCM_VERSION="${ROCM_VERSION:?Set ROCM_VERSION (e.g. 7.2.2)}"
UBUNTU_RELEASE="${UBUNTU_RELEASE:-24.04}"
REPO_URL="https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/pool/main/m"

# --- probe repo first ---
echo "Probing ${REPO_URL}/ ..."
INDEX=$( { curl -fsSL "${REPO_URL}/" \
           | grep -oE 'miopen-hip-gfx[0-9a-z]+kdb/' \
           | sort -u | tr -d '/'; } || true )

if [[ -z "$INDEX" ]]; then
    echo
    echo "No kdb packages exist for ROCm ${ROCM_VERSION}."
    echo "(AMD stopped shipping precompiled kdbs starting at ROCm 7.1.)"
    echo "Nothing to install — MIOpen will JIT-compile and cache kernels per shape."
    exit 0
fi

echo
echo "kdb packages available at ROCm ${ROCM_VERSION}:"
echo "$INDEX" | sed 's/^/  /'
echo

# --- pick architectures ---
if [[ -z "${GFX_ARCH:-}" ]]; then
    echo "GFX_ARCH not set — defaulting to all available archs."
    GFX_ARCHS=()
    while IFS= read -r pkg; do
        # strip "miopen-hip-" prefix and "kdb" suffix → bare arch like gfx1030
        GFX_ARCHS+=("${pkg#miopen-hip-}")
        GFX_ARCHS[-1]="${GFX_ARCHS[-1]%kdb}"
    done <<< "$INDEX"
else
    IFS=';' read -ra GFX_ARCHS <<< "$GFX_ARCH"
fi
echo "Targeting archs: ${GFX_ARCHS[*]}"
echo

# --- resolve PyTorch share dir ---
if [[ -n "${DEST:-}" ]]; then
    DEST_DIR="$DEST"
else
    TORCH_DIR=$(python -c \
        "import torch, os; print(os.path.dirname(torch.__file__))" 2>/dev/null || true)
    if [[ -z "$TORCH_DIR" || ! -d "$TORCH_DIR" ]]; then
        echo "ERROR: couldn't import torch. Activate the venv first," >&2
        echo "       or set DEST=/some/path/share/miopen to override." >&2
        exit 1
    fi
    DEST_DIR="${TORCH_DIR}/share"
fi
echo "Install destination: ${DEST_DIR}/miopen/"
echo

# --- download + extract ---
EXTRACT_DIR=$(mktemp -d -t miopen-kdb-XXXXXX)
trap 'rm -rf "$EXTRACT_DIR"' EXIT
cd "$EXTRACT_DIR"

ANY_DOWNLOADED=0
for arch in "${GFX_ARCHS[@]}"; do
    # List files inside that package directory
    PKG_DIR="${REPO_URL}/miopen-hip-${arch}kdb/"
    FILES=$(curl -fsSL "$PKG_DIR" 2>/dev/null \
            | grep -oE "miopen-hip-${arch}kdb_[^\"]*${UBUNTU_RELEASE}[^\"]*\.deb" \
            | sort -u | head -1) || true

    if [[ -z "$FILES" ]]; then
        echo "WARN: no $arch .deb for Ubuntu $UBUNTU_RELEASE in $PKG_DIR — skipping"
        continue
    fi

    for f in $FILES; do
        echo "Downloading $f ..."
        curl -fsSL -O "${PKG_DIR}${f}"
        ANY_DOWNLOADED=1
    done
done

if (( ANY_DOWNLOADED == 0 )); then
    echo "ERROR: nothing downloaded." >&2
    exit 1
fi

echo
echo "Extracting (ar + tar) ..."
for deb in *.deb; do
    echo "  $deb"
    ar x "$deb"
    for d in data.tar.*; do
        tar -xf "$d"
        rm -f "$d"
    done
    rm -f control.tar.* debian-binary "$deb"
done

# --- install ---
SRC=$(echo opt/rocm-*/share/miopen)
if [[ ! -d "$SRC" ]]; then
    echo "ERROR: extracted tree lacks opt/rocm-*/share/miopen" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp -ra "$SRC" "$DEST_DIR/"

echo
echo "Done. Installed kdb files:"
ls "$DEST_DIR/miopen/db/" 2>/dev/null | grep -E '\.kdb$' || echo "  (none)"
