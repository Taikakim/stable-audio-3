#!/usr/bin/env bash
# stable-audio-3/install.sh — set up SA3 on the TheRock 7.14 / Python 3.13 stack
# with the CK flash-attn backend for RDNA4 (gfx1201).
#
# Standalone usage (no SAO/ meta-repo required):
#   git clone <this-repo> && cd stable-audio-3 && ./install.sh
#
# Options:
#   --venv PATH        venv path (default: .venv)
#   --no-flash-attn    skip the from-source flash-attn build (~70 min on a
#                      Ryzen 9 9900X). The SDPA fallback works for small models.
#   --jobs N           MAX_JOBS for the flash-attn build (default: 6 → 12 clang
#                      processes, fits a 6-core / 12-thread budget on a 9900X).
#   --gpu-arch ARCH    GPU target (default: gfx1201; use gfx1200 for RX 9060).
#   --help
#
# What this does, in order:
#   1.  Sanity-check Python 3.13, uv, system hipcc (for the FA build).
#   2.  Create .venv with python3.13.
#   3.  Install torch + ROCm + triton from TheRock's multi-arch nightly index,
#       with `--index-url` ONLY (NO --extra-index-url pypi — uv's resolver will
#       pick PyPI CUDA torch otherwise, dragging in nvidia-* deps).
#   4.  Install SA3's runtime deps from PyPI with --no-deps (so nothing pulls
#       PyPI triton over the ROCm one).
#   5.  Pin transformers + tokenizers compatibly (transformers 5.9 wants
#       tokenizers<=0.23.0).
#   6.  Install SA3 itself (editable).
#   7.  If --no-flash-attn is NOT passed: clone ROCm/flash-attention at the
#       rdna_fmha_gfx1100_gfx1201 branch, apply the glue patches from
#       rocking/update_ck, and build the CK backend. See
#       SAO/docs/flash-attn-ck-rdna4.md for the long version.
#
# Critical runtime knob — set BEFORE `import flash_attn`:
#   export FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE
# Otherwise the wrapper auto-routes to aiter on HIP (and aiter isn't installed
# in this venv).

set -euo pipefail

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
VENV=".venv"
BUILD_FA=1
MAX_JOBS=6
GPU_ARCH="gfx1201"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    --no-flash-attn) BUILD_FA=0; shift ;;
    --jobs) MAX_JOBS="$2"; shift 2 ;;
    --gpu-arch) GPU_ARCH="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

REPO="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO"

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
need() {
  command -v "$1" &>/dev/null || { echo "ERROR: missing required tool: $1" >&2; exit 1; }
}
need git
need uv
if [[ "$BUILD_FA" == "1" ]]; then
  command -v hipcc &>/dev/null || {
    echo "ERROR: hipcc not found in PATH. Install rocm-hip-sdk (Arch: pacman -S rocm-hip-sdk)." >&2
    echo "       Or skip the FA build with --no-flash-attn." >&2
    exit 1
  }
  command -v gh &>/dev/null || {
    echo "ERROR: gh (GitHub CLI) not found — needed to fetch the rocking/update_ck patch files." >&2
    echo "       Install (Arch: pacman -S github-cli) or skip with --no-flash-attn." >&2
    exit 1
  }
fi

echo "==> SA3 install — TheRock 7.14 / Python 3.13 / RDNA4 ($GPU_ARCH)"
echo "    Repo:        $REPO"
echo "    Venv:        $REPO/$VENV"
echo "    Build FA:    $BUILD_FA  (MAX_JOBS=$MAX_JOBS)"
echo

# ---------------------------------------------------------------------------
# 1. Venv (Python 3.13 — required for the TheRock cp313 wheels)
# ---------------------------------------------------------------------------
if [[ ! -d "$VENV" ]]; then
  echo "==> Creating venv: $VENV (python 3.13)"
  uv venv --python 3.13 "$VENV"
else
  echo "==> Venv exists: $VENV"
fi
PY="$REPO/$VENV/bin/python"

# ---------------------------------------------------------------------------
# 2. TheRock torch + ROCm + triton (with the device-gfx extra)
# ---------------------------------------------------------------------------
echo "==> Installing TheRock torch/triton stack for $GPU_ARCH"
uv pip install --python "$PY" \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "torch[device-$GPU_ARCH]" "torchvision[device-$GPU_ARCH]" torchaudio

# ---------------------------------------------------------------------------
# 3. Runtime deps from PyPI (all with --no-deps to protect the ROCm stack).
#    transformers needs tokenizers<=0.23.0 (5.9 constraint).
# ---------------------------------------------------------------------------
echo "==> Installing runtime Python deps from PyPI (--no-deps protects torch/triton)"
uv pip install --python "$PY" --index-url https://pypi.org/simple/ --no-deps \
  tqdm einops einops-exts safetensors \
  "tokenizers>=0.22.0,<=0.23.0" transformers \
  huggingface-hub httpx httpcore h11 anyio sniffio \
  soundfile librosa scipy audioread joblib decorator lazy-loader msgpack pooch \
  platformdirs scikit-learn soxr standard-aifc standard-chunk standard-sunau threadpoolctl \
  llvmlite numba cffi \
  requests certifi charset-normalizer urllib3 idna \
  pillow matplotlib pandas accelerate pyyaml psutil \
  ninja pybind11 wheel setuptools pip

# ---------------------------------------------------------------------------
# 4. SA3 itself (editable so local edits are live)
# ---------------------------------------------------------------------------
echo "==> Installing SA3 (editable)"
uv pip install --python "$PY" --no-deps -e .

# ---------------------------------------------------------------------------
# 5. Flash Attention 2 — CK backend for RDNA4
# ---------------------------------------------------------------------------
if [[ "$BUILD_FA" == "1" ]]; then
  FA_DIR="$REPO/build-flash-attn"
  echo "==> Building flash-attn (CK) for $GPU_ARCH at $FA_DIR (~70 min on Ryzen 9 9900X at MAX_JOBS=$MAX_JOBS)"

  if [[ ! -d "$FA_DIR/.git" ]]; then
    git clone --depth 1 --branch rdna_fmha_gfx1100_gfx1201 \
      https://github.com/ROCm/flash-attention.git "$FA_DIR"
  fi
  cd "$FA_DIR"

  echo "    Initializing submodules (CK + cutlass + aiter; ~1 GB)…"
  git submodule update --init --recursive --depth 1

  # Upstream skew workaround: the rdna branch's csrc/flash_attn_ck/ glue is
  # older than its CK submodule pin. Pull three fixed files from sibling branch
  # rocking/update_ck (commit d81a98630 — adds sink_ptr/d_sink_ptr to
  # fmha_bwd_args and the is_gfx1x_arch helpers). Skip if already patched.
  echo "    Patching glue files from rocking/update_ck (if needed)…"
  needs_patch=0
  grep -q "sink_ptr" csrc/flash_attn_ck/mha_bwd.cpp || needs_patch=1
  grep -q "is_gfx1x_arch" csrc/flash_attn_ck/flash_common.hpp || needs_patch=1
  if [[ "$needs_patch" == "1" ]]; then
    for f in mha_bwd.cpp mha_varlen_bwd.cpp flash_common.hpp; do
      cp "csrc/flash_attn_ck/$f" "csrc/flash_attn_ck/$f.orig" 2>/dev/null || true
      gh api "repos/ROCm/flash-attention/contents/csrc/flash_attn_ck/$f?ref=rocking/update_ck" \
        --jq '.content' | base64 -d > "csrc/flash_attn_ck/$f"
    done
    # Regenerate flash_common_hip.hpp (the hipify pass at build start should do
    # this but doing it now avoids edge cases on incremental rebuilds).
    sed -E 's|flash_common\.hpp|flash_common_hip.hpp|g;
            s|mask\.hpp|mask_hip.hpp|g;
            s|fmha_fwd\.hpp|fmha_fwd_hip.hpp|g;
            s|fmha_bwd\.hpp|fmha_bwd_hip.hpp|g' \
        csrc/flash_attn_ck/flash_common.hpp \
      | (printf '// !!! This is a file automatically generated by hipify!!!\n#include "hip/hip_runtime.h"\n'; cat) \
      > csrc/flash_attn_ck/flash_common_hip.hpp
    echo "      patched: mha_bwd.cpp, mha_varlen_bwd.cpp, flash_common.hpp, flash_common_hip.hpp"
  else
    echo "      already patched (sink_ptr + is_gfx1x_arch present)"
  fi

  # Quick sanity that the symbols are present (catches the case where the gh
  # api call returned an empty body — has happened).
  grep -q "sink_ptr" csrc/flash_attn_ck/mha_bwd.cpp \
    || { echo "ERROR: patch failed: mha_bwd.cpp still missing sink_ptr" >&2; exit 1; }
  grep -q "is_gfx1x_arch" csrc/flash_attn_ck/flash_common.hpp \
    || { echo "ERROR: patch failed: flash_common.hpp still missing is_gfx1x_arch" >&2; exit 1; }

  echo "    Building (~70 min at MAX_JOBS=$MAX_JOBS; ninja is incremental — re-running picks up where it left off if interrupted)"
  GPU_ARCHS="$GPU_ARCH" FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE MAX_JOBS="$MAX_JOBS" \
    uv pip install --python "$PY" --no-build-isolation --no-deps -v .

  cd "$REPO"
else
  echo "==> Skipping flash-attn build (--no-flash-attn). SDPA fallback will be used."
fi

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
echo "==> Verifying install"
FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE "$PY" - <<'PY'
import torch
print(f"torch        {torch.__version__}")
print(f"HIP          {torch.version.hip}")
print(f"GPU          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
import stable_audio_3
print(f"SA3          {stable_audio_3.__file__}")
try:
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    print(f"flash_attn   {flash_attn.__version__}  (CK backend, fwd+varlen present)")
except ImportError as e:
    print(f"flash_attn   NOT INSTALLED ({e}) — SA3 will use SDPA fallback")
PY

echo
echo "==> Done."
echo "    Activate:  source $VENV/bin/activate"
echo "    Run UI:    FLASH_ATTENTION_TRITON_AMD_ENABLE=FALSE python run_gradio.py --model small-music-base"
echo "    Notes:     SAO/docs/flash-attn-ck-rdna4.md (full CK build recipe + pitfalls table)"
