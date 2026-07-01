
import os

# ---------------------------------------------------------------------
# Performance tuning (GEMMs & kernels)
# ---------------------------------------------------------------------
# TunableOps: hipBLASLt GEMM algo selection persisted to disk.
# TUNING=0 = read-only (use existing CSV).
# Flip to 1 for a warmup run to populate the CSV, then back to 0 for
# steady-state inference.
os.environ["PYTORCH_TUNABLEOP_ENABLED"]  = "1"
os.environ["PYTORCH_TUNABLEOP_TUNING"]   = "1"
os.environ["PYTORCH_TUNABLEOP_VERBOSE"]  = "1"
os.environ["PYTORCH_TUNABLEOP_FILENAME"] = "/home/kim/pytorch-tunings-7.2.3"

# Print which SDPA backend was selected per call (flash / mem-efficient / math).
os.environ["PYTORCH_SDP_KERNEL_VERBOSE"] = "1"

# Stop on first NaN in autograd. PyTorch 2.9+. Off for inference.
# os.environ["TORCH_NAN_CHECK"] = "1"

# ---------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------
# PYTORCH_HIP_ALLOC_CONF was renamed to PYTORCH_ALLOC_CONF in PyTorch 2.9.
# Don't set both. expandable_segments=True is unsupported on gfx1201 as
# of ROCm 7.2.3 — omitted here on purpose.
os.environ["PYTORCH_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"

# Release cached blocks when free memory falls below this threshold.
os.environ["PYTORCH_HIP_FREE_MEMORY_THRESHOLD_MB"] = "1024"

# Avoid a known AMD CPU/GPU desync in kernel-argument passing.
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

# ---------------------------------------------------------------------
# MIOpen kernel selection
# ---------------------------------------------------------------------
# MIOPEN_FIND_MODE values (ROCm 7.x):
#   1 NORMAL            - full benchmark, slowest start, best result
#   2 FAST              - FindDb hit, or immediate-mode fallback
#   3 HYBRID            - FindDb hit, or full find machinery (default if unset)
#   5 DYNAMIC_HYBRID    - like 3 but skip non-dynamic kernels on miss
#   6 TRUST_VERIFY      - verify FindDb timings; tune on miss (bounded)
#   7 TRUST_VERIFY_FULL - like 6, no tuning time limit
# 2 = fast start, ideal for short inference. Switch to 6 for long training.
os.environ["MIOPEN_FIND_MODE"] = "2"

# Persist MIOpen tunings on internal NVMe.
os.environ["MIOPEN_CUSTOM_CACHE_DIR"] = "/home/kim/pytorch-tunings-7.2.3/miopen/cache"
os.environ["MIOPEN_USER_DB_PATH"]    = "/home/kim/pytorch-tunings-7.2.3/miopen/db"

# ---------------------------------------------------------------------
# Flash Attention / Triton
# ---------------------------------------------------------------------
os.environ["FLASH_ATTENTION_TRITON_AMD_ENABLE"] = "TRUE"
os.environ["TRITON_CACHE_DIR"] = "/home/kim/pytorch-tunings/triton_cache"

# =====================================================================
# Everything below this line is safe to import — env is now frozen in.
# =====================================================================
import torchaudio
from stable_audio_3 import StableAudioModel

model = StableAudioModel.from_pretrained("medium")
audio = model.generate(
    prompt="Twisted, aggressive EBM techno, 132 BPM, pulsating bassline, hard kick drum",
    duration=400,
)
torchaudio.save("/tmp/sao3_medium_test.wav", audio.squeeze(0).cpu(), sample_rate=44100)
print("wrote /tmp/sao3_medium_test.wav")
