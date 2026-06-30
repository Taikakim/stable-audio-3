# Apply the ROCm/AMD env profile from rocm_env.yaml BEFORE torch is imported
# (the model import below pulls torch). setdefault semantics: shell exports win.
from stable_audio_3.rocm_env import apply_profile as _apply_rocm_profile

_apply_rocm_profile("inference")

from stable_audio_3.model import AutoencoderModel as AutoencoderModel  # noqa: E402
from stable_audio_3.model import StableAudioModel as StableAudioModel  # noqa: E402
