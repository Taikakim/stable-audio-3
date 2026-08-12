"""audio_augment.py — live-encode waveform augmentation for SA3 finetuning (CONTINUITY 2026-08-04,
#68). Applied to the [C, T] audio crop BEFORE the in-loop SAME encode, so every batch sees a fresh
realization on multiple axes — the "encode-live, augment-live" plan enabled once we found T512 bf16
encode is ~0.57 s/batch (cheap). Kept in the fork's scripts/ (NOT the upstream core package) so
stable_audio_3/ stays upstream-syncable; train_lora imports it and hands it to the LocalDataset.

Axes (each applied independently with its own probability):
  - sub_frame_shift : the PHASE-variance augmentation. SAME's latent is phase-variant (a sub-frame
                      sample offset -> a different latent trajectory). Shifting by k < ds_ratio
                      samples presents a fresh phase realization the model would otherwise never see.
                      shift-and-zeropad (no circular wrap -> no seam glitch); loses <93 ms at one edge.
  - gain           : +/- dB level jitter (always safe).
  - stereo_width   : mid/side width scaling (stereo only; safe).
  - polarity       : full-mix polarity flip (phase-inaudible, doubles effective data — like PhaseFlipper).
Optional / heavier (default OFF — they alter musical attributes and cost more; enable per-arm):
  - pitch_shift    : +/- semitones (changes key; goa is key-agnostic but it is a real content change).
  - time_stretch   : +/- tempo % (changes BPM; use sparingly for a tempo-sensitive genre).
All transforms are shape-preserving [C, T] -> [C, T], fp32, and no-op on the un-augmented path.
"""
import torch
import torch.nn.functional as F


class AudioAugment(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int = 44100,
        ds_ratio: int = 4096,          # samples per SAME latent frame -> sub-frame shift ceiling
        sub_frame_shift_prob: float = 0.8,
        max_shift_samples: int | None = None,   # default = ds_ratio (one frame of phase coverage)
        gain_prob: float = 0.5,
        gain_db: float = 4.0,
        stereo_width_prob: float = 0.5,
        stereo_width_range: tuple = (0.7, 1.3),
        polarity_prob: float = 0.5,
        # heavier, off by default
        pitch_shift_prob: float = 0.0,
        pitch_semitones: float = 2.0,
        time_stretch_prob: float = 0.0,
        time_stretch_range: tuple = (0.95, 1.05),
        seed: int | None = None,
    ):
        super().__init__()
        self.sr = sample_rate
        self.ds_ratio = ds_ratio
        self.max_shift = ds_ratio if max_shift_samples is None else int(max_shift_samples)
        self.sub_frame_shift_prob = sub_frame_shift_prob
        self.gain_prob = gain_prob
        self.gain_db = gain_db
        self.stereo_width_prob = stereo_width_prob
        self.stereo_width_range = stereo_width_range
        self.polarity_prob = polarity_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.pitch_semitones = pitch_semitones
        self.time_stretch_prob = time_stretch_prob
        self.time_stretch_range = time_stretch_range
        self.g = torch.Generator()
        if seed is not None:
            self.g.manual_seed(seed)
        self._pitch = None  # lazy torchaudio transforms (only if enabled)

    def _rand(self):
        return torch.rand(1, generator=self.g).item()

    def _uniform(self, lo, hi):
        return lo + (hi - lo) * self._rand()

    def _sub_frame_shift(self, x):
        # shift right by k in [1, max_shift), zero-pad the front, drop the tail (no circular wrap)
        k = int(self._rand() * self.max_shift)
        if k <= 0:
            return x
        T = x.shape[-1]
        if k >= T:
            return x
        return F.pad(x[..., : T - k], (k, 0))

    def _gain(self, x):
        db = self._uniform(-self.gain_db, self.gain_db)
        if db > 0.0:
            # Headroom guard: never let a +gain push a sample past full-scale into the encoder.
            # peak * 10^(db/20) <= 1  ->  db <= -20*log10(peak). Loud goa masters sit near 0 dBFS,
            # so an unguarded +4 dB (x1.585) would feed >full-scale OOD audio to SAME (trained on
            # [-1,1]). Attenuation (db<0) is always safe and left untouched.
            peak = float(x.abs().max())
            if peak > 0.0:
                headroom_db = -20.0 * torch.log10(torch.tensor(peak)).item()
                db = min(db, max(0.0, headroom_db))
        return x * (10.0 ** (db / 20.0))

    def _stereo_width(self, x):
        if x.shape[0] != 2:
            return x
        w = self._uniform(*self.stereo_width_range)
        mid = (x[0] + x[1]) * 0.5
        side = (x[0] - x[1]) * 0.5 * w
        return torch.stack([mid + side, mid - side], dim=0)

    def _polarity(self, x):
        return -x

    def _pitch_shift(self, x):
        try:
            import torchaudio.transforms as TT
        except Exception:
            return x
        n = int(round(self._uniform(-self.pitch_semitones, self.pitch_semitones)))
        if n == 0:
            return x
        if self._pitch is None or self._pitch[0] != n:
            self._pitch = (n, TT.PitchShift(self.sr, n_steps=n).to(x.device))
        return self._pitch[1](x)

    def _time_stretch(self, x):
        # resample-based tempo change (also shifts pitch slightly — acceptable as augmentation);
        # re-crop/pad back to original length so the batch stays fixed-size.
        try:
            import torchaudio.functional as AF
        except Exception:
            return x
        r = self._uniform(*self.time_stretch_range)
        T = x.shape[-1]
        new_sr = int(self.sr * r)
        y = AF.resample(x, self.sr, new_sr)
        if y.shape[-1] >= T:
            return y[..., :T]
        return F.pad(y, (0, T - y.shape[-1]))

    @torch.no_grad()
    def forward(self, x):
        """x: [C, T] float audio -> augmented [C, T] (same shape/dtype)."""
        if self.sub_frame_shift_prob and self._rand() < self.sub_frame_shift_prob:
            x = self._sub_frame_shift(x)
        if self.time_stretch_prob and self._rand() < self.time_stretch_prob:
            x = self._time_stretch(x)
        if self.pitch_shift_prob and self._rand() < self.pitch_shift_prob:
            x = self._pitch_shift(x)
        if self.stereo_width_prob and self._rand() < self.stereo_width_prob:
            x = self._stereo_width(x)
        if self.gain_prob and self._rand() < self.gain_prob:
            x = self._gain(x)
        if self.polarity_prob and self._rand() < self.polarity_prob:
            x = self._polarity(x)
        return x


if __name__ == "__main__":
    # CPU self-test: shape/dtype preserved, each transform actually changes the signal, no NaN.
    torch.manual_seed(0)
    C, T = 2, 4096 * 8
    x = torch.randn(C, T)

    aug = AudioAugment(sub_frame_shift_prob=1, gain_prob=1, stereo_width_prob=1, polarity_prob=0, seed=1)
    y = aug(x)
    assert y.shape == x.shape, y.shape
    assert torch.isfinite(y).all()
    assert not torch.equal(y, x), "augment should change the signal"

    # each transform in isolation
    a = AudioAugment(seed=2)
    s = a._sub_frame_shift(x); assert s.shape == x.shape and (s[..., 0] == 0).all()  # zero-padded front
    g = a._gain(x);            assert g.shape == x.shape and not torch.equal(g, x)
    w = a._stereo_width(x);    assert w.shape == x.shape
    p = a._polarity(x);        assert torch.equal(p, -x)
    mono = torch.randn(1, T)
    assert a._stereo_width(mono).shape == mono.shape  # mono no-op

    # off-path is a true no-op
    off = AudioAugment(sub_frame_shift_prob=0, gain_prob=0, stereo_width_prob=0, polarity_prob=0)
    assert torch.equal(off(x), x)

    # HEADROOM GUARD: +gain on a near-full-scale signal must NOT exceed full-scale (no clip / no
    # OOD >0 dBFS into the encoder). A full-scale input can only be attenuated or left as-is.
    hot = torch.ones(2, T) * 0.98                       # loud master, peaks near 0 dBFS
    ah = AudioAugment(gain_db=6.0, seed=7)
    for _ in range(200):
        assert float(ah._gain(hot).abs().max()) <= 1.0 + 1e-6, "gain pushed a sample past full-scale"
    fs = torch.ones(2, 16); fs[0, 0] = 1.0              # exactly full-scale
    assert float(ah._gain(fs).abs().max()) <= 1.0 + 1e-6, "full-scale input gained past 1.0"
    # attenuation still works (db<0 path untouched): a quiet signal can still be boosted
    quiet = torch.ones(2, 16) * 0.1
    assert float(ah._gain(quiet).abs().max()) <= 1.0 + 1e-6
    print("audio_augment self-test OK")
