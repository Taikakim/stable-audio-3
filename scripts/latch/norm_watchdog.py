# Owner: WINTERMUTE (created 2026-10-21)
"""Per-group weight-size watchdog: halve a parameter group's lr when its weights grow too fast.

Why: every cooked LatCH run of October 2026 showed the weight size running away before the loss did.
Test plan B3 (lr 3e-2, sign group uncapped): sign-group weight RMS 0.13 -> 1.8 -> 69 while the spectral
group stayed healthy; the air head at lr 5e-2: spectral weight RMS x15 in under three epochs. Healthy runs
grow too, but slowly (B2: sign group x1.5 per epoch at most after warmup), so the rule compares the RMS now
with the RMS one window ago and acts on the ratio, per group, only after warmup.
"""

import math


class GroupNormWatchdog:
    """Call ``step(step, param_groups)`` after every optimizer step.

    ratio:  cut when a group's weight RMS grew by more than this factor over the last ``window`` steps.
    factor: multiply that group's lr by this on a cut (0.5 = halve).
    start:  first step that is checked (end of warmup: weights grow fast from init by design).
    min_scale: never cut a group below this fraction of its starting lr.
    After a cut the group's history restarts, so the next cut needs a fresh window of growth.
    """

    def __init__(self, n_groups, ratio=2.0, window=1000, start=0, factor=0.5, check_every=None, min_scale=1 / 64):
        if ratio <= 1.0:
            raise ValueError(f"ratio must be > 1, got {ratio}")
        self.ratio, self.window, self.start, self.factor, self.min_scale = ratio, int(window), int(start), factor, min_scale
        self.check_every = int(check_every or max(1, self.window // 4))
        self.hist = [[] for _ in range(n_groups)]
        self.scale = [1.0] * n_groups
        self.events = []

    @staticmethod
    def group_rms(group):
        sq, n = 0.0, 0
        for p in group["params"]:
            sq += float(p.detach().float().pow(2).sum())
            n += p.numel()
        return math.sqrt(sq / max(n, 1))

    def step(self, step, param_groups):
        """Returns the cuts made at this step: dicts with group, step, rms_then, rms_now, ratio, lr_old, lr_new."""
        if step < self.start or (step - self.start) % self.check_every:
            return []
        cuts = []
        for i, g in enumerate(param_groups):
            r = self.group_rms(g)
            h = self.hist[i]
            h.append((step, r))
            old = [k for k, (s, _) in enumerate(h) if step - s >= self.window]
            if not old:
                continue
            ref = old[-1]                       # the most recent point at least one window back
            del h[:ref]
            s0, r0 = h[0]
            if r0 <= 0 or r / r0 <= self.ratio or self.scale[i] * self.factor < self.min_scale:
                continue
            lr_old = g["lr"]
            g["lr"] = lr_old * self.factor
            self.scale[i] *= self.factor
            ev = {"group": g.get("group_type", str(i)), "index": i, "step": step, "since_step": s0,
                  "rms_then": r0, "rms_now": r, "ratio": r / r0, "lr_old": lr_old, "lr_new": g["lr"]}
            cuts.append(ev)
            self.events.append(ev)
            self.hist[i] = [(step, r)]
        return cuts
