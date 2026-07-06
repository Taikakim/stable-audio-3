"""Familiarity-normalized loss weighting (Kim 2026-07-07).

Idea: "bump down the gradient update multipliers for stuff already close to what
the model does ... but let the remote areas grow." Per-crop EMA of the crop's own
RELATIVE training loss (its per-sample loss divided by the batch mean, which
removes most of the timestep-draw luck) is the familiarity signal: consistently
low relative loss = familiar = down-weight; consistently high = remote = keep or
boost the gradient.

Design constraints:
  * batch weights normalized to mean 1.0 -> effective LR unchanged
  * unseen crops get neutral pre-normalization weight 1.0 (first epoch = natural
    warm-up: everything neutral until history exists)
  * pre-normalization weights clipped to `clip` so one outlier crop can't
    dominate a batch
  * beta=0 disables (all weights exactly 1.0)
"""


class FamiliarityReweighter:
    def __init__(self, beta=1.0, decay=0.9, clip=(0.25, 4.0)):
        self.beta = float(beta)
        self.decay = float(decay)
        self.clip = (float(clip[0]), float(clip[1]))
        self.ema = {}  # crop id -> EMA of relative per-sample loss

    def weights(self, ids, rel_losses):
        """ids: hashable per-crop identifiers; rel_losses: this batch's per-sample
        losses (any positive scale — normalized to batch-relative internally).
        Returns a list of weights, mean exactly 1.0. Also updates the EMAs."""
        rel_losses = [float(x) for x in rel_losses]
        m = sum(rel_losses) / len(rel_losses)
        rel = [x / m if m > 0 else 1.0 for x in rel_losses]

        raw = []
        for i in ids:
            e = self.ema.get(i)
            if e is None or self.beta == 0.0:
                raw.append(1.0)
            else:
                w = e ** self.beta
                raw.append(min(max(w, self.clip[0]), self.clip[1]))

        # update EMAs AFTER computing weights (weight from past visits only)
        for i, r in zip(ids, rel):
            e = self.ema.get(i)
            self.ema[i] = r if e is None else self.decay * e + (1 - self.decay) * r

        mean_w = sum(raw) / len(raw)
        return [w / mean_w for w in raw]
