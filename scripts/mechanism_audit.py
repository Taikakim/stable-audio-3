"""Mechanism audit — say, out loud, which optimizer mechanisms are actually doing anything.

CONTINUITY, 2026-09-22.

WHY THIS EXISTS. A run named `modular_opt_cubic5_sf_ev_lr3e-4_radbrake08` was launched with
--modular-ev, --var-dampening, --var-barrier-weight and --var-damp-opt. All three of those
mechanisms were INERT for the entire run:

  * escape velocity   -> d_t is initialised to 1.0 and only ever grows via max(d, d_hat).
                         Measured d_hat ~1.2e-3 at correctly-dimensioned lr, so the branch
                         returns exactly 1.0 forever, while costing two blocking .item()
                         GPU syncs per parameter per step.
  * VADD tier 1       -> train/var_barrier_loss read 0.000 on every logged step.
  * VADD tier 2       -> running_latent_std was 0.92-1.12, never above the 1.20 threshold,
                         so the dampening branch never fired.

None of this raised an error. The run simply attributed its behaviour to machinery that was
not running. That is the accept-but-ignore failure this project keeps rediscovering, and it
is the tool-level companion to "audit the instrument before you believe a null".

TWO KINDS OF INERT, and they need different checks:

  (a) STATICALLY UNREACHABLE -- the flag cannot affect anything given the other flags
      (e.g. --modular-sf-r with --modular-schedule-free absent). Detectable at launch,
      before a single step runs. That is audit_args().

  (b) DYNAMICALLY INERT -- the code runs every step but its output never leaves the identity
      value, because the data never triggers it. Only detectable by watching. That is
      MechanismAuditCallback, which reads the per-component telemetry the optimizer already
      computes (`_comp_telem`) rather than instrumenting anything new.

Enabled by default; set SA3_MECHANISM_AUDIT=0 to silence. It is print-only and reads an
already-populated dict, so it costs nothing.
"""

import os

import pytorch_lightning as pl


# Telemetry key -> (human name, identity value, the flag that asks for it).
# "Identity" = the value the multiplier takes when the mechanism is doing nothing at all.
_MECHANISMS = {
    "comp/ev_d":              ("escape velocity (Prodigy d_t)", 1.0, "--modular-ev"),
    "comp/snr_gate":          ("SNR gate",                      1.0, "--modular-snr-gate"),
    "comp/var_damp_kappa":    ("VADD tier 2 (step dampening)",  1.0, "--var-damp-opt"),
    "comp/radial_brake_scale":("radial brake",                  1.0, "--modular-radial-brake"),
    "comp/sf_ck":             ("Schedule-Free averaging c_k",   1.0, "--modular-schedule-free"),
    "comp/normuon_gain":      ("NorMuon row scaling",           1.0, "--modular-normuon"),
}

_TOL = 1e-9


def audit_args(args) -> list[str]:
    """Static reachability check: flags that cannot reach code given the other flags.

    Returns a list of human-readable warnings. Empty list means nothing obviously unreachable
    -- NOT that every mechanism is active, which only the runtime audit can tell you.
    """
    warn: list[str] = []
    g = lambda n, d=None: getattr(args, n, d)

    if g("optimizer") != "modular":
        return warn

    sf_on = bool(g("modular_schedule_free", False))
    if not sf_on:
        for flag, attr in (("--modular-sf-c-warmup", "modular_sf_c_warmup"),
                           ("--modular-sf-r", "modular_sf_r")):
            if g(attr) is not None:
                warn.append(f"{flag} is set but --modular-schedule-free is OFF: "
                            f"Schedule-Free averaging never runs, so this value is ignored.")

    if g("var_dampening") is None:
        for flag, attr in (("--var-barrier-weight", "var_barrier_weight"),
                           ("--var-damp-opt", "var_damp_opt")):
            if g(attr):
                warn.append(f"{flag} is set but --var-dampening is unset: BOTH VADD tiers are "
                            f"disabled (train.py gates them on var_dampening being not-None).")

    wd = g("modular_wd", 0.0) or 0.0
    if wd <= 0 and g("modular_wd_overtraining", False):
        warn.append("--modular-wd-overtraining is set but --modular-wd is 0: the overtraining "
                    "sqrt(epoch) scaling multiplies a zero weight decay and does nothing.")

    if g("modular_ev", False):
        warn.append("--modular-ev: escape velocity starts at d=1.0 and only grows. At a "
                    "correctly-dimensioned lr the measured d_hat is ~1e-3, so d stays pinned "
                    "at 1.0 while costing 2 GPU syncs per parameter per step. Expect the "
                    "runtime audit below to report it INERT; see "
                    "docs/POLYAK_STEP_SIZE_AND_VELOCITY_DYNAMICS.md section 5.")

    return warn


class MechanismAuditCallback(pl.Callback):
    """Track whether each mechanism's multiplier ever leaves its identity value."""

    def __init__(self, report_every: int = 500):
        self.report_every = report_every
        # key -> [ever_moved, max_abs_deviation, n_samples]
        self.seen: dict[str, list] = {}
        self.barrier_nonzero = False
        self.max_latent_std = 0.0
        self.barrier_weight = 0.0
        self.barrier_thresh = 0.0
        self.reported = False

    def _sample(self, pl_module, trainer):
        for opt in (getattr(trainer, "optimizers", None) or []):
            inner = getattr(opt, "optimizer", opt)
            telem = getattr(inner, "_comp_telem", None)
            if not telem:
                continue
            for key, (_name, identity, _flag) in _MECHANISMS.items():
                if key not in telem:
                    continue
                dev = abs(float(telem[key]) - identity)
                rec = self.seen.setdefault(key, [False, 0.0, 0])
                rec[0] = rec[0] or dev > _TOL
                rec[1] = max(rec[1], dev)
                rec[2] += 1

        # VADD tier 1 is a hinge: loss += w * relu(std - barrier)^2. It contributes exactly
        # zero unless the weight is non-zero AND the observed std exceeds the barrier. Both
        # are module attributes, so this needs no extra instrumentation.
        std = getattr(pl_module, "running_latent_std", None)
        if std is not None:
            self.max_latent_std = max(self.max_latent_std, float(std))
        self.barrier_weight = float(getattr(pl_module, "latent_var_weight", 0.0) or 0.0)
        self.barrier_thresh = float(getattr(pl_module, "latent_var_barrier", 0.0) or 0.0)
        if (self.barrier_weight > 0 and self.barrier_thresh > 0
                and self.max_latent_std > self.barrier_thresh):
            self.barrier_nonzero = True

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Sample BEFORE diffusion.py's on_train_batch_end clears _comp_telem. Callback hooks
        # run before the LightningModule hook, so this ordering holds.
        self._sample(pl_module, trainer)
        step = trainer.global_step
        if step and step % self.report_every == 0:
            self.report(trainer, final=False)

    def on_train_end(self, trainer, pl_module):
        self.report(trainer, final=True)

    def report(self, trainer, final: bool):
        if not self.seen:
            print("\n[MECHANISM AUDIT] No component telemetry seen -- cannot judge activity. "
                  "This means the optimizer never populated _comp_telem.", flush=True)
            return
        head = "FINAL" if final else f"step {trainer.global_step}"
        lines = [f"\n[MECHANISM AUDIT] {head} -- is each mechanism actually doing anything?"]
        inert = []
        for key, (name, identity, flag) in _MECHANISMS.items():
            rec = self.seen.get(key)
            if rec is None:
                continue
            moved, maxdev, n = rec
            if moved:
                lines.append(f"    ACTIVE  {name:<32} max deviation from {identity}: {maxdev:.4g}  ({n} samples)")
            else:
                lines.append(f"    INERT   {name:<32} never left {identity} in {n} samples  [{flag}]")
                inert.append((name, flag))

        if self.barrier_weight <= 0 or self.barrier_thresh <= 0:
            detail = "disabled (weight or barrier is 0)"
        elif self.barrier_nonzero:
            detail = (f"latent std reached {self.max_latent_std:.3f} > barrier "
                      f"{self.barrier_thresh:.3f}")
        else:
            detail = (f"max latent std {self.max_latent_std:.3f} never exceeded barrier "
                      f"{self.barrier_thresh:.3f}, so the hinge stayed 0.0")
        lines.append(f"    {'ACTIVE ' if self.barrier_nonzero else 'INERT  '} "
                     f"{'VADD tier 1 (barrier loss)':<32} {detail}")
        if not self.barrier_nonzero:
            inert.append(("VADD tier 1 (barrier loss)", "--var-barrier-weight"))
        if inert:
            lines.append(f"    => {len(inert)} mechanism(s) had NO effect on this run. Any conclusion "
                         f"attributing behaviour to them is unsupported.")
        print("\n".join(lines) + "\n", flush=True)


def maybe_build(args=None, report_every: int = 500):
    """Print the static audit and return the runtime callback (None if disabled)."""
    if os.environ.get("SA3_MECHANISM_AUDIT", "1") in ("0", "false", "no"):
        return None
    if args is not None:
        for w in audit_args(args):
            print(f"[MECHANISM AUDIT] WARNING: {w}", flush=True)
    return MechanismAuditCallback(report_every=report_every)
