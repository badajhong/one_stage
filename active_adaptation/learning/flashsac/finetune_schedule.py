"""Causal, sample-weighted decisions for adaptive perception/SAC phases.

Each recorded observation is one complete rollout iteration. The first SAC
reference can also include its current partial iteration before learning starts.
Errors are RMS values over valid states, not an unweighted average of
per-iteration RMS values. This module has no torch or simulator dependency so
schedule behavior can be tested on CPU.
"""

import math
from collections import deque
from numbers import Integral


_POSITIVE_INTEGER_FIELDS = (
    "perception_error_window", "perception_trigger_patience", "plateau_window",
    "plateau_patience", "plateau_check_interval", "cycle_perception_iters_min",
    "cycle_perception_iters_max", "cycle_sac_iters_min", "cycle_sac_iters_max",
)


def validate_config(cfg):
    """Reject malformed bounds; a window larger than the phase cap is allowed."""
    for name in _POSITIVE_INTEGER_FIELDS:
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("perception_trigger", "plateau_tol"):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0")
    for phase in ("perception", "sac"):
        if getattr(cfg, f"cycle_{phase}_iters_min") > getattr(cfg, f"cycle_{phase}_iters_max"):
            raise ValueError(f"cycle_{phase}_iters_min must not exceed cycle_{phase}_iters_max")


class AdaptivePerceptionSchedule:
    """Track contiguous rollout errors and decide when the current phase ends.

    The reference is the recent error observed before the first SAC update or
    achieved by a refresh, not a claimed lower bound on the best possible error.
    The caller resets phase history at each transition and supplies the number
    of completed phase iterations.
    """

    def __init__(self, cfg):
        validate_config(cfg)
        self.cfg = cfg
        self.reference_error = None
        self._history = deque(maxlen=max(2 * cfg.plateau_window, cfg.perception_error_window))
        self.reset_phase()

    def reset_phase(self):
        """Discard observations from the previous policy/perception phase."""
        self._history.clear()
        self.last_error = None
        self.smoothed_error = None
        self.relative_error = None
        self.plateau_improvement = None
        self.trigger_streak = 0
        self.plateau_streak = 0
        self._last_sac_iteration = None
        self._last_sac_reason = None
        self._last_refresh_iteration = None
        self._last_refresh_reason = None
        self._last_plateau_iteration = None

    @staticmethod
    def _rms(observations):
        return math.sqrt(math.fsum(pair[0] for pair in observations)
                         / math.fsum(pair[1] for pair in observations))

    def _update_error_metrics(self):
        if self._history:
            self.smoothed_error = self._rms(list(self._history)[-self.cfg.perception_error_window:])
            if self.reference_error is not None:
                self.relative_error = ((self.smoothed_error - self.reference_error)
                                       / max(self.reference_error, 1e-12))

    def observe(self, mse_sum, valid_count):
        """Record a rollout, breaking both decision streaks on invalid data.

        ``mse_sum`` sums each state's mean squared latent error. Missing data must
        not look like zero error or bridge two otherwise disconnected windows.
        """
        mse_sum, valid_count = float(mse_sum), float(valid_count)
        if (not math.isfinite(mse_sum) or not math.isfinite(valid_count)
                or mse_sum < 0 or valid_count <= 0):
            self._history.clear()
            self.last_error = self.smoothed_error = self.relative_error = None
            self.plateau_improvement = None
            self.trigger_streak = self.plateau_streak = 0
            self._last_plateau_iteration = None
            return
        self._history.append((mse_sum, valid_count))
        self.last_error = math.sqrt(mse_sum / valid_count)
        self._update_error_metrics()

    @staticmethod
    def _validate_iteration(iterations):
        if isinstance(iterations, bool) or not isinstance(iterations, Integral) or iterations < 0:
            raise ValueError("completed phase iterations must be a nonnegative integer")

    def sac_reason(self, iterations):
        """Return an error trigger after cooldown, or the unconditional SAC cap."""
        self._validate_iteration(iterations)
        if iterations >= self.cfg.cycle_sac_iters_max:
            return "sac_max"
        if iterations == self._last_sac_iteration:
            return self._last_sac_reason
        if self._last_sac_iteration is not None and iterations != self._last_sac_iteration + 1:
            self.trigger_streak = 0
        self._last_sac_iteration = iterations
        self._last_sac_reason = None
        if (iterations < self.cfg.cycle_sac_iters_min
                or len(self._history) < self.cfg.perception_error_window
                or self.relative_error is None):
            self.trigger_streak = 0
            return None
        if self.relative_error >= self.cfg.perception_trigger:
            self.trigger_streak += 1
        else:
            self.trigger_streak = 0
        if self.trigger_streak >= self.cfg.perception_trigger_patience:
            self._last_sac_reason = "error_trigger"
        return self._last_sac_reason

    def refresh_reason(self, iterations):
        """Compare adjacent rollout windows at spaced, consecutive checks."""
        self._validate_iteration(iterations)
        if iterations >= self.cfg.cycle_perception_iters_max:
            return "perception_max"
        if iterations == self._last_refresh_iteration:
            return self._last_refresh_reason
        self._last_refresh_iteration = iterations
        self._last_refresh_reason = None
        window = self.cfg.plateau_window
        if (iterations < self.cfg.cycle_perception_iters_min or len(self._history) < 2 * window):
            self.plateau_streak = 0
            return None
        if iterations % self.cfg.plateau_check_interval:
            return None
        if (self._last_plateau_iteration is not None
                and iterations != self._last_plateau_iteration + self.cfg.plateau_check_interval):
            self.plateau_streak = 0
        self._last_plateau_iteration = iterations
        observations = list(self._history)
        previous = self._rms(observations[-2 * window:-window])
        current = self._rms(observations[-window:])
        # The equivalent difference form also treats two all-zero windows as a
        # plateau, while worsening from zero remains a negative improvement.
        self.plateau_improvement = (previous - current) / max(previous, 1e-12)
        if self.plateau_improvement < self.cfg.plateau_tol:
            self.plateau_streak += 1
        else:
            self.plateau_streak = 0
        if self.plateau_streak >= self.cfg.plateau_patience:
            self._last_refresh_reason = "plateau"
        return self._last_refresh_reason

    def establish_reference(self):
        """Use the valid recent tail achieved at the end of a refresh."""
        if not self._history:
            raise RuntimeError("cannot establish perception reference without a valid rollout observation")
        self.reference_error = self._rms(list(self._history)[-self.cfg.perception_error_window:])
        self._update_error_metrics()
        return self.reference_error

    def try_initialize_reference(self, mse_sum, valid_count):
        """Set the first SAC reference from rollout data collected before learning.

        The current partial iteration occupies one slot in the recent error
        window. It is only used for this estimate: ``observe`` must still record
        the complete iteration later, so no samples enter history twice. Invalid
        current data returns False without using older history as a substitute.
        A reference already established by this method or a refresh is preserved.
        """
        if self.reference_error is not None:
            return False
        try:
            mse_sum, valid_count = float(mse_sum), float(valid_count)
        except (TypeError, ValueError):
            return False
        if (not math.isfinite(mse_sum) or not math.isfinite(valid_count)
                or mse_sum < 0 or valid_count <= 0):
            return False
        observations = [*self._history, (mse_sum, valid_count)]
        self.reference_error = self._rms(observations[-self.cfg.perception_error_window:])
        self._update_error_metrics()
        return True

    def state_dict(self):
        """Include window and streak state so resuming preserves causal decisions."""
        return {
            "version": 1,
            "history": [list(pair) for pair in self._history],
            **{name: getattr(self, name) for name in (
                "reference_error", "last_error", "smoothed_error", "relative_error",
                "plateau_improvement", "trigger_streak", "plateau_streak",
                "_last_sac_iteration", "_last_sac_reason", "_last_refresh_iteration",
                "_last_refresh_reason", "_last_plateau_iteration",
            )},
        }

    def load_state_dict(self, state):
        """Validate before replacing state; retain the newest data if windows changed."""
        if state.get("version") != 1:
            raise ValueError("unsupported adaptive perception schedule state version")
        restored = {}
        history = []
        for pair in state["history"]:
            if len(pair) != 2:
                raise ValueError("adaptive schedule history entries must be (mse_sum, count)")
            mse_sum, count = map(float, pair)
            if not math.isfinite(mse_sum) or mse_sum < 0 or not math.isfinite(count) or count <= 0:
                raise ValueError("adaptive schedule history contains an invalid observation")
            history.append((mse_sum, count))
        for name in ("reference_error", "last_error", "smoothed_error", "relative_error", "plateau_improvement"):
            value = state[name]
            if value is not None:
                value = float(value)
                if not math.isfinite(value) or (name.endswith("error") and name != "relative_error" and value < 0):
                    raise ValueError(f"invalid adaptive schedule {name}")
            restored[name] = value
        for name in ("trigger_streak", "plateau_streak", "_last_sac_iteration",
                     "_last_refresh_iteration", "_last_plateau_iteration"):
            value = state[name]
            if value is None and name.startswith("_last_"):
                restored[name] = None
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"invalid adaptive schedule {name}")
            restored[name] = int(value)
        for name, reason in (("_last_sac_reason", "error_trigger"), ("_last_refresh_reason", "plateau")):
            if state[name] not in (None, reason):
                raise ValueError(f"invalid adaptive schedule {name}")
            restored[name] = state[name]
        self._history.clear()
        self._history.extend(history)
        for name, value in restored.items():
            setattr(self, name, value)
        self._update_error_metrics()
