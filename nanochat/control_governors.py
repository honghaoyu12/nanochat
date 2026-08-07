"""Autonomous outer-loop governors for nanochat optimizer controllers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import median
from typing import Any

from nanochat.controlled_muon import NanochatMuonControlStats, NanochatMuonController


AUTONOMOUS_COOLDOWN_VARIANTS = {
    "controlled_muon_ema_autocooldown": "controlled_muon_ema",
    "controlled_muon_pi_ema_autocooldown": "controlled_muon_pi_ema",
    "controlled_muon_pid_ema_autocooldown": "controlled_muon_pid_ema",
}


@dataclass
class ValidationProgressStats:
    step: int
    tokens: int
    val_bpb: float
    window_progress_per_billion: float | None
    plateau_candidate: bool
    plateau_streak: int
    plateau_confirmed: bool
    window_observations: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationProgressDetector:
    """Detect sustained low validation progress using a robust token slope."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        window_evals: int = 5,
        patience_windows: int = 2,
        min_relative_progress_per_billion: float = 0.05,
    ) -> None:
        if window_evals < 3:
            raise ValueError("cooldown window_evals must be at least 3")
        if patience_windows < 1:
            raise ValueError("cooldown patience_windows must be positive")
        if (
            not math.isfinite(min_relative_progress_per_billion)
            or min_relative_progress_per_billion < 0
        ):
            raise ValueError("cooldown progress threshold must be finite and non-negative")
        self.window_evals = int(window_evals)
        self.patience_windows = int(patience_windows)
        self.min_relative_progress_per_billion = float(
            min_relative_progress_per_billion
        )
        self.history: list[tuple[int, int, float]] = []
        self.plateau_streak = 0
        self.last_stats: ValidationProgressStats | None = None

    def reset(self) -> None:
        self.history = []
        self.plateau_streak = 0
        self.last_stats = None

    def _progress_rate(self) -> float:
        slopes = []
        for i, (_, tokens_i, bpb_i) in enumerate(self.history):
            for _, tokens_j, bpb_j in self.history[i + 1 :]:
                delta_billions = (tokens_j - tokens_i) / 1e9
                if delta_billions > 0:
                    slopes.append((bpb_j - bpb_i) / delta_billions)
        if not slopes:
            raise RuntimeError("validation progress window has no positive token span")
        bpb_scale = median(abs(item[2]) for item in self.history)
        if not math.isfinite(bpb_scale) or bpb_scale <= 0:
            raise ValueError("validation BPB scale must be finite and positive")
        return -median(slopes) / bpb_scale

    def observe(self, *, step: int, tokens: int, val_bpb: float) -> ValidationProgressStats:
        step_i = int(step)
        tokens_i = int(tokens)
        val_bpb_f = float(val_bpb)
        if step_i < 0 or tokens_i < 0:
            raise ValueError("validation step and tokens must be non-negative")
        if not math.isfinite(val_bpb_f) or val_bpb_f <= 0:
            raise ValueError("validation BPB must be finite and positive")
        if self.history:
            last_step, last_tokens, _ = self.history[-1]
            if step_i <= last_step or tokens_i <= last_tokens:
                raise ValueError("validation step and tokens must increase strictly")

        self.history.append((step_i, tokens_i, val_bpb_f))
        self.history = self.history[-self.window_evals :]
        progress = None
        candidate = False
        if len(self.history) == self.window_evals:
            progress = self._progress_rate()
            candidate = progress <= self.min_relative_progress_per_billion
            self.plateau_streak = self.plateau_streak + 1 if candidate else 0
        else:
            self.plateau_streak = 0
        confirmed = candidate and self.plateau_streak >= self.patience_windows
        stats = ValidationProgressStats(
            step=step_i,
            tokens=tokens_i,
            val_bpb=val_bpb_f,
            window_progress_per_billion=progress,
            plateau_candidate=candidate,
            plateau_streak=self.plateau_streak,
            plateau_confirmed=confirmed,
            window_observations=len(self.history),
        )
        self.last_stats = stats
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "window_evals": self.window_evals,
            "patience_windows": self.patience_windows,
            "min_relative_progress_per_billion": self.min_relative_progress_per_billion,
            "history": [list(item) for item in self.history],
            "plateau_streak": self.plateau_streak,
            "last_stats": None if self.last_stats is None else self.last_stats.as_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", 1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported validation progress detector schema")
        configured = (
            self.window_evals,
            self.patience_windows,
            self.min_relative_progress_per_billion,
        )
        saved = (
            int(state["window_evals"]),
            int(state["patience_windows"]),
            float(state["min_relative_progress_per_billion"]),
        )
        if saved != configured:
            raise ValueError("cooldown detector checkpoint configuration mismatch")
        history = []
        for item in state.get("history", []):
            if len(item) != 3:
                raise ValueError("invalid cooldown detector history item")
            history.append((int(item[0]), int(item[1]), float(item[2])))
        self.history = history[-self.window_evals :]
        self.plateau_streak = int(state.get("plateau_streak", 0))
        last_stats = state.get("last_stats")
        self.last_stats = (
            None if last_stats is None else ValidationProgressStats(**last_stats)
        )


@dataclass
class CooldownValidationStats:
    step: int
    tokens: int
    val_bpb: float
    window_progress_per_billion: float | None
    plateau_candidate: bool
    plateau_streak: int
    plateau_confirmed: bool
    window_observations: int
    governor_state: str
    holdoff_evals_remaining: int
    cooldown_event: bool
    cooldown_event_count: int
    alpha_proposed: float
    alpha_cap_target: float
    alpha_cap: float
    alpha_applied: float
    cap_is_binding: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GovernedControlStats:
    alpha_proposed: float
    alpha_cap_target: float
    alpha_cap: float
    alpha_applied: float
    cap_is_binding: bool
    governor_state: str
    cooldown_event_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AlphaCeilingGovernor:
    """Apply autonomous, bounded, monotone cooldown to an alpha proposal."""

    SCHEMA_VERSION = 1
    MONITORING = "MONITORING"
    REDUCING = "REDUCING"
    HOLDOFF = "HOLDOFF"
    VALID_STATES = {MONITORING, REDUCING, HOLDOFF}

    def __init__(
        self,
        *,
        reference_alpha: float,
        alpha_min: float,
        alpha_max: float,
        detector: ValidationProgressDetector,
        event_log_reduction: float = math.log(2.0),
        transition_steps: int = 200,
        holdoff_evals: int = 2,
        minimum_cap_ratio: float = 0.10,
        maximum_events: int = 3,
    ) -> None:
        numeric = (
            reference_alpha,
            alpha_min,
            alpha_max,
            event_log_reduction,
            minimum_cap_ratio,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("cooldown governor numeric settings must be finite")
        if not 0 < alpha_min <= reference_alpha <= alpha_max:
            raise ValueError("cooldown alpha values must satisfy min <= reference <= max")
        if event_log_reduction <= 0:
            raise ValueError("cooldown event_log_reduction must be positive")
        if transition_steps < 1:
            raise ValueError("cooldown transition_steps must be positive")
        if holdoff_evals < 0:
            raise ValueError("cooldown holdoff_evals must be non-negative")
        if not 0 < minimum_cap_ratio <= 1:
            raise ValueError("cooldown minimum_cap_ratio must be in (0, 1]")
        if maximum_events < 1:
            raise ValueError("cooldown maximum_events must be positive")

        self.reference_alpha = float(reference_alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.detector = detector
        self.event_log_reduction = float(event_log_reduction)
        self.transition_steps = int(transition_steps)
        self.holdoff_evals = int(holdoff_evals)
        self.minimum_cap_ratio = float(minimum_cap_ratio)
        self.maximum_events = int(maximum_events)
        self.alpha_cap_floor = max(
            self.alpha_min, self.reference_alpha * self.minimum_cap_ratio
        )
        if self.alpha_cap_floor > self.alpha_max:
            raise ValueError("cooldown cap floor exceeds alpha_max")

        self.state = self.MONITORING
        self.alpha_cap = self.alpha_max
        self.alpha_cap_target = self.alpha_max
        self.transition_log_step = 0.0
        self.transition_steps_remaining = 0
        self.holdoff_evals_remaining = 0
        self.cooldown_event_count = 0
        self.last_event_step: int | None = None
        self.last_event_tokens: int | None = None
        self.last_prepared_step: int | None = None
        self.last_validation_step: int | None = None
        self.last_validation_tokens: int | None = None
        self.last_alpha_proposed = self.reference_alpha
        self.last_alpha_applied = self.reference_alpha
        self.last_validation_stats: CooldownValidationStats | None = None

    def is_binding(self, alpha: float, *, tolerance: float = 1e-12) -> bool:
        return float(alpha) >= self.alpha_cap - tolerance * max(1.0, abs(self.alpha_cap))

    def constrain(self, alpha_proposed: float) -> float:
        proposed = float(alpha_proposed)
        if not math.isfinite(proposed) or proposed <= 0:
            raise ValueError("governed alpha proposal must be finite and positive")
        self.last_alpha_proposed = proposed
        self.last_alpha_applied = min(proposed, self.alpha_cap)
        return self.last_alpha_applied

    def prepare_step(self, *, step: int, alpha_applied: float) -> float:
        step_i = int(step)
        if step_i < 0:
            raise ValueError("governor step must be non-negative")
        if self.last_prepared_step is not None:
            if step_i < self.last_prepared_step:
                raise ValueError("governor prepare_step cannot move backward")
            if step_i == self.last_prepared_step:
                return self.constrain(alpha_applied)
        self.last_prepared_step = step_i

        if self.state == self.REDUCING:
            self.alpha_cap = max(
                self.alpha_cap_target,
                self.alpha_cap * math.exp(-self.transition_log_step),
            )
            self.transition_steps_remaining = max(
                0, self.transition_steps_remaining - 1
            )
            if self.transition_steps_remaining == 0 or math.isclose(
                self.alpha_cap, self.alpha_cap_target, rel_tol=1e-12, abs_tol=1e-15
            ):
                self.alpha_cap = self.alpha_cap_target
                self.state = self.HOLDOFF
                self.holdoff_evals_remaining = self.holdoff_evals
                self.detector.reset()
                if self.holdoff_evals_remaining == 0:
                    self.state = self.MONITORING
        return self.constrain(alpha_applied)

    def observe_validation(
        self,
        *,
        step: int,
        tokens: int,
        val_bpb: float,
        allow_event: bool = True,
    ) -> CooldownValidationStats:
        step_i = int(step)
        tokens_i = int(tokens)
        val_bpb_f = float(val_bpb)
        if step_i < 0 or tokens_i < 0:
            raise ValueError("validation step and tokens must be non-negative")
        if not math.isfinite(val_bpb_f) or val_bpb_f <= 0:
            raise ValueError("validation BPB must be finite and positive")
        if self.last_validation_step is not None and (
            step_i <= self.last_validation_step
            or tokens_i <= self.last_validation_tokens
        ):
            raise ValueError("validation step and tokens must increase strictly")
        self.last_validation_step = step_i
        self.last_validation_tokens = tokens_i

        progress_stats = None
        cooldown_event = False
        if self.state == self.MONITORING:
            progress_stats = self.detector.observe(
                step=step, tokens=tokens, val_bpb=val_bpb
            )
            can_reduce = (
                allow_event
                and progress_stats.plateau_confirmed
                and self.cooldown_event_count < self.maximum_events
                and self.alpha_cap_target > self.alpha_cap_floor
            )
            if can_reduce:
                cap_at_event = min(self.alpha_cap, self.last_alpha_applied)
                target = max(
                    self.alpha_cap_floor,
                    cap_at_event * math.exp(-self.event_log_reduction),
                )
                if target < cap_at_event:
                    self.alpha_cap = cap_at_event
                    self.alpha_cap_target = target
                    self.transition_log_step = (
                        math.log(cap_at_event / target) / self.transition_steps
                    )
                    self.transition_steps_remaining = self.transition_steps
                    self.state = self.REDUCING
                    self.cooldown_event_count += 1
                    self.last_event_step = int(step)
                    self.last_event_tokens = int(tokens)
                    cooldown_event = True
                    self.detector.reset()
        elif self.state == self.HOLDOFF:
            self.holdoff_evals_remaining = max(
                0, self.holdoff_evals_remaining - 1
            )
            if self.holdoff_evals_remaining == 0:
                self.state = self.MONITORING
                self.detector.reset()
        elif self.state != self.REDUCING:
            raise RuntimeError(f"unknown cooldown governor state: {self.state}")

        if progress_stats is None:
            progress, candidate, streak, confirmed, observations = (
                None, False, 0, False, 0
            )
        else:
            progress = progress_stats.window_progress_per_billion
            candidate = progress_stats.plateau_candidate
            streak = progress_stats.plateau_streak
            confirmed = progress_stats.plateau_confirmed
            observations = progress_stats.window_observations

        stats = CooldownValidationStats(
            step=step_i,
            tokens=tokens_i,
            val_bpb=val_bpb_f,
            window_progress_per_billion=progress,
            plateau_candidate=candidate,
            plateau_streak=streak,
            plateau_confirmed=confirmed,
            window_observations=observations,
            governor_state=self.state,
            holdoff_evals_remaining=self.holdoff_evals_remaining,
            cooldown_event=cooldown_event,
            cooldown_event_count=self.cooldown_event_count,
            alpha_proposed=self.last_alpha_proposed,
            alpha_cap_target=self.alpha_cap_target,
            alpha_cap=self.alpha_cap,
            alpha_applied=self.last_alpha_applied,
            cap_is_binding=self.is_binding(self.last_alpha_applied),
        )
        self.last_validation_stats = stats
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "reference_alpha": self.reference_alpha,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "event_log_reduction": self.event_log_reduction,
            "transition_steps": self.transition_steps,
            "holdoff_evals": self.holdoff_evals,
            "minimum_cap_ratio": self.minimum_cap_ratio,
            "maximum_events": self.maximum_events,
            "alpha_cap_floor": self.alpha_cap_floor,
            "state": self.state,
            "alpha_cap": self.alpha_cap,
            "alpha_cap_target": self.alpha_cap_target,
            "transition_log_step": self.transition_log_step,
            "transition_steps_remaining": self.transition_steps_remaining,
            "holdoff_evals_remaining": self.holdoff_evals_remaining,
            "cooldown_event_count": self.cooldown_event_count,
            "last_event_step": self.last_event_step,
            "last_event_tokens": self.last_event_tokens,
            "last_prepared_step": self.last_prepared_step,
            "last_validation_step": self.last_validation_step,
            "last_validation_tokens": self.last_validation_tokens,
            "last_alpha_proposed": self.last_alpha_proposed,
            "last_alpha_applied": self.last_alpha_applied,
            "last_validation_stats": (
                None
                if self.last_validation_stats is None
                else self.last_validation_stats.as_dict()
            ),
            "detector": self.detector.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", 1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported alpha ceiling governor schema")
        configured = {
            "reference_alpha": self.reference_alpha,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "event_log_reduction": self.event_log_reduction,
            "transition_steps": self.transition_steps,
            "holdoff_evals": self.holdoff_evals,
            "minimum_cap_ratio": self.minimum_cap_ratio,
            "maximum_events": self.maximum_events,
        }
        for name, value in configured.items():
            saved = state[name]
            matches = (
                math.isclose(float(saved), value, rel_tol=1e-12, abs_tol=1e-15)
                if isinstance(value, float)
                else int(saved) == value
            )
            if not matches:
                raise ValueError(f"cooldown governor checkpoint mismatch for {name}")
        state_name = str(state["state"])
        if state_name not in self.VALID_STATES:
            raise ValueError(f"invalid saved cooldown governor state: {state_name}")
        self.state = state_name
        self.alpha_cap = float(state["alpha_cap"])
        self.alpha_cap_target = float(state["alpha_cap_target"])
        self.transition_log_step = float(state.get("transition_log_step", 0.0))
        self.transition_steps_remaining = int(
            state.get("transition_steps_remaining", 0)
        )
        self.holdoff_evals_remaining = int(
            state.get("holdoff_evals_remaining", 0)
        )
        self.cooldown_event_count = int(state.get("cooldown_event_count", 0))
        self.last_event_step = state.get("last_event_step")
        self.last_event_tokens = state.get("last_event_tokens")
        self.last_prepared_step = state.get("last_prepared_step")
        self.last_validation_step = state.get("last_validation_step")
        self.last_validation_tokens = state.get("last_validation_tokens")
        self.last_alpha_proposed = float(
            state.get("last_alpha_proposed", self.reference_alpha)
        )
        self.last_alpha_applied = float(
            state.get("last_alpha_applied", self.reference_alpha)
        )
        last_stats = state.get("last_validation_stats")
        self.last_validation_stats = (
            None if last_stats is None else CooldownValidationStats(**last_stats)
        )
        self.detector.load_state_dict(state["detector"])


class GovernedMuonController:
    """Compose an unchanged Muon feedback controller with an alpha ceiling."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        variant: str,
        base_controller: NanochatMuonController,
        governor: AlphaCeilingGovernor,
    ) -> None:
        expected_base = AUTONOMOUS_COOLDOWN_VARIANTS.get(variant)
        if expected_base is None:
            raise ValueError(f"unknown autonomous cooldown variant: {variant}")
        if base_controller.variant != expected_base:
            raise ValueError(
                f"{variant} requires base controller {expected_base}, "
                f"got {base_controller.variant}"
            )
        self.variant = variant
        self.base_controller = base_controller
        self.governor = governor
        self.last_governed_stats: GovernedControlStats | None = None

    @property
    def alpha(self) -> float:
        return self.base_controller.alpha

    @property
    def num_updates(self) -> int:
        return self.base_controller.num_updates

    def prepare_step(self, step: int) -> float:
        applied = self.governor.prepare_step(
            step=step, alpha_applied=self.base_controller.alpha
        )
        self.base_controller.set_alpha(applied)
        return applied

    def update(self, **kwargs: Any) -> NanochatMuonControlStats:
        cap_was_binding = self.governor.is_binding(self.base_controller.alpha)
        stats = self.base_controller.update(
            **kwargs, freeze_positive_integral=cap_was_binding
        )
        alpha_proposed = stats.alpha_next
        alpha_applied = self.governor.constrain(alpha_proposed)
        self.base_controller.set_alpha(alpha_applied)
        stats.alpha_next = alpha_applied
        stats.alpha_update_factor = alpha_applied / stats.alpha
        stats.factor_applied = stats.alpha_update_factor
        self.last_governed_stats = GovernedControlStats(
            alpha_proposed=alpha_proposed,
            alpha_cap_target=self.governor.alpha_cap_target,
            alpha_cap=self.governor.alpha_cap,
            alpha_applied=alpha_applied,
            cap_is_binding=self.governor.is_binding(alpha_applied),
            governor_state=self.governor.state,
            cooldown_event_count=self.governor.cooldown_event_count,
        )
        return stats

    def observe_validation(self, **kwargs: Any) -> CooldownValidationStats:
        return self.governor.observe_validation(**kwargs)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "variant": self.variant,
            "base_controller": self.base_controller.state_dict(),
            "governor": self.governor.state_dict(),
            "last_governed_stats": (
                None
                if self.last_governed_stats is None
                else self.last_governed_stats.as_dict()
            ),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", 1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported governed Muon controller schema")
        if state.get("variant") != self.variant:
            raise ValueError(
                f"cannot load {state.get('variant')} state into {self.variant}"
            )
        self.base_controller.load_state_dict(state["base_controller"])
        self.governor.load_state_dict(state["governor"])
        self.base_controller.set_alpha(
            min(self.base_controller.alpha, self.governor.alpha_cap)
        )
        last_stats = state.get("last_governed_stats")
        self.last_governed_stats = (
            None if last_stats is None else GovernedControlStats(**last_stats)
        )

    def diagnostics_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "alpha": self.alpha,
            "num_updates": self.num_updates,
            "base_controller": self.base_controller.diagnostics_dict(),
            "governor": self.governor.state_dict(),
            "last_governed_stats": (
                None
                if self.last_governed_stats is None
                else self.last_governed_stats.as_dict()
            ),
        }
