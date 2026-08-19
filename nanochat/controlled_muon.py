"""Controlled Muon controller utilities for nanochat experiments.

This module intentionally contains only scalar controller logic. The actual
parameter update remains nanochat.optim.MuonAdamW.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


ABSOLUTE_GROUP_SCALING_MODES = {"uniform", "initial_lr_ratio"}
CONTROL_FEEDBACK_SCOPES = {"total", "muon_residual_proxy"}
CONTROL_ACTION_POLICIES = {"legacy", "phase_hold", "phase_hold_recovery"}
PHASE_HOLD_CRUISE_POLICIES = {"hold", "rho_deadband"}
CONTROL_FEEDBACK_STATE_SCHEMA_VERSION = 1


def _deadband(value: float, width: float) -> float:
    magnitude = max(0.0, abs(value) - width)
    return math.copysign(magnitude, value) if magnitude > 0.0 else 0.0


@dataclass(frozen=True)
class SelectedControlFeedback:
    scope: str
    actual_total: float
    predicted_total: float
    actual_for_control: float
    predicted_for_control: float
    grad_norm_for_control: float
    update_norm_for_control: float
    rho_total: float | None
    rho_muon_residual_proxy: float | None


def _positive_denominator_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return None
    ratio = numerator / denominator
    return ratio if math.isfinite(ratio) else None


def select_control_feedback(
    *,
    scope: str,
    actual_total: float,
    predicted_total: float,
    predicted_muon: float,
    predicted_adamw: float,
    total_grad_norm: float,
    muon_grad_norm: float,
    total_update_norm: float,
    muon_update_norm: float,
) -> SelectedControlFeedback:
    """Select the observation consumed by the scalar alpha controller."""
    if scope not in CONTROL_FEEDBACK_SCOPES:
        raise ValueError(f"unknown control feedback scope: {scope}")

    actual_total_f = float(actual_total)
    predicted_total_f = float(predicted_total)
    predicted_muon_f = float(predicted_muon)
    predicted_adamw_f = float(predicted_adamw)
    muon_residual = actual_total_f - predicted_adamw_f

    if scope == "total":
        actual_for_control = actual_total_f
        predicted_for_control = predicted_total_f
        grad_norm_for_control = float(total_grad_norm)
        update_norm_for_control = float(total_update_norm)
    else:
        actual_for_control = muon_residual
        predicted_for_control = predicted_muon_f
        grad_norm_for_control = float(muon_grad_norm)
        update_norm_for_control = float(muon_update_norm)

    return SelectedControlFeedback(
        scope=scope,
        actual_total=actual_total_f,
        predicted_total=predicted_total_f,
        actual_for_control=actual_for_control,
        predicted_for_control=predicted_for_control,
        grad_norm_for_control=grad_norm_for_control,
        update_norm_for_control=update_norm_for_control,
        rho_total=_positive_denominator_ratio(actual_total_f, predicted_total_f),
        rho_muon_residual_proxy=_positive_denominator_ratio(muon_residual, predicted_muon_f),
    )


def validate_control_feedback_configuration(
    *, scope: str, controlled: bool, control_scope: str
) -> None:
    if scope not in CONTROL_FEEDBACK_SCOPES:
        raise ValueError(f"unknown control feedback scope: {scope}")
    if scope == "total":
        return
    if not controlled:
        raise ValueError("muon_residual_proxy feedback requires a controlled optimizer")
    if control_scope not in {"muon_only", "all_groups"}:
        raise ValueError(f"unknown control scope: {control_scope}")


def control_feedback_state_dict(scope: str) -> dict[str, Any]:
    if scope not in CONTROL_FEEDBACK_SCOPES:
        raise ValueError(f"unknown control feedback scope: {scope}")
    return {
        "schema_version": CONTROL_FEEDBACK_STATE_SCHEMA_VERSION,
        "scope": scope,
    }


def validate_resumed_control_feedback(
    *, configured_scope: str, saved_state: dict[str, Any] | None
) -> None:
    """Reject resumes that would silently change feedback semantics."""
    if configured_scope not in CONTROL_FEEDBACK_SCOPES:
        raise ValueError(f"unknown control feedback scope: {configured_scope}")
    if saved_state is None:
        if configured_scope != "total":
            raise ValueError(
                "legacy checkpoint has no feedback scope; resume it with total feedback"
            )
        return
    schema_version = int(saved_state.get("schema_version", 1))
    if schema_version != CONTROL_FEEDBACK_STATE_SCHEMA_VERSION:
        raise ValueError(f"unsupported control feedback schema: {schema_version}")
    saved_scope = saved_state.get("scope", "total")
    if saved_scope not in CONTROL_FEEDBACK_SCOPES:
        raise ValueError(f"unknown saved control feedback scope: {saved_scope}")
    if saved_scope != configured_scope:
        raise ValueError(
            f"cannot resume {saved_scope} feedback with {configured_scope} configured"
        )


def validate_absolute_group_scaling(
    *,
    mode: str,
    controlled: bool,
    control_scope: str,
    alpha_mode: str,
    alpha_reference: str,
) -> None:
    """Validate fixed group coefficients for absolute all-groups control."""
    if mode not in ABSOLUTE_GROUP_SCALING_MODES:
        raise ValueError(f"unknown absolute group scaling mode: {mode}")
    if mode == "uniform":
        return
    if not controlled:
        raise ValueError("initial_lr_ratio group scaling requires a controlled optimizer")
    if control_scope != "all_groups":
        raise ValueError("initial_lr_ratio group scaling requires control-scope=all_groups")
    if alpha_mode != "absolute":
        raise ValueError("initial_lr_ratio group scaling requires control-alpha-mode=absolute")
    if alpha_reference != "none":
        raise ValueError("initial_lr_ratio group scaling requires control-alpha-reference=none")


def absolute_group_lr_scale(
    *,
    mode: str,
    initial_lr: float,
    anchor_lr: float,
) -> float:
    """Return the fixed coefficient that maps alpha to a group LR."""
    if mode not in ABSOLUTE_GROUP_SCALING_MODES:
        raise ValueError(f"unknown absolute group scaling mode: {mode}")
    if not all(math.isfinite(value) and value > 0 for value in (initial_lr, anchor_lr)):
        raise ValueError("absolute group scaling LRs must be finite and positive")
    return 1.0 if mode == "uniform" else initial_lr / anchor_lr


@dataclass
class NanochatMuonControlStats:
    step: int
    alpha: float
    alpha_next: float
    alpha_update_factor: float
    rho: float
    rho_clipped: float
    rho_ema: float | None
    rho_control: float
    loss_before: float
    loss_after: float
    actual_decrease: float
    predicted_decrease: float
    predicted_decrease_safe: float
    predicted_was_floored: bool
    rho_was_clipped: bool
    trust_region_expanded: bool
    trust_good_count: int
    factor_applied: float
    control_error: float = 0.0
    p_term: float = 0.0
    i_term: float = 0.0
    d_term: float = 0.0
    control_log_factor: float = 0.0
    integral_state: float = 0.0
    derivative_state: float = 0.0
    alignment_c: float | None = None
    alignment_penalty_term: float = 0.0
    alignment_allows_trust_expansion: bool = True
    alignment_bad_step: bool = False
    integral_accumulation_frozen: bool = False
    skipped_reason: str | None = None
    feedback_actual_decrease: float | None = None
    feedback_predicted_decrease: float | None = None
    rho_star: float = 0.0
    feedback_observation_valid: bool = True
    feedback_invalid_reason: str | None = None
    startup_active: bool = False
    startup_alpha_reference: float | None = None
    startup_log_term: float = 0.0
    startup_emergency: bool = False
    startup_monotone_clamped: bool = False
    startup_weight: float = 0.0
    kp_applied: float = 0.0
    factor_max_applied: float = 0.0
    action_policy: str = "legacy"
    phase_start_weight: float = 0.0
    phase_cruise_weight: float = 0.0
    phase_late_weight: float = 0.0
    phase_start_action: float = 0.0
    phase_cruise_action: float = 0.0
    phase_late_action: float = 0.0
    phase_cruise_deadband_active: bool = False
    phase_late_exponent: float = 1.0
    recovery_enabled: bool = False
    recovery_terminal_ratio: float = 1.0
    recovery_exponent: float = 4.0
    recovery_alpha_peak: float | None = None
    recovery_peak_frozen: bool = False
    recovery_late_phase_raw: float = 0.0
    recovery_late_phase_monotone: float = 0.0
    recovery_progress: float = 0.0
    recovery_alpha_uncapped: float | None = None
    recovery_alpha_cap: float | None = None
    recovery_cap_binding: bool = False
    recovery_cap_binding_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class NanochatMuonController:
    """Scalar alpha controller for nanochat Muon groups.

    Variants are named by controller family and rho signal:

    - controlled_muon_raw / ema / ema_trust: P controller
    - controlled_muon_pi_raw / pi_ema / pi_ema_trust: PI controller
    - controlled_muon_pid_raw / pid_ema / pid_ema_trust: PID controller

    All variants update a single alpha by exponentiating a bounded log-factor.
    """

    VALID_VARIANTS = {
        "controlled_muon_raw",
        "controlled_muon_ema",
        "controlled_muon_ema_trust",
        "controlled_muon_pi_raw",
        "controlled_muon_pi_ema",
        "controlled_muon_pi_ema_trust",
        "controlled_muon_pid_raw",
        "controlled_muon_pid_ema",
        "controlled_muon_pid_ema_trust",
    }

    def __init__(
        self,
        *,
        variant: str,
        alpha_init: float,
        alpha_min: float,
        alpha_max: float,
        rho_star: float = 0.7,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        rho_beta: float = 0.9,
        integral_beta: float = 0.95,
        integral_clip: float = 10.0,
        derivative_beta: float = 0.0,
        factor_min: float = 0.9,
        factor_max: float = 1.1,
        rho_clip_min: float = -1.0,
        rho_clip_max: float = 3.0,
        predicted_eps: float = 1e-12,
        predicted_floor_scale: float = 1e-12,
        trust_region_rho_threshold: float = 0.9,
        trust_region_alpha_threshold: float = 1e-4,
        trust_region_expand_factor: float = 1.5,
        trust_region_max_factor: float = 1.5,
        trust_region_patience: int = 2,
        alignment_aware: bool = False,
        alignment_c_min: float = 0.02,
        alignment_c_bad: float = -0.01,
        alignment_penalty: float = 0.10,
        alignment_bad_step_shrink: float = 0.5,
        alignment_max_log_alpha_change: float = 0.05,
        alignment_eps: float = 1e-12,
        startup_alpha_reference_ratio: float = 1.0,
        startup_alpha_reference_gain: float = 0.5,
        startup_one_sided_safety: bool = False,
        startup_monotone: bool = False,
        startup_emergency_rho: float = -0.25,
        startup_kp: float | None = None,
        startup_factor_max: float | None = None,
        action_policy: str = "legacy",
        phase_hold_cruise_policy: str = "hold",
        phase_hold_start_rho: float = 0.25,
        phase_hold_cruise_rho: float = 0.55,
        phase_hold_cruise_kp: float = 0.0,
        phase_hold_cruise_deadband: float = 0.05,
        phase_hold_late_rho: float = 0.90,
        phase_hold_late_kp: float = 0.03,
        phase_hold_late_exponent: float = 2.0,
        recovery_terminal_ratio: float = 1.0,
        recovery_exponent: float = 4.0,
    ) -> None:
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown controlled Muon variant: {variant}")
        if alpha_init <= 0 or alpha_min <= 0 or alpha_max <= 0 or alpha_min > alpha_max:
            raise ValueError("alpha bounds must satisfy 0 < alpha_min <= alpha_max")
        if not 0 <= rho_beta < 1:
            raise ValueError("rho_beta must be in [0, 1)")
        if not 0 <= integral_beta < 1:
            raise ValueError("integral_beta must be in [0, 1)")
        if not 0 <= derivative_beta < 1:
            raise ValueError("derivative_beta must be in [0, 1)")
        if integral_clip <= 0:
            raise ValueError("integral_clip must be positive")
        if factor_min <= 0 or factor_max <= 0 or factor_min > factor_max:
            raise ValueError("factor bounds must satisfy 0 < factor_min <= factor_max")
        if rho_clip_min >= rho_clip_max:
            raise ValueError("rho_clip_min must be smaller than rho_clip_max")
        if trust_region_patience < 1:
            raise ValueError("trust_region_patience must be positive")
        if not math.isfinite(alignment_c_min) or not math.isfinite(alignment_c_bad):
            raise ValueError("alignment thresholds must be finite")
        if alignment_c_bad > alignment_c_min:
            raise ValueError("alignment_c_bad must be <= alignment_c_min")
        if not math.isfinite(alignment_penalty) or alignment_penalty < 0:
            raise ValueError("alignment_penalty must be finite and non-negative")
        if not math.isfinite(alignment_bad_step_shrink) or not 0 < alignment_bad_step_shrink < 1:
            raise ValueError("alignment_bad_step_shrink must be in (0, 1)")
        if not math.isfinite(alignment_max_log_alpha_change) or alignment_max_log_alpha_change <= 0:
            raise ValueError("alignment_max_log_alpha_change must be finite and positive")
        if not math.isfinite(alignment_eps) or alignment_eps <= 0:
            raise ValueError("alignment_eps must be finite and positive")
        if not math.isfinite(startup_alpha_reference_ratio) or startup_alpha_reference_ratio < 1.0:
            raise ValueError("startup alpha reference ratio must be finite and at least 1")
        if not math.isfinite(startup_alpha_reference_gain) or startup_alpha_reference_gain <= 0:
            raise ValueError("startup alpha reference gain must be finite and positive")
        if not math.isfinite(startup_emergency_rho):
            raise ValueError("startup emergency rho must be finite")
        startup_kp = kp if startup_kp is None else float(startup_kp)
        startup_factor_max = factor_max if startup_factor_max is None else float(startup_factor_max)
        if not math.isfinite(startup_kp) or startup_kp < 0.0:
            raise ValueError("startup kp must be finite and non-negative")
        if not math.isfinite(startup_factor_max) or startup_factor_max < factor_min:
            raise ValueError("startup factor max must be finite and at least factor_min")
        if action_policy not in CONTROL_ACTION_POLICIES:
            raise ValueError(f"unknown control action policy: {action_policy}")
        phase_hold_enabled = action_policy in {"phase_hold", "phase_hold_recovery"}
        if phase_hold_enabled:
            if phase_hold_cruise_policy not in PHASE_HOLD_CRUISE_POLICIES:
                raise ValueError(f"unknown phase-hold cruise policy: {phase_hold_cruise_policy}")
            phase_values = (
                phase_hold_start_rho,
                phase_hold_cruise_rho,
                phase_hold_cruise_kp,
                phase_hold_cruise_deadband,
                phase_hold_late_rho,
                phase_hold_late_kp,
                phase_hold_late_exponent,
            )
            if not all(math.isfinite(float(value)) for value in phase_values):
                raise ValueError("phase-hold parameters must be finite")
            if not phase_hold_start_rho < phase_hold_cruise_rho < phase_hold_late_rho:
                raise ValueError("phase-hold rho targets must satisfy start < cruise < late")
            if phase_hold_cruise_kp < 0.0 or phase_hold_late_kp < 0.0:
                raise ValueError("phase-hold gains must be non-negative")
            if phase_hold_cruise_deadband < 0.0:
                raise ValueError("phase-hold cruise deadband must be non-negative")
            if phase_hold_late_exponent < 1.0:
                raise ValueError("phase-hold late exponent must be at least 1")
            if phase_hold_cruise_policy == "hold" and phase_hold_cruise_kp != 0.0:
                raise ValueError("phase-hold exact cruise requires zero cruise gain")
            if phase_hold_cruise_policy == "rho_deadband" and phase_hold_cruise_kp <= 0.0:
                raise ValueError("phase-hold rho deadband requires positive cruise gain")
        if action_policy == "phase_hold_recovery":
            if not math.isfinite(recovery_terminal_ratio) or not 0.0 < recovery_terminal_ratio <= 1.0:
                raise ValueError("recovery terminal ratio must be finite and in (0, 1]")
            if not math.isfinite(recovery_exponent) or recovery_exponent < 1.0:
                raise ValueError("recovery exponent must be finite and at least 1")

        self.variant = variant
        self.controller_family = self._parse_family(variant)
        if self.controller_family in {"pi", "pid"} and ki <= 0:
            raise ValueError("PI/PID controlled Muon variants require --control-ki > 0")
        if self.controller_family == "pid" and kd <= 0:
            raise ValueError("PID controlled Muon variants require --control-kd > 0")

        self.alpha = _clip(float(alpha_init), float(alpha_min), float(alpha_max))
        self.log_alpha = math.log(self.alpha)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.rho_star = float(rho_star)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.rho_beta = float(rho_beta)
        self.integral_beta = float(integral_beta)
        self.integral_clip = float(integral_clip)
        self.derivative_beta = float(derivative_beta)
        self.factor_min = float(factor_min)
        self.factor_max = float(factor_max)
        self.rho_clip_min = float(rho_clip_min)
        self.rho_clip_max = float(rho_clip_max)
        self.predicted_eps = float(predicted_eps)
        self.predicted_floor_scale = float(predicted_floor_scale)
        self.trust_region_rho_threshold = float(trust_region_rho_threshold)
        self.trust_region_alpha_threshold = float(trust_region_alpha_threshold)
        self.trust_region_expand_factor = float(trust_region_expand_factor)
        self.trust_region_max_factor = float(trust_region_max_factor)
        self.trust_region_patience = int(trust_region_patience)
        self.alignment_aware = bool(alignment_aware)
        self.alignment_c_min = float(alignment_c_min)
        self.alignment_c_bad = float(alignment_c_bad)
        self.alignment_penalty = float(alignment_penalty)
        self.alignment_bad_step_shrink = float(alignment_bad_step_shrink)
        self.alignment_max_log_alpha_change = float(alignment_max_log_alpha_change)
        self.alignment_eps = float(alignment_eps)
        self.alpha_init = self.alpha
        self.startup_alpha_reference_ratio = float(startup_alpha_reference_ratio)
        self.startup_alpha_reference_gain = float(startup_alpha_reference_gain)
        self.startup_one_sided_safety = bool(startup_one_sided_safety)
        self.startup_monotone = bool(startup_monotone)
        self.startup_emergency_rho = float(startup_emergency_rho)
        self.startup_kp = startup_kp
        self.startup_factor_max = startup_factor_max
        self.action_policy = action_policy
        self.phase_hold_cruise_policy = phase_hold_cruise_policy
        self.phase_hold_start_rho = float(phase_hold_start_rho)
        self.phase_hold_cruise_rho = float(phase_hold_cruise_rho)
        self.phase_hold_cruise_kp = float(phase_hold_cruise_kp)
        self.phase_hold_cruise_deadband = float(phase_hold_cruise_deadband)
        self.phase_hold_late_rho = float(phase_hold_late_rho)
        self.phase_hold_late_kp = float(phase_hold_late_kp)
        self.phase_hold_late_exponent = float(phase_hold_late_exponent)
        self.recovery_terminal_ratio = float(recovery_terminal_ratio)
        self.recovery_exponent = float(recovery_exponent)

        self.use_rho_ema = variant.endswith("_ema") or variant.endswith("_ema_trust")
        self.use_trust_region = variant.endswith("_ema_trust")
        self.rho_ema: float | None = None
        self.integral_state = 0.0
        self.derivative_state = 0.0
        self.prev_error: float | None = None
        self.trust_good_count = 0
        self.num_updates = 0
        self.last_stats: NanochatMuonControlStats | None = None
        self.recovery_alpha_peak = self.alpha
        self.recovery_peak_frozen = False
        self.recovery_max_late_phase = 0.0
        self.recovery_cap = self.alpha
        self.recovery_cap_binding_count = 0

    @staticmethod
    def _parse_family(variant: str) -> str:
        if variant.startswith("controlled_muon_pid_"):
            return "pid"
        if variant.startswith("controlled_muon_pi_"):
            return "pi"
        return "p"

    def state_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "controller_family": self.controller_family,
            "alpha": self.alpha,
            "log_alpha": self.log_alpha,
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "rho_star": self.rho_star,
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "rho_beta": self.rho_beta,
            "integral_beta": self.integral_beta,
            "integral_clip": self.integral_clip,
            "derivative_beta": self.derivative_beta,
            "factor_min": self.factor_min,
            "factor_max": self.factor_max,
            "rho_clip_min": self.rho_clip_min,
            "rho_clip_max": self.rho_clip_max,
            "predicted_eps": self.predicted_eps,
            "predicted_floor_scale": self.predicted_floor_scale,
            "trust_region_rho_threshold": self.trust_region_rho_threshold,
            "trust_region_alpha_threshold": self.trust_region_alpha_threshold,
            "trust_region_expand_factor": self.trust_region_expand_factor,
            "trust_region_max_factor": self.trust_region_max_factor,
            "trust_region_patience": self.trust_region_patience,
            "alignment_aware": self.alignment_aware,
            "alignment_c_min": self.alignment_c_min,
            "alignment_c_bad": self.alignment_c_bad,
            "alignment_penalty": self.alignment_penalty,
            "alignment_bad_step_shrink": self.alignment_bad_step_shrink,
            "alignment_max_log_alpha_change": self.alignment_max_log_alpha_change,
            "alignment_eps": self.alignment_eps,
            "alpha_init": self.alpha_init,
            "startup_alpha_reference_ratio": self.startup_alpha_reference_ratio,
            "startup_alpha_reference_gain": self.startup_alpha_reference_gain,
            "startup_one_sided_safety": self.startup_one_sided_safety,
            "startup_monotone": self.startup_monotone,
            "startup_emergency_rho": self.startup_emergency_rho,
            "startup_kp": self.startup_kp,
            "startup_factor_max": self.startup_factor_max,
            "action_policy": self.action_policy,
            "phase_hold_cruise_policy": self.phase_hold_cruise_policy,
            "phase_hold_start_rho": self.phase_hold_start_rho,
            "phase_hold_cruise_rho": self.phase_hold_cruise_rho,
            "phase_hold_cruise_kp": self.phase_hold_cruise_kp,
            "phase_hold_cruise_deadband": self.phase_hold_cruise_deadband,
            "phase_hold_late_rho": self.phase_hold_late_rho,
            "phase_hold_late_kp": self.phase_hold_late_kp,
            "phase_hold_late_exponent": self.phase_hold_late_exponent,
            "recovery_terminal_ratio": self.recovery_terminal_ratio,
            "recovery_exponent": self.recovery_exponent,
            "recovery_alpha_peak": self.recovery_alpha_peak,
            "recovery_peak_frozen": self.recovery_peak_frozen,
            "recovery_max_late_phase": self.recovery_max_late_phase,
            "recovery_cap": self.recovery_cap,
            "recovery_cap_binding_count": self.recovery_cap_binding_count,
            "rho_ema": self.rho_ema,
            "integral_state": self.integral_state,
            "derivative_state": self.derivative_state,
            "prev_error": self.prev_error,
            "trust_good_count": self.trust_good_count,
            "num_updates": self.num_updates,
            "last_stats": None if self.last_stats is None else self.last_stats.as_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_variant = state.get("variant", self.variant)
        if saved_variant != self.variant:
            raise ValueError(f"cannot load {saved_variant} controller state into {self.variant}")
        saved_action_policy = state.get("action_policy", "legacy")
        if saved_action_policy != self.action_policy:
            raise ValueError(
                f"cannot load {saved_action_policy} action policy with {self.action_policy} configured"
            )
        saved_cruise_policy = state.get("phase_hold_cruise_policy", "hold")
        phase_hold_enabled = self.action_policy in {"phase_hold", "phase_hold_recovery"}
        if phase_hold_enabled and saved_cruise_policy != self.phase_hold_cruise_policy:
            raise ValueError("cannot load controller state with different phase-hold cruise policy")
        for name in (
            "phase_hold_start_rho",
            "phase_hold_cruise_rho",
            "phase_hold_cruise_kp",
            "phase_hold_cruise_deadband",
            "phase_hold_late_rho",
            "phase_hold_late_kp",
            "phase_hold_late_exponent",
        ):
            if phase_hold_enabled and not math.isclose(
                float(state[name]), getattr(self, name), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"phase-hold controller configuration mismatch: {name}")
        if self.action_policy == "phase_hold_recovery":
            for name in ("recovery_terminal_ratio", "recovery_exponent"):
                if not math.isclose(
                    float(state[name]), getattr(self, name), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise ValueError(f"recovery controller configuration mismatch: {name}")

        for name in (
            "alpha_min",
            "alpha_max",
            "rho_star",
            "kp",
            "ki",
            "kd",
            "rho_beta",
            "integral_beta",
            "integral_clip",
            "derivative_beta",
            "factor_min",
            "factor_max",
            "rho_clip_min",
            "rho_clip_max",
            "predicted_eps",
            "predicted_floor_scale",
            "trust_region_rho_threshold",
            "trust_region_alpha_threshold",
            "trust_region_expand_factor",
            "trust_region_max_factor",
            "alignment_c_min",
            "alignment_c_bad",
            "alignment_penalty",
            "alignment_bad_step_shrink",
            "alignment_max_log_alpha_change",
            "alignment_eps",
            "alpha_init",
            "startup_alpha_reference_ratio",
            "startup_alpha_reference_gain",
            "startup_emergency_rho",
            "startup_kp",
            "startup_factor_max",
        ):
            if name in state:
                setattr(self, name, float(state[name]))
        if "trust_region_patience" in state:
            self.trust_region_patience = int(state["trust_region_patience"])
        saved_alignment_aware = bool(state.get("alignment_aware", self.alignment_aware))
        if saved_alignment_aware != self.alignment_aware:
            raise ValueError("cannot load controller state with different alignment-aware semantics")
        for name in ("startup_one_sided_safety", "startup_monotone"):
            saved = bool(state.get(name, getattr(self, name)))
            if saved != getattr(self, name):
                raise ValueError(f"cannot load controller state with different {name} semantics")

        self.controller_family = self._parse_family(self.variant)
        self.alpha = _clip(float(state["alpha"]), self.alpha_min, self.alpha_max)
        self.log_alpha = float(state.get("log_alpha", math.log(self.alpha)))
        if not math.isfinite(self.log_alpha) or not math.isclose(math.exp(self.log_alpha), self.alpha, rel_tol=1e-6, abs_tol=1e-12):
            self.log_alpha = math.log(self.alpha)
        self.rho_ema = None if state.get("rho_ema") is None else float(state["rho_ema"])
        self.integral_state = float(state.get("integral_state", 0.0))
        self.derivative_state = float(state.get("derivative_state", 0.0))
        self.prev_error = None if state.get("prev_error") is None else float(state["prev_error"])
        self.trust_good_count = int(state.get("trust_good_count", 0))
        self.num_updates = int(state.get("num_updates", 0))
        if self.action_policy == "phase_hold_recovery":
            self.recovery_alpha_peak = _clip(
                float(state["recovery_alpha_peak"]), self.alpha_min, self.alpha_max
            )
            self.recovery_peak_frozen = bool(state["recovery_peak_frozen"])
            self.recovery_max_late_phase = _clip(
                float(state["recovery_max_late_phase"]), 0.0, 1.0
            )
            self.recovery_cap = _clip(
                float(state["recovery_cap"]), self.alpha_min, self.alpha_max
            )
            self.recovery_cap_binding_count = int(state["recovery_cap_binding_count"])
        self.use_rho_ema = self.variant.endswith("_ema") or self.variant.endswith("_ema_trust")
        self.use_trust_region = self.variant.endswith("_ema_trust")
        last_stats = state.get("last_stats")
        if last_stats is not None:
            allowed = set(NanochatMuonControlStats.__dataclass_fields__)
            last_stats = {k: v for k, v in last_stats.items() if k in allowed}
        self.last_stats = None if last_stats is None else NanochatMuonControlStats(**last_stats)

    def set_alpha(self, alpha: float) -> None:
        """Set controller alpha while preserving the configured hard bounds."""
        alpha_f = float(alpha)
        if not math.isfinite(alpha_f) or alpha_f <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = _clip(alpha_f, self.alpha_min, self.alpha_max)
        self.log_alpha = math.log(self.alpha)

    def _compute_pid_terms(
        self,
        error: float,
        valid: bool,
        freeze_positive_integral: bool = False,
        kp_override: float | None = None,
    ) -> tuple[float, float, float, float, float]:
        if not valid:
            self.integral_state = 0.0
            self.derivative_state = 0.0
            self.prev_error = None
            return 0.0, 0.0, 0.0, 0.0, 0.0

        if self.controller_family in {"pi", "pid"}:
            if not (freeze_positive_integral and error > 0):
                self.integral_state = self.integral_beta * self.integral_state + error
                self.integral_state = _clip(
                    self.integral_state, -self.integral_clip, self.integral_clip
                )
        else:
            self.integral_state = 0.0

        raw_derivative = 0.0 if self.prev_error is None else error - self.prev_error
        if self.controller_family == "pid":
            self.derivative_state = self.derivative_beta * self.derivative_state + (1.0 - self.derivative_beta) * raw_derivative
        else:
            self.derivative_state = 0.0
        self.prev_error = error
        kp_applied = self.kp if kp_override is None else float(kp_override)
        p_term = kp_applied * error
        i_term = self.ki * self.integral_state if self.controller_family in {"pi", "pid"} else 0.0
        d_term = self.kd * self.derivative_state if self.controller_family == "pid" else 0.0
        log_factor = p_term + i_term + d_term
        return p_term, i_term, d_term, log_factor, self.derivative_state

    def update(
        self,
        *,
        step: int,
        loss_before: float,
        loss_after: float,
        predicted_decrease: float,
        grad_norm: float | None = None,
        update_norm: float | None = None,
        feedback_actual_decrease: float | None = None,
        feedback_observation_valid: bool = True,
        feedback_invalid_reason: str | None = None,
        startup_active: bool = False,
        startup_weight: float = 0.0,
        late_phase: float = 0.0,
        freeze_positive_integral: bool = False,
        rho_star_override: float | None = None,
    ) -> NanochatMuonControlStats:
        loss_before_f = float(loss_before)
        loss_after_f = float(loss_after)
        actual_decrease = loss_before_f - loss_after_f
        actual_for_control = (
            actual_decrease
            if feedback_actual_decrease is None
            else float(feedback_actual_decrease)
        )
        return self.update_from_observation(
            step=step,
            loss_before=loss_before_f,
            loss_after=loss_after_f,
            actual_decrease_total=actual_decrease,
            actual_decrease_for_control=actual_for_control,
            predicted_decrease_for_control=predicted_decrease,
            grad_norm=grad_norm,
            update_norm=update_norm,
            feedback_observation_valid=feedback_observation_valid,
            feedback_invalid_reason=feedback_invalid_reason,
            startup_active=startup_active,
            startup_weight=startup_weight,
            late_phase=late_phase,
            freeze_positive_integral=freeze_positive_integral,
            rho_star_override=rho_star_override,
        )

    def update_from_observation(
        self,
        *,
        step: int,
        loss_before: float,
        loss_after: float,
        actual_decrease_total: float,
        actual_decrease_for_control: float,
        predicted_decrease_for_control: float,
        grad_norm: float | None = None,
        update_norm: float | None = None,
        feedback_observation_valid: bool = True,
        feedback_invalid_reason: str | None = None,
        startup_active: bool = False,
        startup_weight: float = 0.0,
        late_phase: float = 0.0,
        freeze_positive_integral: bool = False,
        rho_star_override: float | None = None,
    ) -> NanochatMuonControlStats:
        """Update alpha from an explicitly selected feedback observation."""
        alpha_used = self.alpha
        loss_before_f = float(loss_before)
        loss_after_f = float(loss_after)
        actual_decrease = float(actual_decrease_total)
        feedback_actual_decrease = float(actual_decrease_for_control)
        predicted_decrease = float(predicted_decrease_for_control)
        floor_base = abs(loss_before_f) if math.isfinite(loss_before_f) else 1.0
        floor = max(self.predicted_eps, floor_base * self.predicted_floor_scale)
        predicted_is_valid = math.isfinite(predicted_decrease) and predicted_decrease > floor
        predicted_safe = float(predicted_decrease) if predicted_is_valid else floor
        predicted_was_floored = not predicted_is_valid
        measurement_is_valid = math.isfinite(feedback_actual_decrease)

        if measurement_is_valid:
            rho = feedback_actual_decrease / predicted_safe
            rho_is_valid = math.isfinite(rho)
        else:
            rho = self.rho_clip_min
            rho_is_valid = False

        if rho_is_valid:
            rho_clipped = _clip(rho, self.rho_clip_min, self.rho_clip_max)
            rho_was_clipped = rho_clipped != rho
        else:
            rho_clipped = self.rho_clip_min
            rho_was_clipped = True

        skipped_reason = None
        if not predicted_is_valid:
            skipped_reason = "predicted_decrease_not_positive"
        if not rho_is_valid:
            skipped_reason = "nonfinite_loss_or_rho"

        alignment_c: float | None = None
        alignment_bad_step = False
        alignment_allows_trust_expansion = True
        if self.alignment_aware:
            norms_are_valid = (
                grad_norm is not None
                and update_norm is not None
                and math.isfinite(float(grad_norm))
                and math.isfinite(float(update_norm))
                and float(grad_norm) >= 0.0
                and float(update_norm) >= 0.0
            )
            if norms_are_valid:
                denominator = float(grad_norm) * float(update_norm) + self.alignment_eps
                alignment_c = float(predicted_decrease) / denominator
            else:
                alignment_c = float("nan")
            alignment_is_finite = math.isfinite(alignment_c)
            alignment_bad_step = (
                not alignment_is_finite
                or alignment_c < self.alignment_c_bad
                or not measurement_is_valid
                or feedback_actual_decrease <= 0.0
            )
            alignment_allows_trust_expansion = (
                alignment_is_finite and alignment_c >= self.alignment_c_min
            )
            if alignment_bad_step:
                if not alignment_is_finite:
                    skipped_reason = "nonfinite_alignment"
                elif alignment_c < self.alignment_c_bad:
                    skipped_reason = "alignment_below_bad_threshold"
                elif feedback_actual_decrease <= 0.0:
                    skipped_reason = "nonpositive_actual_decrease"

        feedback_observation_valid = bool(feedback_observation_valid)
        measurement_for_control_is_valid = (
            feedback_observation_valid
            and predicted_is_valid
            and rho_is_valid
            and not alignment_bad_step
        )
        if not feedback_observation_valid and feedback_invalid_reason is None:
            feedback_invalid_reason = "feedback_observation_invalid"
        if not feedback_observation_valid:
            skipped_reason = feedback_invalid_reason

        if self.use_rho_ema:
            if measurement_for_control_is_valid:
                if self.rho_ema is None:
                    self.rho_ema = rho_clipped
                else:
                    self.rho_ema = self.rho_beta * self.rho_ema + (1.0 - self.rho_beta) * rho_clipped
            rho_control = self.rho_ema if self.rho_ema is not None else rho_clipped
        else:
            rho_control = rho_clipped
        rho_star = self.rho_star if rho_star_override is None else float(rho_star_override)
        if not math.isfinite(rho_star):
            raise ValueError("rho_star_override must be finite")
        startup_weight = _clip(float(startup_weight), 0.0, 1.0)
        late_phase = _clip(float(late_phase), 0.0, 1.0)
        startup_phase = 1.0 - startup_weight
        late_authority = late_phase ** self.phase_hold_late_exponent
        phase_start_weight = startup_weight
        phase_cruise_weight = startup_phase * (1.0 - late_authority)
        phase_late_weight = startup_phase * late_authority
        kp_applied = self.kp + startup_weight * (self.startup_kp - self.kp)
        factor_max_applied = math.exp(
            math.log(self.factor_max)
            + startup_weight * (math.log(self.startup_factor_max) - math.log(self.factor_max))
        )
        phase_start_action = 0.0
        phase_cruise_action = 0.0
        phase_late_action = 0.0
        phase_cruise_deadband_active = False
        phase_hold_startup_emergency = (
            measurement_for_control_is_valid
            and rho_control <= self.startup_emergency_rho
        )
        phase_hold_enabled = self.action_policy in {"phase_hold", "phase_hold_recovery"}
        if phase_hold_enabled:
            rho_star = (
                phase_start_weight * self.phase_hold_start_rho
                + phase_cruise_weight * self.phase_hold_cruise_rho
                + phase_late_weight * self.phase_hold_late_rho
            )
            error = rho_control - rho_star
            if measurement_for_control_is_valid:
                raw_start_action = self.startup_kp * (
                    rho_control - self.phase_hold_start_rho
                )
                phase_start_action = (
                    raw_start_action
                    if phase_hold_startup_emergency
                    else max(raw_start_action, 0.0)
                )
                cruise_error = rho_control - self.phase_hold_cruise_rho
                phase_cruise_deadband_active = (
                    abs(cruise_error) <= self.phase_hold_cruise_deadband
                )
                if self.phase_hold_cruise_policy == "rho_deadband":
                    phase_cruise_action = self.phase_hold_cruise_kp * _deadband(
                        cruise_error, self.phase_hold_cruise_deadband
                    )
                phase_late_action = min(
                    self.phase_hold_late_kp
                    * (rho_control - self.phase_hold_late_rho),
                    0.0,
                )
            control_log_factor = (
                phase_start_weight * phase_start_action
                + phase_cruise_weight * phase_cruise_action
                + phase_late_weight * phase_late_action
            )
            p_term = control_log_factor
            i_term = 0.0
            d_term = 0.0
            self.integral_state = 0.0
            self.derivative_state = 0.0
            self.prev_error = error if measurement_for_control_is_valid else None
            derivative_state = self.derivative_state
            kp_applied = (
                phase_start_weight * self.startup_kp
                + phase_cruise_weight * self.phase_hold_cruise_kp
                + phase_late_weight * self.phase_hold_late_kp
            )
        else:
            error = rho_control - rho_star
            p_term, i_term, d_term, control_log_factor, derivative_state = self._compute_pid_terms(
                error,
                measurement_for_control_is_valid,
                freeze_positive_integral=freeze_positive_integral,
                kp_override=kp_applied,
            )
        alignment_penalty_term = 0.0
        if self.alignment_aware and measurement_for_control_is_valid and alignment_c is not None:
            alignment_penalty_term = self.alignment_penalty * max(
                0.0, self.alignment_c_min - alignment_c
            )
            control_log_factor -= alignment_penalty_term
            control_log_factor = _clip(
                control_log_factor,
                -self.alignment_max_log_alpha_change,
                self.alignment_max_log_alpha_change,
            )
        startup_alpha_reference = None
        startup_log_term = 0.0
        startup_emergency = alignment_bad_step or (
            phase_hold_enabled and phase_hold_startup_emergency
        )
        if startup_active and self.startup_alpha_reference_ratio > 1.0:
            startup_alpha_reference = _clip(
                self.alpha_init * self.startup_alpha_reference_ratio,
                self.alpha_min,
                self.alpha_max,
            )
            startup_log_term = self.startup_alpha_reference_gain * math.log(
                startup_alpha_reference / alpha_used
            )
            control_log_factor += startup_log_term
            startup_emergency = startup_emergency or (
                measurement_for_control_is_valid
                and rho_control <= self.startup_emergency_rho
            )
            if self.startup_one_sided_safety and not startup_emergency:
                control_log_factor = max(0.0, control_log_factor)
                if startup_alpha_reference is not None and alpha_used < startup_alpha_reference:
                    control_log_factor = max(control_log_factor, math.log(self.factor_max))
        raw_factor = math.exp(control_log_factor)
        factor = _clip(raw_factor, self.factor_min, factor_max_applied)
        if alignment_bad_step:
            factor = self.alignment_bad_step_shrink
        elif not measurement_for_control_is_valid and not (
            startup_active and startup_log_term > 0.0
        ):
            factor = min(factor, 1.0)

        if (
            measurement_for_control_is_valid
            and alignment_allows_trust_expansion
            and rho_control >= self.trust_region_rho_threshold
        ):
            self.trust_good_count += 1
        else:
            self.trust_good_count = 0

        trust_region_expanded = (
            self.use_trust_region
            and measurement_for_control_is_valid
            and alignment_allows_trust_expansion
            and rho_control >= self.trust_region_rho_threshold
            and alpha_used <= self.trust_region_alpha_threshold
            and self.trust_good_count >= self.trust_region_patience
        )
        if trust_region_expanded:
            factor = max(factor, self.trust_region_expand_factor)
            factor = min(factor, self.trust_region_max_factor)

        hard_factor_max = factor_max_applied
        if self.use_trust_region:
            hard_factor_max = max(hard_factor_max, self.trust_region_max_factor)
        factor_lower = self.factor_min if measurement_for_control_is_valid else min(self.factor_min, 1.0)
        if alignment_bad_step:
            factor_lower = min(factor_lower, self.alignment_bad_step_shrink)
        factor = _clip(factor, factor_lower, hard_factor_max)

        alpha_next = _clip(alpha_used * factor, self.alpha_min, self.alpha_max)
        startup_monotone_clamped = False
        if (
            startup_active
            and self.startup_monotone
            and not startup_emergency
            and alpha_next < alpha_used
        ):
            alpha_next = alpha_used
            startup_monotone_clamped = True
        recovery_enabled = self.action_policy == "phase_hold_recovery"
        recovery_alpha_uncapped = alpha_next
        recovery_late_phase_raw = late_phase
        recovery_progress = 0.0
        recovery_cap_binding = False
        if recovery_enabled:
            if not self.recovery_peak_frozen and late_phase == 0.0:
                self.recovery_alpha_peak = max(self.recovery_alpha_peak, alpha_next)
                self.recovery_cap = self.recovery_alpha_peak
            else:
                if not self.recovery_peak_frozen:
                    self.recovery_peak_frozen = True
                self.recovery_max_late_phase = max(
                    self.recovery_max_late_phase, late_phase
                )
                recovery_progress = (
                    self.recovery_max_late_phase ** self.recovery_exponent
                )
                proposed_cap = self.recovery_alpha_peak * (
                    self.recovery_terminal_ratio ** recovery_progress
                )
                self.recovery_cap = min(
                    self.recovery_cap,
                    _clip(proposed_cap, self.alpha_min, self.alpha_max),
                )
                alpha_next = min(alpha_next, self.recovery_cap)
                recovery_cap_binding = alpha_next < recovery_alpha_uncapped
                if recovery_cap_binding:
                    self.recovery_cap_binding_count += 1
        self.alpha = alpha_next
        self.log_alpha = math.log(alpha_next)
        self.num_updates += 1

        stats = NanochatMuonControlStats(
            step=int(step),
            alpha=alpha_used,
            alpha_next=alpha_next,
            alpha_update_factor=alpha_next / alpha_used,
            rho=rho,
            rho_clipped=rho_clipped,
            rho_ema=self.rho_ema,
            rho_control=rho_control,
            loss_before=loss_before_f,
            loss_after=loss_after_f,
            actual_decrease=actual_decrease,
            predicted_decrease=float(predicted_decrease),
            predicted_decrease_safe=predicted_safe,
            predicted_was_floored=predicted_was_floored,
            rho_was_clipped=rho_was_clipped,
            integral_accumulation_frozen=(
                measurement_for_control_is_valid
                and self.controller_family in {"pi", "pid"}
                and freeze_positive_integral
                and error > 0
            ),
            trust_region_expanded=trust_region_expanded,
            trust_good_count=self.trust_good_count,
            factor_applied=alpha_next / alpha_used,
            control_error=error,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term,
            control_log_factor=control_log_factor,
            integral_state=self.integral_state,
            derivative_state=derivative_state,
            alignment_c=alignment_c,
            alignment_penalty_term=alignment_penalty_term,
            alignment_allows_trust_expansion=alignment_allows_trust_expansion,
            alignment_bad_step=alignment_bad_step,
            skipped_reason=skipped_reason,
            feedback_actual_decrease=feedback_actual_decrease,
            feedback_predicted_decrease=predicted_decrease,
            rho_star=rho_star,
            feedback_observation_valid=feedback_observation_valid,
            feedback_invalid_reason=feedback_invalid_reason,
            startup_active=bool(startup_active),
            startup_alpha_reference=startup_alpha_reference,
            startup_log_term=startup_log_term,
            startup_emergency=startup_emergency,
            startup_monotone_clamped=startup_monotone_clamped,
            startup_weight=startup_weight,
            kp_applied=kp_applied,
            factor_max_applied=factor_max_applied,
            action_policy=self.action_policy,
            phase_start_weight=phase_start_weight,
            phase_cruise_weight=phase_cruise_weight,
            phase_late_weight=phase_late_weight,
            phase_start_action=phase_start_action,
            phase_cruise_action=phase_cruise_action,
            phase_late_action=phase_late_action,
            phase_cruise_deadband_active=phase_cruise_deadband_active,
            phase_late_exponent=self.phase_hold_late_exponent,
            recovery_enabled=recovery_enabled,
            recovery_terminal_ratio=self.recovery_terminal_ratio,
            recovery_exponent=self.recovery_exponent,
            recovery_alpha_peak=self.recovery_alpha_peak if recovery_enabled else None,
            recovery_peak_frozen=self.recovery_peak_frozen if recovery_enabled else False,
            recovery_late_phase_raw=recovery_late_phase_raw,
            recovery_late_phase_monotone=(
                self.recovery_max_late_phase if recovery_enabled else 0.0
            ),
            recovery_progress=recovery_progress,
            recovery_alpha_uncapped=recovery_alpha_uncapped,
            recovery_alpha_cap=self.recovery_cap if recovery_enabled else None,
            recovery_cap_binding=recovery_cap_binding,
            recovery_cap_binding_count=(
                self.recovery_cap_binding_count if recovery_enabled else 0
            ),
        )
        self.last_stats = stats
        return stats

    def diagnostics_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "controller_family": self.controller_family,
            "alpha": self.alpha,
            "log_alpha": self.log_alpha,
            "rho_ema": self.rho_ema,
            "integral_state": self.integral_state,
            "derivative_state": self.derivative_state,
            "prev_error": self.prev_error,
            "trust_good_count": self.trust_good_count,
            "alignment_aware": self.alignment_aware,
            "action_policy": self.action_policy,
            "num_updates": self.num_updates,
            "last_stats": None if self.last_stats is None else self.last_stats.as_dict(),
        }

class LossProgressRhoReference:
    """Monotone, horizon-free rho target driven by smoothed loss progress."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        rho_middle: float,
        rho_late: float,
        beta_fast: float = 0.90,
        beta_slow: float = 0.99,
        beta_reference: float = 0.999,
        beta_phase: float = 0.99,
        progress_ratio_high: float = 0.50,
        progress_ratio_low: float = 0.10,
        minimum_observations: int = 50,
        eps: float = 1e-12,
    ) -> None:
        values = (
            rho_middle,
            rho_late,
            beta_fast,
            beta_slow,
            beta_reference,
            beta_phase,
            progress_ratio_high,
            progress_ratio_low,
            eps,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("rho progress reference values must be finite")
        if rho_middle >= rho_late:
            raise ValueError("rho_middle must be smaller than rho_late")
        if not 0.0 <= beta_fast < beta_slow < 1.0:
            raise ValueError("rho progress betas must satisfy 0 <= fast < slow < 1")
        if not 0.0 <= beta_reference < 1.0:
            raise ValueError("beta_reference must be in [0, 1)")
        if not 0.0 <= beta_phase < 1.0:
            raise ValueError("beta_phase must be in [0, 1)")
        if not 0.0 <= progress_ratio_low < progress_ratio_high <= 1.0:
            raise ValueError(
                "progress ratio thresholds must satisfy 0 <= low < high <= 1"
            )
        if minimum_observations < 1:
            raise ValueError("minimum_observations must be positive")
        if eps <= 0.0:
            raise ValueError("eps must be positive")

        self.rho_middle = float(rho_middle)
        self.rho_late = float(rho_late)
        self.beta_fast = float(beta_fast)
        self.beta_slow = float(beta_slow)
        self.beta_reference = float(beta_reference)
        self.beta_phase = float(beta_phase)
        self.progress_ratio_high = float(progress_ratio_high)
        self.progress_ratio_low = float(progress_ratio_low)
        self.minimum_observations = int(minimum_observations)
        self.eps = float(eps)

        self.observation_count = 0
        self.loss_fast: float | None = None
        self.loss_slow: float | None = None
        self.relative_progress = 0.0
        self.progress_reference = 0.0
        self.progress_ratio = 1.0
        self.phase_candidate = 0.0
        self.phase = 0.0
        self.rho_star = self.rho_middle

    @staticmethod
    def _same_float(saved: Any, configured: float) -> bool:
        return math.isclose(
            float(saved), configured, rel_tol=1e-12, abs_tol=1e-12
        )

    def observe(self, loss: float) -> float:
        """Observe one training loss and return the current rho target."""
        loss_f = float(loss)
        if not math.isfinite(loss_f) or loss_f <= 0.0:
            raise ValueError("training loss must be finite and positive")

        if self.loss_fast is None:
            self.loss_fast = loss_f
            self.loss_slow = loss_f
        else:
            self.loss_fast = self.beta_fast * self.loss_fast + (1.0 - self.beta_fast) * loss_f
            self.loss_slow = self.beta_slow * self.loss_slow + (1.0 - self.beta_slow) * loss_f

        self.observation_count += 1
        self.relative_progress = max(
            0.0,
            self.loss_slow - self.loss_fast,
        ) / max(abs(self.loss_slow), self.eps)
        self.progress_reference = max(
            self.relative_progress,
            self.beta_reference * self.progress_reference,
        )
        if self.progress_reference > self.eps:
            self.progress_ratio = _clip(
                self.relative_progress / self.progress_reference,
                0.0,
                1.0,
            )
        else:
            self.progress_ratio = 1.0

        if self.observation_count < self.minimum_observations:
            self.phase_candidate = 0.0
        else:
            phase_linear = _clip(
                (self.progress_ratio_high - self.progress_ratio)
                / (self.progress_ratio_high - self.progress_ratio_low),
                0.0,
                1.0,
            )
            self.phase_candidate = phase_linear * phase_linear * (3.0 - 2.0 * phase_linear)
        phase_filtered = (
            self.beta_phase * self.phase
            + (1.0 - self.beta_phase) * self.phase_candidate
        )
        self.phase = max(self.phase, _clip(phase_filtered, 0.0, 1.0))
        self.rho_star = self.rho_middle + self.phase * (self.rho_late - self.rho_middle)
        return self.rho_star

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "rho_middle": self.rho_middle,
            "rho_late": self.rho_late,
            "beta_fast": self.beta_fast,
            "beta_slow": self.beta_slow,
            "beta_reference": self.beta_reference,
            "beta_phase": self.beta_phase,
            "progress_ratio_high": self.progress_ratio_high,
            "progress_ratio_low": self.progress_ratio_low,
            "minimum_observations": self.minimum_observations,
            "eps": self.eps,
            "observation_count": self.observation_count,
            "loss_fast": self.loss_fast,
            "loss_slow": self.loss_slow,
            "relative_progress": self.relative_progress,
            "progress_reference": self.progress_reference,
            "progress_ratio": self.progress_ratio,
            "phase_candidate": self.phase_candidate,
            "phase": self.phase,
            "rho_star": self.rho_star,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", 1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported rho progress reference schema")
        for name in (
            "rho_middle",
            "rho_late",
            "beta_fast",
            "beta_slow",
            "beta_reference",
            "beta_phase",
            "progress_ratio_high",
            "progress_ratio_low",
            "eps",
        ):
            if not self._same_float(state[name], getattr(self, name)):
                raise ValueError(f"rho progress reference configuration mismatch: {name}")
        if int(state["minimum_observations"]) != self.minimum_observations:
            raise ValueError("rho progress reference configuration mismatch: minimum_observations")

        self.observation_count = int(state.get("observation_count", 0))
        self.loss_fast = None if state.get("loss_fast") is None else float(state["loss_fast"])
        self.loss_slow = None if state.get("loss_slow") is None else float(state["loss_slow"])
        self.relative_progress = float(state.get("relative_progress", 0.0))
        self.progress_reference = float(state.get("progress_reference", 0.0))
        self.progress_ratio = float(state.get("progress_ratio", 1.0))
        self.phase_candidate = float(state.get("phase_candidate", 0.0))
        self.phase = _clip(float(state.get("phase", 0.0)), 0.0, 1.0)
        self.rho_star = float(state.get("rho_star", self.rho_middle))
        if not all(
            math.isfinite(value)
            for value in (
                self.relative_progress,
                self.progress_reference,
                self.progress_ratio,
                self.phase_candidate,
                self.phase,
                self.rho_star,
            )
        ):
            raise ValueError("rho progress reference state contains nonfinite values")

    def diagnostics_dict(self) -> dict[str, Any]:
        return {
            "rho_reference_mode": "loss_progress",
            "rho_star_applied": self.rho_star,
            "rho_phase_observation_count": self.observation_count,
            "rho_phase_loss_fast": self.loss_fast,
            "rho_phase_loss_slow": self.loss_slow,
            "rho_phase_relative_progress": self.relative_progress,
            "rho_phase_progress_reference": self.progress_reference,
            "rho_phase_progress_ratio": self.progress_ratio,
            "rho_phase_candidate": self.phase_candidate,
            "rho_phase": self.phase,
        }


class ThreeStageLossProgressRhoReference:
    """Horizon-free startup, cruise, and late rho reference."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        rho_start: float,
        rho_cruise: float,
        rho_late: float,
        startup_beta_fast: float = 0.70,
        startup_beta_slow: float = 0.95,
        startup_beta_reference: float = 0.995,
        startup_beta_phase: float = 0.90,
        startup_progress_ratio_high: float = 0.80,
        startup_progress_ratio_low: float = 0.40,
        startup_minimum_observations: int = 10,
        late_beta_fast: float = 0.98,
        late_beta_slow: float = 0.998,
        late_beta_reference: float = 0.9995,
        late_beta_phase: float = 0.99,
        late_progress_ratio_high: float = 0.65,
        late_progress_ratio_low: float = 0.15,
        late_minimum_observations: int = 50,
    ) -> None:
        if not all(math.isfinite(float(value)) for value in (rho_start, rho_cruise, rho_late)):
            raise ValueError("three-stage rho targets must be finite")
        if not rho_start < rho_cruise < rho_late:
            raise ValueError("three-stage rho targets must satisfy start < cruise < late")
        self.rho_start = float(rho_start)
        self.rho_cruise = float(rho_cruise)
        self.rho_late = float(rho_late)
        self.startup_reference = LossProgressRhoReference(
            rho_middle=0.0,
            rho_late=1.0,
            beta_fast=startup_beta_fast,
            beta_slow=startup_beta_slow,
            beta_reference=startup_beta_reference,
            beta_phase=startup_beta_phase,
            progress_ratio_high=startup_progress_ratio_high,
            progress_ratio_low=startup_progress_ratio_low,
            minimum_observations=startup_minimum_observations,
        )
        self.late_reference = LossProgressRhoReference(
            rho_middle=0.0,
            rho_late=1.0,
            beta_fast=late_beta_fast,
            beta_slow=late_beta_slow,
            beta_reference=late_beta_reference,
            beta_phase=late_beta_phase,
            progress_ratio_high=late_progress_ratio_high,
            progress_ratio_low=late_progress_ratio_low,
            minimum_observations=late_minimum_observations,
        )
        self.startup_phase = 0.0
        self.startup_weight = 1.0
        self.late_phase = 0.0
        self.rho_star = self.rho_start

    @property
    def observation_count(self) -> int:
        return self.late_reference.observation_count

    @property
    def loss_fast(self) -> float | None:
        return self.late_reference.loss_fast

    @property
    def loss_slow(self) -> float | None:
        return self.late_reference.loss_slow

    @property
    def relative_progress(self) -> float:
        return self.late_reference.relative_progress

    @property
    def progress_reference(self) -> float:
        return self.late_reference.progress_reference

    @property
    def progress_ratio(self) -> float:
        return self.late_reference.progress_ratio

    @property
    def phase_candidate(self) -> float:
        return self.late_reference.phase_candidate

    @property
    def phase(self) -> float:
        return self.late_phase

    def observe(self, loss: float) -> float:
        self.startup_reference.observe(loss)
        self.late_reference.observe(loss)
        self.startup_phase = self.startup_reference.phase
        self.startup_weight = 1.0 - self.startup_phase
        self.late_phase = self.late_reference.phase
        self.rho_star = (
            self.rho_start
            + self.startup_phase * (self.rho_cruise - self.rho_start)
            + self.late_phase * (self.rho_late - self.rho_cruise)
        )
        return self.rho_star

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "rho_start": self.rho_start,
            "rho_cruise": self.rho_cruise,
            "rho_late": self.rho_late,
            "startup_reference": self.startup_reference.state_dict(),
            "late_reference": self.late_reference.state_dict(),
            "startup_phase": self.startup_phase,
            "startup_weight": self.startup_weight,
            "late_phase": self.late_phase,
            "rho_star": self.rho_star,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", 1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported three-stage rho reference schema")
        for name in ("rho_start", "rho_cruise", "rho_late"):
            if not math.isclose(float(state[name]), getattr(self, name), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"three-stage rho reference configuration mismatch: {name}")
        self.startup_reference.load_state_dict(state["startup_reference"])
        self.late_reference.load_state_dict(state["late_reference"])
        self.startup_phase = _clip(float(state.get("startup_phase", self.startup_reference.phase)), 0.0, 1.0)
        self.startup_weight = _clip(float(state.get("startup_weight", 1.0 - self.startup_phase)), 0.0, 1.0)
        self.late_phase = _clip(float(state.get("late_phase", self.late_reference.phase)), 0.0, 1.0)
        self.rho_star = float(state.get("rho_star", self.rho_start))
        if not math.isfinite(self.rho_star):
            raise ValueError("three-stage rho reference state contains nonfinite rho_star")

    def diagnostics_dict(self) -> dict[str, Any]:
        diagnostics = self.late_reference.diagnostics_dict()
        diagnostics.update(
            {
                "rho_reference_mode": "loss_progress_three_stage",
                "rho_star_applied": self.rho_star,
                "rho_startup_phase": self.startup_phase,
                "rho_startup_weight": self.startup_weight,
                "rho_startup_progress_ratio": self.startup_reference.progress_ratio,
                "rho_startup_phase_candidate": self.startup_reference.phase_candidate,
                "rho_late_phase": self.late_phase,
            }
        )
        return diagnostics
