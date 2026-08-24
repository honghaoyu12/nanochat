"""Opt-in dual Muon/AdamW multiplier controller.

This module deliberately wraps the existing scalar controller instead of
modifying its behavior. The first dual implementation is multiplier-only:
the wrapped controller chooses a global multiplier and this module allocates
a bounded relative correction between Muon and AdamW groups.
"""

from __future__ import annotations

import math
from typing import Any

from nanochat.controlled_muon import NanochatMuonController


DUAL_CONTROLLER_STATE_SCHEMA_VERSION = 1


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _deadband(value: float, width: float) -> float:
    magnitude = max(0.0, abs(value) - width)
    return math.copysign(magnitude, value) if magnitude > 0.0 else 0.0


class NanochatDualActuatorController:
    """Shared global controller plus a bounded Muon/AdamW allocation state."""

    def __init__(
        self,
        *,
        global_controller: NanochatMuonController,
        allocation_kp: float = 0.02,
        allocation_deadband: float = 0.10,
        allocation_log_min: float = -0.20,
        allocation_log_max: float = 0.20,
        multiplier_min: float = 0.80,
        multiplier_max: float = 1.20,
        calibration_steps: int = 25,
        calibration_beta: float = 0.90,
        min_contribution_fraction: float = 0.05,
        allocation_period: int = 5,
        score_eps: float = 1e-12,
    ) -> None:
        if not math.isfinite(allocation_kp) or allocation_kp < 0.0:
            raise ValueError("dual allocation kp must be finite and non-negative")
        if not math.isfinite(allocation_deadband) or allocation_deadband < 0.0:
            raise ValueError("dual allocation deadband must be finite and non-negative")
        if (
            not math.isfinite(allocation_log_min)
            or not math.isfinite(allocation_log_max)
            or allocation_log_min > allocation_log_max
        ):
            raise ValueError("dual allocation log bounds must be finite and ordered")
        if (
            not math.isfinite(multiplier_min)
            or not math.isfinite(multiplier_max)
            or not 0.0 < multiplier_min <= multiplier_max
        ):
            raise ValueError("dual multiplier bounds must satisfy 0 < min <= max")
        if (
            float(global_controller.alpha_min) < float(multiplier_min)
            or float(global_controller.alpha_max) > float(multiplier_max)
        ):
            raise ValueError(
                "dual global controller alpha bounds must lie within individual multiplier bounds"
            )
        if calibration_steps < 0:
            raise ValueError("dual calibration steps must be non-negative")
        if not 0.0 <= calibration_beta < 1.0:
            raise ValueError("dual calibration beta must be in [0, 1)")
        if not 0.0 <= min_contribution_fraction <= 0.5:
            raise ValueError("dual minimum contribution fraction must be in [0, 0.5]")
        if allocation_period <= 0:
            raise ValueError("dual allocation period must be positive")
        if not math.isfinite(score_eps) or score_eps <= 0.0:
            raise ValueError("dual score epsilon must be finite and positive")

        self.global_controller = global_controller
        self.allocation_kp = float(allocation_kp)
        self.allocation_deadband = float(allocation_deadband)
        self.allocation_log_min = float(allocation_log_min)
        self.allocation_log_max = float(allocation_log_max)
        self.multiplier_min = float(multiplier_min)
        self.multiplier_max = float(multiplier_max)
        self.calibration_steps = int(calibration_steps)
        self.calibration_beta = float(calibration_beta)
        self.min_contribution_fraction = float(min_contribution_fraction)
        self.allocation_period = int(allocation_period)
        self.score_eps = float(score_eps)

        self.allocation_log_scale = 0.0
        self._muon_weight = 0.5
        self._adamw_weight = 0.5
        self._muon_multiplier = float(global_controller.alpha)
        self._adamw_multiplier = float(global_controller.alpha)
        self.muon_score_center: float | None = None
        self.adamw_score_center: float | None = None
        self.muon_score_scale: float | None = None
        self.adamw_score_scale: float | None = None
        self.calibration_count = 0
        self.last_dual_stats: dict[str, Any] = self._empty_stats()

    @property
    def alpha(self) -> float:
        return self.global_controller.alpha

    @property
    def num_updates(self) -> int:
        return self.global_controller.num_updates

    @property
    def last_stats(self):
        return self.global_controller.last_stats

    @property
    def muon_multiplier(self) -> float:
        return self._muon_multiplier

    @property
    def adamw_multiplier(self) -> float:
        return self._adamw_multiplier

    def set_alpha(self, alpha: float) -> None:
        self.global_controller.set_alpha(alpha)
        self._recompute_multipliers()

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "global_multiplier": None,
            "muon_multiplier": None,
            "adamw_multiplier": None,
            "allocation_log_scale": 0.0,
            "muon_contribution_fraction": None,
            "adamw_contribution_fraction": None,
            "muon_component_score": None,
            "adamw_component_score": None,
            "muon_component_valid": False,
            "adamw_component_valid": False,
            "allocation_update_frozen": True,
            "allocation_deadband_active": False,
            "muon_bound_hit": False,
            "adamw_bound_hit": False,
            "per_update_log_change_muon": 0.0,
            "per_update_log_change_adamw": 0.0,
            "calibration_count": 0,
            "allocation_error": 0.0,
        }

    def _recompute_multipliers(self) -> None:
        log_global = math.log(max(self.score_eps, float(self.global_controller.alpha)))
        log_muon = log_global + self._adamw_weight * self.allocation_log_scale
        log_adamw = log_global - self._muon_weight * self.allocation_log_scale
        raw_muon = math.exp(log_muon)
        raw_adamw = math.exp(log_adamw)
        self._muon_multiplier = _clip(raw_muon, self.multiplier_min, self.multiplier_max)
        self._adamw_multiplier = _clip(raw_adamw, self.multiplier_min, self.multiplier_max)

    def _update_calibration(self, name: str, score: float) -> None:
        center_name = f"{name}_score_center"
        scale_name = f"{name}_score_scale"
        center = getattr(self, center_name)
        scale = getattr(self, scale_name)
        if center is None:
            center = score
            scale = max(abs(score) * 0.1, self.score_eps)
        else:
            deviation = abs(score - center)
            beta = self.calibration_beta
            center = beta * center + (1.0 - beta) * score
            scale = beta * scale + (1.0 - beta) * max(deviation, self.score_eps)
            scale = max(scale, self.score_eps)
        setattr(self, center_name, float(center))
        setattr(self, scale_name, float(scale))

    def _component_score(
        self, predicted_decrease: float, grad_norm: float, update_norm: float
    ) -> float | None:
        values = (predicted_decrease, grad_norm, update_norm)
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0 for value in values
        ):
            return None
        denominator = float(grad_norm) * float(update_norm)
        if float(predicted_decrease) <= self.score_eps or denominator <= self.score_eps:
            return None
        score = float(predicted_decrease) / (denominator + self.score_eps)
        return score if math.isfinite(score) and score > 0.0 else None

    def _update_allocation(
        self,
        *,
        predicted_decrease_muon: float,
        predicted_decrease_adamw: float,
        muon_grad_norm: float,
        adamw_grad_norm: float,
        muon_update_norm: float,
        adamw_update_norm: float,
        feedback_observation_valid: bool,
    ) -> None:
        predicted_total = float(predicted_decrease_muon) + float(predicted_decrease_adamw)
        muon_fraction = (
            float(predicted_decrease_muon) / predicted_total
            if math.isfinite(predicted_total) and predicted_total > self.score_eps
            else None
        )
        adamw_fraction = None if muon_fraction is None else 1.0 - muon_fraction
        muon_score = self._component_score(
            predicted_decrease_muon, muon_grad_norm, muon_update_norm
        )
        adamw_score = self._component_score(
            predicted_decrease_adamw, adamw_grad_norm, adamw_update_norm
        )
        component_valid = (
            bool(feedback_observation_valid)
            and muon_fraction is not None
            and adamw_fraction is not None
            and muon_fraction >= self.min_contribution_fraction
            and adamw_fraction >= self.min_contribution_fraction
            and muon_score is not None
            and adamw_score is not None
        )
        frozen = not component_valid
        deadband_active = False
        allocation_error = 0.0
        if component_valid:
            if (
                self.muon_score_center is None
                or self.adamw_score_center is None
                or self.calibration_count < self.calibration_steps
            ):
                self._update_calibration("muon", muon_score)
                self._update_calibration("adamw", adamw_score)
                self.calibration_count += 1
                frozen = True
            else:
                muon_scale = max(float(self.muon_score_scale), self.score_eps)
                adamw_scale = max(float(self.adamw_score_scale), self.score_eps)
                muon_z = (muon_score - self.muon_score_center) / muon_scale
                adamw_z = (adamw_score - self.adamw_score_center) / adamw_scale
                allocation_error = muon_z - adamw_z
                action = _deadband(allocation_error, self.allocation_deadband)
                deadband_active = action == 0.0
                if self.num_updates % self.allocation_period == 0:
                    self.allocation_log_scale = _clip(
                        self.allocation_log_scale + self.allocation_kp * action,
                        self.allocation_log_min,
                        self.allocation_log_max,
                    )
                frozen = False
            self._muon_weight = _clip(muon_fraction, 0.0, 1.0)
            self._adamw_weight = _clip(adamw_fraction, 0.0, 1.0)

        old_muon = self._muon_multiplier
        old_adamw = self._adamw_multiplier
        self._recompute_multipliers()
        self.last_dual_stats = {
            "global_multiplier": self.global_controller.alpha,
            "muon_multiplier": self._muon_multiplier,
            "adamw_multiplier": self._adamw_multiplier,
            "allocation_log_scale": self.allocation_log_scale,
            "muon_contribution_fraction": muon_fraction,
            "adamw_contribution_fraction": adamw_fraction,
            "muon_component_score": muon_score,
            "adamw_component_score": adamw_score,
            "muon_component_valid": muon_score is not None,
            "adamw_component_valid": adamw_score is not None,
            "allocation_update_frozen": frozen,
            "allocation_deadband_active": deadband_active,
            "muon_bound_hit": math.isclose(self._muon_multiplier, self.multiplier_min) or math.isclose(self._muon_multiplier, self.multiplier_max),
            "adamw_bound_hit": math.isclose(self._adamw_multiplier, self.multiplier_min) or math.isclose(self._adamw_multiplier, self.multiplier_max),
            "per_update_log_change_muon": math.log(self._muon_multiplier / old_muon),
            "per_update_log_change_adamw": math.log(self._adamw_multiplier / old_adamw),
            "calibration_count": self.calibration_count,
            "allocation_error": allocation_error,
        }

    def update(self, **kwargs: Any):
        component_values = {
            name: kwargs.pop(name)
            for name in (
                "predicted_decrease_muon",
                "predicted_decrease_adamw",
                "muon_grad_norm",
                "adamw_grad_norm",
                "muon_update_norm",
                "adamw_update_norm",
            )
        }
        stats = self.global_controller.update(**kwargs)
        # The scalar controller's public validity flag records the caller's
        # gate. Its skipped_reason additionally captures intrinsic rejection
        # (for example nonfinite loss/rho, a floored prediction, or a bad
        # alignment step). Allocation must follow the effective decision.
        feedback_observation_valid = bool(
            stats.feedback_observation_valid and stats.skipped_reason is None
        )
        self._update_allocation(
            **component_values,
            feedback_observation_valid=feedback_observation_valid,
        )
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_type": "dual",
            "schema_version": DUAL_CONTROLLER_STATE_SCHEMA_VERSION,
            "global_controller": self.global_controller.state_dict(),
            "allocation_kp": self.allocation_kp,
            "allocation_deadband": self.allocation_deadband,
            "allocation_log_min": self.allocation_log_min,
            "allocation_log_max": self.allocation_log_max,
            "multiplier_min": self.multiplier_min,
            "multiplier_max": self.multiplier_max,
            "calibration_steps": self.calibration_steps,
            "calibration_beta": self.calibration_beta,
            "min_contribution_fraction": self.min_contribution_fraction,
            "allocation_period": self.allocation_period,
            "score_eps": self.score_eps,
            "allocation_log_scale": self.allocation_log_scale,
            "muon_weight": self._muon_weight,
            "adamw_weight": self._adamw_weight,
            "muon_multiplier": self._muon_multiplier,
            "adamw_multiplier": self._adamw_multiplier,
            "muon_score_center": self.muon_score_center,
            "adamw_score_center": self.adamw_score_center,
            "muon_score_scale": self.muon_score_scale,
            "adamw_score_scale": self.adamw_score_scale,
            "calibration_count": self.calibration_count,
            "last_dual_stats": self.last_dual_stats,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("controller_type", "legacy") != "dual":
            raise ValueError("cannot load a legacy controller state into dual control")
        if int(state.get("schema_version", 1)) != DUAL_CONTROLLER_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported dual controller state schema")
        for name in (
            "allocation_kp",
            "allocation_deadband",
            "allocation_log_min",
            "allocation_log_max",
            "multiplier_min",
            "multiplier_max",
            "calibration_steps",
            "calibration_beta",
            "min_contribution_fraction",
            "allocation_period",
            "score_eps",
        ):
            expected = getattr(self, name)
            saved = state.get(name, expected)
            if isinstance(expected, int):
                if int(saved) != expected:
                    raise ValueError(f"dual controller configuration mismatch: {name}")
            elif not math.isclose(
                float(saved), float(expected), rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"dual controller configuration mismatch: {name}")
        self.global_controller.load_state_dict(state["global_controller"])
        self.allocation_log_scale = _clip(
            float(state.get("allocation_log_scale", 0.0)),
            self.allocation_log_min,
            self.allocation_log_max,
        )
        self._muon_weight = _clip(float(state.get("muon_weight", 0.5)), 0.0, 1.0)
        self._adamw_weight = _clip(
            float(state.get("adamw_weight", 1.0 - self._muon_weight)), 0.0, 1.0
        )
        self._muon_multiplier = _clip(
            float(state.get("muon_multiplier", self.global_controller.alpha)),
            self.multiplier_min,
            self.multiplier_max,
        )
        self._adamw_multiplier = _clip(
            float(state.get("adamw_multiplier", self.global_controller.alpha)),
            self.multiplier_min,
            self.multiplier_max,
        )
        for name in (
            "muon_score_center",
            "adamw_score_center",
            "muon_score_scale",
            "adamw_score_scale",
        ):
            value = state.get(name)
            setattr(self, name, None if value is None else float(value))
        self.calibration_count = int(state.get("calibration_count", 0))
        self.last_dual_stats = dict(state.get("last_dual_stats", self._empty_stats()))
