"""Causal Muon/AdamW attribution and dual-actuator control.

This module is intentionally separate from ``dual_controller``.  The legacy
dual controller remains a prediction-based allocation policy; this variant
uses sparse counterfactual parameter replays to measure each family's causal
loss decrease.  Replays only modify model parameters and never touch the
optimizer or its state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from nanochat.controlled_muon import NanochatMuonController


CAUSAL_DUAL_CONTROLLER_STATE_SCHEMA_VERSION = 1


def _clip(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _deadband(value: float, width: float) -> float:
    magnitude = max(0.0, abs(value) - width)
    return math.copysign(magnitude, value) if magnitude > 0.0 else 0.0


@dataclass(frozen=True)
class CausalProbeResult:
    """Loss decomposition for one post-update probe batch."""

    loss_pre: float
    loss_total: float
    loss_muon: float
    loss_adamw: float

    @property
    def actual_decrease_total(self) -> float:
        return self.loss_pre - self.loss_total

    @property
    def actual_decrease_muon(self) -> float:
        return self.loss_pre - self.loss_muon

    @property
    def actual_decrease_adamw(self) -> float:
        return self.loss_pre - self.loss_adamw

    @property
    def interaction_residual(self) -> float:
        return (
            self.actual_decrease_total
            - self.actual_decrease_muon
            - self.actual_decrease_adamw
        )


def snapshot_optimizer_parameters(optimizer: torch.optim.Optimizer) -> dict[int, torch.Tensor]:
    """Copy parameters before the ordinary update for a causal probe."""
    return {
        id(parameter): parameter.detach().clone()
        for group in optimizer.param_groups
        for parameter in group["params"]
    }


def _capture_rng_state() -> tuple[torch.Tensor, torch.Tensor | None]:
    cpu_state = torch.random.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()
    return cpu_state, cuda_state


def _restore_rng_state(state: tuple[torch.Tensor, torch.Tensor | None]) -> None:
    cpu_state, cuda_state = state
    torch.random.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)


def _average_probe_loss(
    model: torch.nn.Module,
    probe_batches: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    loss_sum = 0.0
    count = 0
    with torch.no_grad():
        for xb, yb in probe_batches:
            loss_sum += float(model(xb, yb).detach().item())
            count += 1
    if count == 0:
        raise RuntimeError("causal probe requested with no stored probe batches")
    return loss_sum / count


def _copy_parameters(
    optimizer: torch.optim.Optimizer,
    snapshots: dict[int, torch.Tensor],
    *,
    family: str | None = None,
) -> None:
    for group in optimizer.param_groups:
        if family is not None and group["kind"] != family:
            continue
        for parameter in group["params"]:
            parameter.copy_(snapshots[id(parameter)])


@torch.no_grad()
def evaluate_causal_component_probe(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    probe_batches: list[tuple[torch.Tensor, torch.Tensor]],
    pre_update_parameters: dict[int, torch.Tensor],
) -> CausalProbeResult:
    """Evaluate pre, total, Muon-only, and AdamW-only states.

    The function is called immediately after the ordinary optimizer update.
    The current parameters are therefore the actual total-update state.  A
    snapshot of that state is retained only for the duration of this sparse
    probe.  The optimizer object, gradients, and optimizer state are never
    read-modified-written by replay.
    """
    actual_parameters = {
        id(parameter): parameter.detach().clone()
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    replay_rng_state = _capture_rng_state()

    def evaluate_current_state() -> float:
        _restore_rng_state(replay_rng_state)
        return _average_probe_loss(model, probe_batches)

    try:
        # The total state is already live after the ordinary optimizer step.
        loss_total = evaluate_current_state()

        _copy_parameters(optimizer, pre_update_parameters)
        loss_pre = evaluate_current_state()

        # Keep Muon's actual update and remove AdamW's update.
        _copy_parameters(optimizer, pre_update_parameters, family="adamw")
        for group in optimizer.param_groups:
            if group["kind"] == "muon":
                for parameter in group["params"]:
                    parameter.copy_(actual_parameters[id(parameter)])
        loss_muon = evaluate_current_state()

        # Keep AdamW's actual update and remove Muon's update.
        _copy_parameters(optimizer, pre_update_parameters)
        for group in optimizer.param_groups:
            if group["kind"] == "adamw":
                for parameter in group["params"]:
                    parameter.copy_(actual_parameters[id(parameter)])
        loss_adamw = evaluate_current_state()
    finally:
        # This must run even when model forward/replay raises.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.copy_(actual_parameters[id(parameter)])
        _restore_rng_state(replay_rng_state)

    return CausalProbeResult(
        loss_pre=loss_pre,
        loss_total=loss_total,
        loss_muon=loss_muon,
        loss_adamw=loss_adamw,
    )


class NanochatCausalDualActuatorController:
    """Global P/PI/PID control plus causal Muon/AdamW allocation."""

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
        rho_beta: float = 0.90,
        interaction_max_ratio: float = 1.0,
        allocation_period: int = 5,
        predicted_eps: float = 1e-12,
    ) -> None:
        if allocation_kp < 0 or not math.isfinite(allocation_kp):
            raise ValueError("causal allocation kp must be finite and non-negative")
        if allocation_deadband < 0 or not math.isfinite(allocation_deadband):
            raise ValueError("causal allocation deadband must be finite and non-negative")
        if (
            not math.isfinite(allocation_log_min)
            or not math.isfinite(allocation_log_max)
            or allocation_log_min > allocation_log_max
        ):
            raise ValueError("causal allocation log bounds must be finite and ordered")
        if (
            not math.isfinite(multiplier_min)
            or not math.isfinite(multiplier_max)
            or not 0 < multiplier_min <= multiplier_max
        ):
            raise ValueError("causal multiplier bounds must be finite and satisfy 0 < min <= max")
        if not multiplier_min <= global_controller.alpha_min <= global_controller.alpha_max <= multiplier_max:
            raise ValueError("global controller alpha bounds must lie within causal actuator bounds")
        if not 0 <= rho_beta < 1:
            raise ValueError("causal rho beta must be in [0, 1)")
        if interaction_max_ratio < 0 or not math.isfinite(interaction_max_ratio):
            raise ValueError("causal interaction ratio must be finite and non-negative")
        if allocation_period <= 0:
            raise ValueError("causal allocation period must be positive")
        if predicted_eps <= 0 or not math.isfinite(predicted_eps):
            raise ValueError("causal predicted epsilon must be finite and positive")

        self.global_controller = global_controller
        self.allocation_kp = float(allocation_kp)
        self.allocation_deadband = float(allocation_deadband)
        self.allocation_log_min = float(allocation_log_min)
        self.allocation_log_max = float(allocation_log_max)
        self.multiplier_min = float(multiplier_min)
        self.multiplier_max = float(multiplier_max)
        self.rho_beta = float(rho_beta)
        self.interaction_max_ratio = float(interaction_max_ratio)
        self.allocation_period = int(allocation_period)
        self.predicted_eps = float(predicted_eps)
        self.allocation_log_scale = 0.0
        self.rho_muon_ema: float | None = None
        self.rho_adamw_ema: float | None = None
        self._muon_weight = 0.5
        self._adamw_weight = 0.5
        self._muon_multiplier = float(global_controller.alpha)
        self._adamw_multiplier = float(global_controller.alpha)
        self.last_causal_stats: dict[str, Any] = self._empty_stats()

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
            "muon_contribution_fraction": None,
            "adamw_contribution_fraction": None,
            "muon_component_valid": False,
            "adamw_component_valid": False,
            "muon_component_score": None,
            "adamw_component_score": None,
            "muon_multiplier": None,
            "adamw_multiplier": None,
            "causal_rho_muon": None,
            "causal_rho_adamw": None,
            "causal_rho_muon_ema": None,
            "causal_rho_adamw_ema": None,
            "causal_interaction_residual": None,
            "causal_interaction_ratio": None,
            "causal_component_valid": False,
            "causal_invalid_reason": "uninitialized",
            "allocation_update_frozen": True,
            "allocation_deadband_active": False,
            "allocation_log_scale": 0.0,
            "allocation_error": 0.0,
            "muon_bound_hit": False,
            "adamw_bound_hit": False,
        }

    def _recompute_multipliers(self) -> None:
        global_alpha = max(self.predicted_eps, float(self.global_controller.alpha))
        raw_muon = global_alpha * math.exp(self._adamw_weight * self.allocation_log_scale)
        raw_adamw = global_alpha * math.exp(-self._muon_weight * self.allocation_log_scale)
        self._muon_multiplier = _clip(raw_muon, self.multiplier_min, self.multiplier_max)
        self._adamw_multiplier = _clip(raw_adamw, self.multiplier_min, self.multiplier_max)

    @staticmethod
    def _ema(previous: float | None, value: float, beta: float) -> float:
        return value if previous is None else beta * previous + (1.0 - beta) * value

    def update(
        self,
        *,
        actual_decrease_muon: float,
        actual_decrease_adamw: float,
        actual_decrease_total: float,
        predicted_decrease_muon: float,
        predicted_decrease_adamw: float,
        interaction_residual: float,
        feedback_observation_valid: bool = True,
        feedback_invalid_reason: str | None = None,
        **global_kwargs: Any,
    ):
        global_kwargs.pop("predicted_decrease", None)
        global_kwargs.pop("feedback_actual_decrease", None)
        stats = self.global_controller.update(
            predicted_decrease=float(predicted_decrease_muon) + float(predicted_decrease_adamw),
            feedback_actual_decrease=actual_decrease_total,
            **global_kwargs,
        )
        predicted_muon = float(predicted_decrease_muon)
        predicted_adamw = float(predicted_decrease_adamw)
        actual_muon = float(actual_decrease_muon)
        actual_adamw = float(actual_decrease_adamw)
        actual_total = float(actual_decrease_total)
        interaction = float(interaction_residual)
        predicted_total = predicted_muon + predicted_adamw
        contribution_valid = bool(
            math.isfinite(predicted_total)
            and predicted_total > self.predicted_eps
            and predicted_muon >= 0.0
            and predicted_adamw >= 0.0
        )
        muon_fraction = predicted_muon / predicted_total if contribution_valid else None
        adamw_fraction = predicted_adamw / predicted_total if contribution_valid else None
        rho_muon = actual_muon / predicted_muon if predicted_muon > self.predicted_eps else None
        rho_adamw = actual_adamw / predicted_adamw if predicted_adamw > self.predicted_eps else None
        total_scale = max(abs(actual_total), abs(actual_muon) + abs(actual_adamw), self.predicted_eps)
        interaction_ratio = abs(interaction) / total_scale
        if not feedback_observation_valid:
            causal_invalid_reason = feedback_invalid_reason or "global_feedback_invalid"
        elif stats.skipped_reason is not None:
            causal_invalid_reason = f"global_{stats.skipped_reason}"
        elif not contribution_valid:
            causal_invalid_reason = "predicted_contribution_invalid"
        elif rho_muon is None or rho_adamw is None:
            causal_invalid_reason = "component_prediction_too_small"
        elif not math.isfinite(rho_muon) or not math.isfinite(rho_adamw):
            causal_invalid_reason = "component_rho_nonfinite"
        elif rho_muon <= 0.0 or rho_adamw <= 0.0:
            causal_invalid_reason = "component_rho_nonpositive"
        elif not math.isfinite(interaction_ratio):
            causal_invalid_reason = "interaction_ratio_nonfinite"
        elif interaction_ratio > self.interaction_max_ratio:
            causal_invalid_reason = "interaction_dominated"
        else:
            causal_invalid_reason = None
        valid = causal_invalid_reason is None
        deadband_active = False
        allocation_error = 0.0
        if valid:
            self._muon_weight = float(muon_fraction)
            self._adamw_weight = float(adamw_fraction)
            self.rho_muon_ema = self._ema(self.rho_muon_ema, rho_muon, self.rho_beta)
            self.rho_adamw_ema = self._ema(self.rho_adamw_ema, rho_adamw, self.rho_beta)
            allocation_error = math.log(self.rho_muon_ema) - math.log(self.rho_adamw_ema)
            action = _deadband(allocation_error, self.allocation_deadband)
            deadband_active = action == 0.0
            if self.num_updates % self.allocation_period == 0:
                self.allocation_log_scale = _clip(
                    self.allocation_log_scale + self.allocation_kp * action,
                    self.allocation_log_min,
                    self.allocation_log_max,
                )
        self._recompute_multipliers()
        self.last_causal_stats = {
            "global_multiplier": self.global_controller.alpha,
            "muon_multiplier": self._muon_multiplier,
            "adamw_multiplier": self._adamw_multiplier,
            "muon_contribution_fraction": muon_fraction,
            "adamw_contribution_fraction": adamw_fraction,
            "muon_component_valid": valid,
            "adamw_component_valid": valid,
            "muon_component_score": rho_muon,
            "adamw_component_score": rho_adamw,
            "causal_rho_muon": rho_muon,
            "causal_rho_adamw": rho_adamw,
            "causal_rho_muon_ema": self.rho_muon_ema,
            "causal_rho_adamw_ema": self.rho_adamw_ema,
            "causal_interaction_residual": interaction,
            "causal_interaction_ratio": interaction_ratio,
            "causal_component_valid": valid,
            "causal_invalid_reason": causal_invalid_reason,
            "allocation_update_frozen": not valid,
            "allocation_deadband_active": deadband_active,
            "allocation_log_scale": self.allocation_log_scale,
            "allocation_error": allocation_error,
            "muon_bound_hit": self._muon_multiplier in (self.multiplier_min, self.multiplier_max),
            "adamw_bound_hit": self._adamw_multiplier in (self.multiplier_min, self.multiplier_max),
            "feedback_invalid_reason": feedback_invalid_reason,
        }
        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_type": "causal_dual",
            "schema_version": CAUSAL_DUAL_CONTROLLER_STATE_SCHEMA_VERSION,
            "global_controller": self.global_controller.state_dict(),
            "allocation_kp": self.allocation_kp,
            "allocation_deadband": self.allocation_deadband,
            "allocation_log_min": self.allocation_log_min,
            "allocation_log_max": self.allocation_log_max,
            "multiplier_min": self.multiplier_min,
            "multiplier_max": self.multiplier_max,
            "rho_beta": self.rho_beta,
            "interaction_max_ratio": self.interaction_max_ratio,
            "allocation_period": self.allocation_period,
            "predicted_eps": self.predicted_eps,
            "allocation_log_scale": self.allocation_log_scale,
            "rho_muon_ema": self.rho_muon_ema,
            "rho_adamw_ema": self.rho_adamw_ema,
            "muon_weight": self._muon_weight,
            "adamw_weight": self._adamw_weight,
            "muon_multiplier": self._muon_multiplier,
            "adamw_multiplier": self._adamw_multiplier,
            "last_causal_stats": self.last_causal_stats,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("controller_type") != "causal_dual":
            raise ValueError("cannot load a non-causal controller state into causal dual control")
        if int(state.get("schema_version", 1)) != CAUSAL_DUAL_CONTROLLER_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported causal dual controller state schema")
        self.global_controller.load_state_dict(state["global_controller"])
        for name in (
            "allocation_kp", "allocation_deadband", "allocation_log_min",
            "allocation_log_max", "multiplier_min", "multiplier_max",
            "rho_beta", "interaction_max_ratio", "allocation_period", "predicted_eps",
        ):
            if isinstance(getattr(self, name), int):
                if int(state[name]) != getattr(self, name):
                    raise ValueError(f"causal controller configuration mismatch: {name}")
            elif not math.isclose(
                float(state[name]),
                float(getattr(self, name)),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"causal controller configuration mismatch: {name}")
        self.allocation_log_scale = float(state["allocation_log_scale"])
        self.rho_muon_ema = state.get("rho_muon_ema")
        self._muon_weight = float(state.get("muon_weight", 0.5))
        self._adamw_weight = float(state.get("adamw_weight", 0.5))
        self.rho_adamw_ema = state.get("rho_adamw_ema")
        self._recompute_multipliers()
        self.last_causal_stats = dict(state.get("last_causal_stats", self._empty_stats()))
