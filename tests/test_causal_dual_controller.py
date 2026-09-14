"""Tests for the opt-in causal Muon/AdamW replay path."""

from copy import deepcopy
import math

import pytest
import torch

from nanochat.controlled_muon import NanochatMuonController
from nanochat.causal_dual_controller import (
    NanochatCausalDualActuatorController,
    CausalProbeResult,
    evaluate_causal_component_probe,
    snapshot_optimizer_parameters,
)


class TinyTwoFamilyModel(torch.nn.Module):
    def __init__(self, *, fail_on_call: int | None = None):
        super().__init__()
        self.muon_weight = torch.nn.Parameter(torch.tensor(1.0))
        self.adamw_bias = torch.nn.Parameter(torch.tensor(1.0))
        self.grad_enabled_calls = []
        self.fail_on_call = fail_on_call
        self.calls = 0

    def forward(self, x, y):
        self.calls += 1
        self.grad_enabled_calls.append(torch.is_grad_enabled())
        if self.fail_on_call == self.calls:
            raise RuntimeError("forced replay failure")
        prediction = self.muon_weight * x + self.adamw_bias
        return (prediction - y).square().mean()


class CountingSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, *args, **kwargs):
        self.step_calls += 1
        return super().step(*args, **kwargs)


def _make_optimizer(model):
    return torch.optim.SGD([
        {"params": [model.muon_weight], "kind": "muon", "lr": 0.1},
        {"params": [model.adamw_bias], "kind": "adamw", "lr": 0.1},
    ], lr=0.1)


def _set_actual_update(model):
    model.muon_weight.data.fill_(1.5)
    model.adamw_bias.data.fill_(0.75)


def test_causal_probe_decomposes_updates_and_restores_actual_state():
    model = TinyTwoFamilyModel()
    optimizer = _make_optimizer(model)
    pre = snapshot_optimizer_parameters(optimizer)
    _set_actual_update(model)
    actual = {id(p): p.detach().clone() for group in optimizer.param_groups for p in group["params"]}
    optimizer_state_before = deepcopy(optimizer.state_dict())

    result = evaluate_causal_component_probe(
        model=model,
        optimizer=optimizer,
        probe_batches=[(torch.tensor([1.0]), torch.tensor([0.0]))],
        pre_update_parameters=pre,
    )

    assert isinstance(result, CausalProbeResult)
    assert result.loss_pre == pytest.approx(4.0)
    assert result.loss_total == pytest.approx(5.0625)
    assert result.loss_muon == pytest.approx(6.25)
    assert result.loss_adamw == pytest.approx(3.0625)
    assert result.actual_decrease_muon == pytest.approx(-2.25)
    assert result.actual_decrease_adamw == pytest.approx(0.9375)
    assert result.interaction_residual == pytest.approx(0.25)
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert torch.equal(parameter, actual[id(parameter)])
    assert optimizer.state_dict() == optimizer_state_before
    assert model.grad_enabled_calls == [False, False, False, False]


def test_causal_probe_restores_actual_state_after_replay_exception():
    model = TinyTwoFamilyModel(fail_on_call=3)
    optimizer = _make_optimizer(model)
    pre = snapshot_optimizer_parameters(optimizer)
    _set_actual_update(model)
    actual = {id(p): p.detach().clone() for group in optimizer.param_groups for p in group["params"]}

    with pytest.raises(RuntimeError, match="forced replay failure"):
        evaluate_causal_component_probe(
            model=model,
            optimizer=optimizer,
            probe_batches=[(torch.tensor([1.0]), torch.tensor([0.0]))],
            pre_update_parameters=pre,
        )

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            assert torch.equal(parameter, actual[id(parameter)])


def test_causal_probe_does_not_add_optimizer_step_or_backward():
    model = TinyTwoFamilyModel()
    optimizer = CountingSGD([
        {"params": [model.muon_weight], "kind": "muon", "lr": 0.1},
        {"params": [model.adamw_bias], "kind": "adamw", "lr": 0.1},
    ], lr=0.1, momentum=0.9)
    model.muon_weight.grad = torch.tensor(-0.5)
    model.adamw_bias.grad = torch.tensor(0.25)
    pre = snapshot_optimizer_parameters(optimizer)
    optimizer.step()
    optimizer_state_after_step = deepcopy(optimizer.state_dict())

    evaluate_causal_component_probe(
        model=model,
        optimizer=optimizer,
        probe_batches=[(torch.tensor([1.0]), torch.tensor([0.0]))],
        pre_update_parameters=pre,
    )

    assert optimizer.step_calls == 1
    assert model.grad_enabled_calls == [False, False, False, False]
    for state, expected_state in zip(
        optimizer.state_dict()["state"].values(),
        optimizer_state_after_step["state"].values(),
    ):
        assert torch.equal(state["momentum_buffer"], expected_state["momentum_buffer"])


def _make_causal_controller(**overrides):
    global_controller = NanochatMuonController(
        variant="controlled_muon_ema",
        alpha_init=1.0,
        alpha_min=0.8,
        alpha_max=1.2,
        rho_star=0.7,
        kp=0.02,
        factor_min=0.9,
        factor_max=1.1,
    )
    config = {
        "global_controller": global_controller,
        "allocation_kp": 0.05,
        "allocation_deadband": 0.0,
        "allocation_log_min": -0.2,
        "allocation_log_max": 0.2,
        "multiplier_min": 0.8,
        "multiplier_max": 1.2,
        "rho_beta": 0.0,
        "interaction_max_ratio": 0.5,
        "allocation_period": 1,
    }
    config.update(overrides)
    return NanochatCausalDualActuatorController(**config)


def _causal_update(controller, *, interaction=0.0):
    return controller.update(
        step=controller.num_updates,
        loss_before=1.0,
        loss_after=0.9,
        predicted_decrease=0.08,
        grad_norm=1.0,
        update_norm=0.1,
        feedback_actual_decrease=0.1,
        actual_decrease_muon=0.08,
        actual_decrease_adamw=0.02,
        actual_decrease_total=0.1,
        predicted_decrease_muon=0.04,
        predicted_decrease_adamw=0.04,
        interaction_residual=interaction,
    )


def test_causal_controller_uses_component_rho_to_separate_actuators():
    controller = _make_causal_controller()
    _causal_update(controller)
    assert controller.last_causal_stats["causal_component_valid"] is True
    assert controller.last_causal_stats["causal_rho_muon"] == pytest.approx(2.0)
    assert controller.last_causal_stats["causal_rho_adamw"] == pytest.approx(0.5)
    assert controller.muon_multiplier > controller.adamw_multiplier


def test_causal_controller_freezes_allocation_when_interaction_dominates():
    controller = _make_causal_controller()
    before = controller.allocation_log_scale
    _causal_update(controller, interaction=1.0)
    assert controller.allocation_log_scale == pytest.approx(before)
    assert controller.last_causal_stats["allocation_update_frozen"] is True
    assert controller.last_causal_stats["causal_invalid_reason"] == "interaction_dominated"


def test_causal_controller_state_round_trip_is_type_isolated():
    controller = _make_causal_controller()
    _causal_update(controller)
    restored = _make_causal_controller()
    restored.load_state_dict(controller.state_dict())
    assert restored.allocation_log_scale == pytest.approx(controller.allocation_log_scale)
    assert restored.muon_multiplier == pytest.approx(controller.muon_multiplier)
    assert restored.adamw_multiplier == pytest.approx(controller.adamw_multiplier)


def test_causal_controller_preserves_weighted_global_authority_before_clipping():
    controller = _make_causal_controller(multiplier_min=0.1, multiplier_max=10.0)
    _causal_update(controller)
    stats = controller.last_causal_stats
    weighted_log_authority = (
        stats["muon_contribution_fraction"] * math.log(controller.muon_multiplier)
        + stats["adamw_contribution_fraction"] * math.log(controller.adamw_multiplier)
    )
    assert weighted_log_authority == pytest.approx(math.log(controller.alpha))


def test_causal_controller_rejects_legacy_state():
    controller = _make_causal_controller()
    with pytest.raises(ValueError, match="non-causal controller state"):
        controller.load_state_dict(controller.global_controller.state_dict())


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("allocation_log_min", float("nan")),
        ("allocation_log_max", float("inf")),
        ("multiplier_min", float("nan")),
        ("multiplier_max", float("inf")),
    ],
)
def test_causal_controller_rejects_nonfinite_actuator_bounds(name, value):
    with pytest.raises(ValueError, match="bounds"):
        _make_causal_controller(**{name: value})
