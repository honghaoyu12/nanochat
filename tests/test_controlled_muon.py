"""CPU tests for scalar controlled-Muon feedback selection and validation."""

import pytest

from nanochat.controlled_muon import (
    NanochatMuonController,
    select_control_feedback,
    validate_control_feedback_configuration,
)


@pytest.mark.parametrize("control_scope", ["muon_only", "all_groups"])
def test_residual_feedback_accepts_both_actuator_scopes(control_scope):
    validate_control_feedback_configuration(
        scope="muon_residual_proxy",
        controlled=True,
        control_scope=control_scope,
    )


def test_residual_feedback_requires_controlled_optimizer():
    with pytest.raises(ValueError, match="controlled optimizer"):
        validate_control_feedback_configuration(
            scope="muon_residual_proxy",
            controlled=False,
            control_scope="all_groups",
        )


def test_residual_feedback_subtracts_predicted_adamw_contribution():
    feedback = select_control_feedback(
        scope="muon_residual_proxy",
        actual_total=0.12,
        predicted_total=0.10,
        predicted_muon=0.07,
        predicted_adamw=0.03,
        total_grad_norm=11.0,
        muon_grad_norm=7.0,
        total_update_norm=5.0,
        muon_update_norm=3.0,
    )

    assert feedback.actual_for_control == pytest.approx(0.09)
    assert feedback.predicted_for_control == pytest.approx(0.07)
    assert feedback.grad_norm_for_control == pytest.approx(7.0)
    assert feedback.update_norm_for_control == pytest.approx(3.0)
    assert feedback.rho_muon_residual_proxy == pytest.approx(0.09 / 0.07)


def test_invalid_feedback_does_not_contaminate_rho_ema():
    controller = NanochatMuonController(
        variant="controlled_muon_ema",
        alpha_init=1.0,
        alpha_min=0.25,
        alpha_max=1.5,
        rho_star=0.45,
        kp=0.06,
        factor_min=0.98,
        factor_max=1.02,
    )
    first = controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        feedback_actual_decrease=0.5,
    )
    assert first.rho_ema == pytest.approx(1.0)
    before = controller.alpha
    second = controller.update(
        step=5,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        feedback_actual_decrease=-3.0,
        feedback_observation_valid=False,
        feedback_invalid_reason="adamw_predicted_fraction_too_large",
    )
    assert second.feedback_observation_valid is False
    assert second.feedback_invalid_reason == "adamw_predicted_fraction_too_large"
    assert second.rho_ema == pytest.approx(1.0)
    assert controller.alpha <= before


def test_startup_reference_reaches_target_in_about_100_steps():
    controller = NanochatMuonController(
        variant="controlled_muon_ema",
        alpha_init=1.0,
        alpha_min=0.25,
        alpha_max=1.5,
        rho_star=0.45,
        kp=0.06,
        factor_min=0.98,
        factor_max=1.02,
        startup_alpha_reference_ratio=1.5,
        startup_alpha_reference_gain=0.5,
        startup_one_sided_safety=True,
        startup_monotone=True,
    )
    alphas = [controller.alpha]
    for update in range(21):
        stats = controller.update(
            step=update * 5,
            loss_before=1.0,
            loss_after=0.5,
            predicted_decrease=0.5,
            feedback_actual_decrease=-3.0,
            feedback_observation_valid=False,
            feedback_invalid_reason="muon_predicted_fraction_too_small",
            startup_active=True,
        )
        alphas.append(stats.alpha_next)
    assert all(right >= left for left, right in zip(alphas, alphas[1:]))
    assert alphas[-1] >= 1.48
    assert alphas[-1] <= 1.5
    assert stats.startup_active is True
    assert stats.startup_alpha_reference == pytest.approx(1.5)
