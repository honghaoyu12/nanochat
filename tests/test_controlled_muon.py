"""CPU tests for scalar controlled-Muon feedback selection and validation."""

import pytest

from nanochat.controlled_muon import (
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
