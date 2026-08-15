"""CPU tests for scalar controlled-Muon feedback selection and validation."""

import pytest

from nanochat.controlled_muon import (
    NanochatMuonController,
    ThreeStageLossProgressRhoReference,
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


def test_startup_gain_interpolates_without_changing_legacy_defaults():
    legacy = NanochatMuonController(
        variant="controlled_muon_ema",
        alpha_init=1.0,
        alpha_min=0.25,
        alpha_max=2.0,
        rho_star=0.3,
        kp=0.1,
        factor_min=0.98,
        factor_max=1.02,
    )
    explicit_legacy = NanochatMuonController(
        variant="controlled_muon_ema",
        alpha_init=1.0,
        alpha_min=0.25,
        alpha_max=2.0,
        rho_star=0.3,
        kp=0.1,
        factor_min=0.98,
        factor_max=1.02,
        startup_kp=0.1,
        startup_factor_max=1.02,
    )
    startup = NanochatMuonController(
        variant="controlled_muon_ema",
        alpha_init=1.0,
        alpha_min=0.25,
        alpha_max=2.0,
        rho_star=0.3,
        kp=0.1,
        factor_min=0.98,
        factor_max=1.02,
        startup_kp=0.5,
        startup_factor_max=1.08,
    )
    kwargs = dict(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
    )
    legacy_stats = legacy.update(**kwargs)
    explicit_stats = explicit_legacy.update(**kwargs)
    startup_stats = startup.update(**kwargs, startup_weight=1.0)
    assert explicit_stats.alpha_next == pytest.approx(legacy_stats.alpha_next)
    assert explicit_stats.p_term == pytest.approx(legacy_stats.p_term)
    assert startup_stats.kp_applied == pytest.approx(0.5)
    assert startup_stats.factor_max_applied == pytest.approx(1.08)
    assert startup_stats.alpha_next == pytest.approx(1.08)


def test_three_stage_reference_moves_startup_then_late_and_roundtrips():
    reference = ThreeStageLossProgressRhoReference(
        rho_start=0.3,
        rho_cruise=0.7,
        rho_late=1.0,
        startup_beta_fast=0.5,
        startup_beta_slow=0.9,
        startup_beta_reference=0.99,
        startup_beta_phase=0.8,
        startup_progress_ratio_high=0.8,
        startup_progress_ratio_low=0.4,
        startup_minimum_observations=10,
        late_beta_fast=0.9,
        late_beta_slow=0.99,
        late_beta_reference=0.999,
        late_beta_phase=0.9,
        late_progress_ratio_high=0.8,
        late_progress_ratio_low=0.2,
        late_minimum_observations=150,
    )
    assert reference.rho_star == pytest.approx(0.3)
    for step in range(120):
        reference.observe(3.0 + 7.0 * (0.95**step))
    assert reference.startup_phase > 0.9
    assert reference.startup_weight < 0.1
    assert reference.late_phase < 0.1
    assert reference.rho_star == pytest.approx(0.7, abs=0.04)
    for _ in range(300):
        reference.observe(3.0)
    assert reference.late_phase > 0.8
    assert reference.rho_star > 0.9

    restored = ThreeStageLossProgressRhoReference(
        rho_start=0.3,
        rho_cruise=0.7,
        rho_late=1.0,
        startup_beta_fast=0.5,
        startup_beta_slow=0.9,
        startup_beta_reference=0.99,
        startup_beta_phase=0.8,
        startup_progress_ratio_high=0.8,
        startup_progress_ratio_low=0.4,
        startup_minimum_observations=10,
        late_beta_fast=0.9,
        late_beta_slow=0.99,
        late_beta_reference=0.999,
        late_beta_phase=0.9,
        late_progress_ratio_high=0.8,
        late_progress_ratio_low=0.2,
        late_minimum_observations=150,
    )
    restored.load_state_dict(reference.state_dict())
    assert restored.rho_star == pytest.approx(reference.rho_star)
    assert restored.startup_phase == pytest.approx(reference.startup_phase)
    assert restored.late_phase == pytest.approx(reference.late_phase)
