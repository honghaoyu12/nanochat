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
        action_policy="legacy",
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


def _phase_hold_controller(**overrides):
    kwargs = {
        "variant": "controlled_muon_ema",
        "alpha_init": 1.0,
        "alpha_min": 0.25,
        "alpha_max": 2.0,
        "rho_star": 0.55,
        "kp": 0.03,
        "rho_beta": 0.0,
        "factor_min": 0.5,
        "factor_max": 2.0,
        "startup_kp": 0.5,
        "startup_factor_max": 2.0,
        "action_policy": "phase_hold",
        "phase_hold_cruise_policy": "hold",
        "phase_hold_start_rho": 0.25,
        "phase_hold_cruise_rho": 0.55,
        "phase_hold_cruise_kp": 0.0,
        "phase_hold_cruise_deadband": 0.05,
        "phase_hold_late_rho": 0.90,
        "phase_hold_late_kp": 0.03,
        "phase_hold_late_exponent": 2.0,
    }
    kwargs.update(overrides)
    return NanochatMuonController(**kwargs)


def _phase_hold_recovery_controller(**overrides):
    kwargs = {
        "action_policy": "phase_hold_recovery",
        "phase_hold_late_kp": 0.0,
        "recovery_terminal_ratio": 0.4,
        "recovery_exponent": 2.0,
    }
    kwargs.update(overrides)
    return _phase_hold_controller(**kwargs)


def test_legacy_policy_ignores_phase_hold_only_parameters():
    controller = NanochatMuonController(
        variant="controlled_muon_ema",
        alpha_init=1.0,
        alpha_min=0.5,
        alpha_max=1.5,
        rho_star=0.96,
        kp=0.04,
        factor_min=0.98,
        factor_max=1.02,
        action_policy="legacy",
        phase_hold_start_rho=0.91,
        phase_hold_cruise_rho=0.91,
        phase_hold_late_rho=0.96,
    )

    assert controller.action_policy == "legacy"


def test_phase_hold_policy_rejects_non_increasing_rho_targets():
    with pytest.raises(ValueError, match="start < cruise < late"):
        _phase_hold_controller(
            phase_hold_start_rho=0.55,
            phase_hold_cruise_rho=0.55,
        )


def test_phase_hold_startup_cruise_and_late_actions_are_one_sided():
    controller = _phase_hold_controller()
    startup = controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=1.0,
        late_phase=0.0,
    )
    assert startup.phase_start_weight == pytest.approx(1.0)
    assert startup.phase_start_action > 0.0
    assert startup.phase_cruise_action == pytest.approx(0.0)
    assert startup.phase_late_action == pytest.approx(0.0)

    cruise = controller.update(
        step=1,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.0,
    )
    assert cruise.phase_cruise_weight == pytest.approx(1.0)
    assert cruise.control_log_factor == pytest.approx(0.0)
    assert cruise.alpha_next == pytest.approx(cruise.alpha)

    late = controller.update(
        step=2,
        loss_before=1.0,
        loss_after=0.75,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.5,
    )
    assert late.phase_late_weight == pytest.approx(0.25)
    assert late.phase_late_action < 0.0
    assert late.control_log_factor == pytest.approx(
        late.phase_late_weight * late.phase_late_action
    )
    assert late.alpha_next < late.alpha

    no_late_increase = controller.update(
        step=3,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=1.0,
    )
    assert no_late_increase.phase_late_action == pytest.approx(0.0)
    assert no_late_increase.alpha_next == pytest.approx(no_late_increase.alpha)


def test_phase_hold_supports_multiplier_scale_rise_hold_and_late_decline():
    controller = _phase_hold_controller(
        alpha_init=1.0,
        alpha_min=0.5,
        alpha_max=1.35,
        startup_kp=0.12,
        startup_factor_max=1.02,
        factor_min=0.98,
        factor_max=1.02,
        phase_hold_start_rho=0.91,
        phase_hold_cruise_rho=0.93,
        phase_hold_late_rho=1.0,
        phase_hold_late_kp=0.04,
        phase_hold_late_exponent=4.0,
    )
    startup = controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=1.0,
        late_phase=0.0,
    )
    assert startup.alpha == pytest.approx(1.0)
    assert 1.0 < startup.alpha_next <= 1.02

    cruise = controller.update(
        step=1,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.0,
    )
    assert cruise.alpha_next == pytest.approx(startup.alpha_next)

    late = controller.update(
        step=2,
        loss_before=1.0,
        loss_after=0.55,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=1.0,
    )
    assert late.phase_late_action < 0.0
    assert 0.5 <= late.alpha_next < cruise.alpha_next


def test_phase_hold_late_exponent_delays_authority_and_weights_sum_to_one():
    q2 = _phase_hold_controller(phase_hold_late_exponent=2.0)
    q4 = _phase_hold_controller(phase_hold_late_exponent=4.0)
    kwargs = {
        "step": 0,
        "loss_before": 1.0,
        "loss_after": 0.75,
        "predicted_decrease": 0.5,
        "startup_weight": 0.0,
        "late_phase": 0.5,
    }
    q2_stats = q2.update(**kwargs)
    q4_stats = q4.update(**kwargs)
    assert q4_stats.phase_late_weight < q2_stats.phase_late_weight
    assert abs(q4_stats.control_log_factor) < abs(q2_stats.control_log_factor)
    for stats in (q2_stats, q4_stats):
        assert (
            stats.phase_start_weight
            + stats.phase_cruise_weight
            + stats.phase_late_weight
        ) == pytest.approx(1.0)


def test_phase_hold_deadband_invalid_feedback_and_state_roundtrip():
    controller = _phase_hold_controller(
        phase_hold_cruise_policy="rho_deadband",
        phase_hold_cruise_kp=0.003,
    )
    inside = controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.71,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.0,
    )
    assert inside.phase_cruise_deadband_active
    assert inside.phase_cruise_action == pytest.approx(0.0)

    outside = controller.update(
        step=1,
        loss_before=1.0,
        loss_after=0.65,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.0,
    )
    assert not outside.phase_cruise_deadband_active
    assert outside.phase_cruise_action > 0.0

    frozen = controller.update(
        step=2,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        feedback_observation_valid=False,
        feedback_invalid_reason="test_invalid",
        startup_weight=0.0,
        late_phase=1.0,
    )
    assert frozen.control_log_factor == pytest.approx(0.0)
    assert frozen.alpha_next == pytest.approx(frozen.alpha)

    restored = _phase_hold_controller(
        phase_hold_cruise_policy="rho_deadband",
        phase_hold_cruise_kp=0.003,
    )
    restored.load_state_dict(controller.state_dict())
    assert restored.alpha == pytest.approx(controller.alpha)
    assert restored.action_policy == "phase_hold"
    assert restored.last_stats is not None
    assert restored.last_stats.phase_late_weight == pytest.approx(
        controller.last_stats.phase_late_weight
    )


def test_phase_hold_recovery_ratio_one_matches_phase_hold_sequence():
    phase_hold = _phase_hold_controller(phase_hold_late_kp=0.0)
    recovery = _phase_hold_recovery_controller(recovery_terminal_ratio=1.0)
    observations = [
        (1.0, 0.0, 0.5),
        (0.4, 0.0, 0.5),
        (0.0, 0.0, 0.5),
        (0.0, 0.2, 0.4),
        (0.0, 0.7, 0.3),
        (0.0, 1.0, 0.2),
    ]

    for step, (startup_weight, late_phase, actual) in enumerate(observations):
        kwargs = {
            "step": step,
            "loss_before": 1.0,
            "loss_after": 1.0 - actual,
            "predicted_decrease": 0.5,
            "startup_weight": startup_weight,
            "late_phase": late_phase,
        }
        phase_stats = phase_hold.update(**kwargs)
        recovery_stats = recovery.update(**kwargs)
        assert recovery_stats.alpha_next == pytest.approx(phase_stats.alpha_next)
        assert recovery_stats.control_log_factor == pytest.approx(
            phase_stats.control_log_factor
        )
        assert recovery_stats.recovery_cap_binding is False


def test_phase_hold_recovery_discovers_peak_only_before_late_phase():
    controller = _phase_hold_recovery_controller(
        alpha_max=4.0,
        startup_kp=0.5,
        startup_factor_max=2.0,
    )
    first = controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=1.0,
        late_phase=0.0,
    )
    assert first.alpha_next > first.alpha
    assert first.recovery_alpha_peak == pytest.approx(first.alpha_next)
    assert first.recovery_peak_frozen is False
    assert first.recovery_alpha_cap == pytest.approx(first.alpha_next)

    cruise = controller.update(
        step=1,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.0,
    )
    assert cruise.alpha_next == pytest.approx(first.alpha_next)
    assert cruise.recovery_alpha_peak == pytest.approx(first.alpha_next)

    late = controller.update(
        step=2,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.1,
    )
    assert late.recovery_peak_frozen is True
    assert late.recovery_alpha_peak == pytest.approx(first.alpha_next)
    frozen_peak = late.recovery_alpha_peak

    controller.update(
        step=3,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=1.0,
        late_phase=0.0,
    )
    assert controller.recovery_alpha_peak == pytest.approx(frozen_peak)


def test_phase_hold_recovery_cap_is_monotone_after_peak_freeze():
    controller = _phase_hold_recovery_controller()
    controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=1.0,
        late_phase=0.0,
    )
    caps = []
    phases = [0.1, 0.4, 0.2, 0.8, 1.0]
    for step, phase in enumerate(phases, start=1):
        stats = controller.update(
            step=step,
            loss_before=1.0,
            loss_after=0.5,
            predicted_decrease=0.5,
            startup_weight=0.0,
            late_phase=phase,
        )
        caps.append(stats.recovery_alpha_cap)
    assert all(right <= left for left, right in zip(caps, caps[1:]))
    assert controller.recovery_max_late_phase == pytest.approx(1.0)
    assert caps[-1] == pytest.approx(
        controller.recovery_alpha_peak * controller.recovery_terminal_ratio
    )


def test_phase_hold_recovery_ratio_and_exponent_order_caps():
    common = {
        "action_policy": "phase_hold_recovery",
        "phase_hold_late_kp": 0.0,
    }
    ratio_small = _phase_hold_controller(
        **common, recovery_terminal_ratio=0.4, recovery_exponent=4.0
    )
    ratio_large = _phase_hold_controller(
        **common, recovery_terminal_ratio=0.6, recovery_exponent=4.0
    )
    exponent_two = _phase_hold_controller(
        **common, recovery_terminal_ratio=0.4, recovery_exponent=2.0
    )
    exponent_four = _phase_hold_controller(
        **common, recovery_terminal_ratio=0.4, recovery_exponent=4.0
    )
    controllers = [ratio_small, ratio_large, exponent_two, exponent_four]
    for controller in controllers:
        controller.update(
            step=0,
            loss_before=1.0,
            loss_after=0.5,
            predicted_decrease=0.5,
            startup_weight=1.0,
            late_phase=0.0,
        )
    stats = [
        controller.update(
            step=1,
            loss_before=1.0,
            loss_after=0.5,
            predicted_decrease=0.5,
            startup_weight=0.0,
            late_phase=0.5,
        )
        for controller in controllers
    ]
    assert stats[0].recovery_alpha_cap <= stats[1].recovery_alpha_cap
    assert stats[2].recovery_alpha_cap <= stats[3].recovery_alpha_cap


def test_phase_hold_recovery_respects_global_alpha_bounds():
    controller = _phase_hold_recovery_controller(
        alpha_min=0.75,
        alpha_max=1.5,
        recovery_terminal_ratio=0.01,
    )
    controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=1.0,
        late_phase=0.0,
    )
    terminal = controller.update(
        step=1,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=1.0,
    )
    assert terminal.alpha_next == pytest.approx(controller.alpha_min)
    assert terminal.recovery_alpha_cap == pytest.approx(controller.alpha_min)


def test_phase_hold_recovery_invalid_feedback_cannot_relax_cap():
    controller = _phase_hold_recovery_controller()
    controller.update(
        step=0,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=1.0,
        late_phase=0.0,
    )
    established = controller.update(
        step=1,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        startup_weight=0.0,
        late_phase=0.6,
    )
    invalid = controller.update(
        step=2,
        loss_before=1.0,
        loss_after=0.5,
        predicted_decrease=0.5,
        feedback_observation_valid=False,
        feedback_invalid_reason="test_invalid",
        startup_weight=0.0,
        late_phase=0.2,
    )
    assert invalid.recovery_late_phase_monotone == pytest.approx(0.6)
    assert invalid.recovery_alpha_cap == pytest.approx(established.recovery_alpha_cap)
    assert invalid.alpha_next <= established.recovery_alpha_cap


def test_phase_hold_recovery_state_roundtrip_reproduces_next_alpha():
    controller = _phase_hold_recovery_controller()
    for step, phase in enumerate((0.0, 0.3, 0.7)):
        controller.update(
            step=step,
            loss_before=1.0,
            loss_after=0.5,
            predicted_decrease=0.5,
            startup_weight=1.0 if step == 0 else 0.0,
            late_phase=phase,
        )
    restored = _phase_hold_recovery_controller()
    restored.load_state_dict(controller.state_dict())
    kwargs = {
        "step": 3,
        "loss_before": 1.0,
        "loss_after": 0.5,
        "predicted_decrease": 0.5,
        "startup_weight": 0.0,
        "late_phase": 0.9,
    }
    expected = controller.update(**kwargs)
    actual = restored.update(**kwargs)
    assert actual.alpha_next == pytest.approx(expected.alpha_next)
    assert actual.recovery_alpha_cap == pytest.approx(expected.recovery_alpha_cap)
    assert actual.recovery_cap_binding_count == expected.recovery_cap_binding_count


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"recovery_terminal_ratio": 0.0}, "terminal ratio"),
        ({"recovery_terminal_ratio": 1.01}, "terminal ratio"),
        ({"recovery_exponent": 0.5}, "exponent"),
    ],
)
def test_phase_hold_recovery_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        _phase_hold_recovery_controller(**overrides)


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
