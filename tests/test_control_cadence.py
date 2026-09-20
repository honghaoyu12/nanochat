"""CPU tests for scheduled Nanochat probe cadence."""

import pytest

from nanochat.control_cadence import (
    ControlCadence,
    ControlProbeFractionSchedule,
    select_probe_microbatch_indices,
)


def test_fraction_schedule_resolves_controller_relative_phases():
    schedule = ControlProbeFractionSchedule.from_spec(
        "0:0.5,1200:0.25,1800:0.125", fixed_fraction=1.0
    )

    assert schedule.canonical_spec == "0:0.5,1200:0.25,1800:0.125"
    assert schedule.active_entry(step=100, control_start_step=100) == (0, 0, 0.5)
    assert schedule.active_entry(step=1300, control_start_step=100) == (1, 1200, 0.25)
    assert schedule.active_entry(step=1900, control_start_step=100) == (2, 1800, 0.125)


def test_fraction_schedule_fixed_mode_preserves_scalar_behavior():
    schedule = ControlProbeFractionSchedule.from_spec("", fixed_fraction=0.25)

    assert schedule.canonical_spec == ""
    assert schedule.active_entry(step=37, control_start_step=5) == (0, 0, 0.25)
    assert schedule.state_dict() == {"fixed_fraction": 0.25, "schedule": ""}


@pytest.mark.parametrize(
    "spec",
    [
        "5:0.5",
        "0:0.5,0:0.25",
        "0:0",
        "0:1.1",
        "0:nan",
        "0:half",
        "0:0.5,,1200:0.25",
    ],
)
def test_invalid_fraction_schedule_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        ControlProbeFractionSchedule.from_spec(spec, fixed_fraction=1.0)


def test_fraction_schedule_resume_rejects_changed_configuration():
    schedule = ControlProbeFractionSchedule.from_spec(
        "0:0.5,1800:0.125", fixed_fraction=1.0
    )

    schedule.validate_resume(schedule.state_dict())
    schedule.validate_resume(None)
    with pytest.raises(ValueError, match="probe fraction checkpoint configuration mismatch"):
        schedule.validate_resume({"fixed_fraction": 1.0, "schedule": "0:0.5,1800:0.25"})
    with pytest.raises(ValueError, match="probe fraction checkpoint configuration mismatch"):
        schedule.validate_resume({"fixed_fraction": 0.5, "schedule": ""})


def test_fixed_period_preserves_legacy_relative_step_behavior():
    cadence = ControlCadence.from_spec("", fixed_period=5)

    assert cadence.canonical_spec == ""
    assert cadence.active_period(step=17, control_start_step=3) == 5
    assert [cadence.is_probe_step(step=step, control_start_step=3) for step in range(3, 14)] == [
        True, False, False, False, False, True, False, False, False, False, True
    ]


def test_schedule_forces_a_probe_at_each_phase_start():
    cadence = ControlCadence.from_spec("0:5,10:20,15:50", fixed_period=1)

    assert cadence.canonical_spec == "0:5,10:20,15:50"
    assert cadence.active_period(step=9, control_start_step=0) == 5
    assert cadence.active_period(step=10, control_start_step=0) == 20
    assert cadence.active_period(step=16, control_start_step=0) == 50
    assert [
        step
        for step in range(0, 66)
        if cadence.is_probe_step(step=step, control_start_step=0)
    ] == [0, 5, 10, 15, 65]


def test_schedule_is_relative_to_control_start_step():
    cadence = ControlCadence.from_spec("0:5,4:10", fixed_period=1)

    assert cadence.is_probe_step(step=9, control_start_step=5)
    assert not cadence.is_probe_step(step=13, control_start_step=5)
    assert cadence.is_probe_step(step=19, control_start_step=5)
    assert not cadence.is_probe_step(step=18, control_start_step=5)
    assert not cadence.is_probe_step(step=4, control_start_step=5)


@pytest.mark.parametrize(
    "spec",
    [
        "5:10",
        "0:5,0:10",
        "0:0",
        "-1:5",
        "0:five",
        "0:5,,10:20",
    ],
)
def test_invalid_schedule_specs_are_rejected(spec):
    if spec == "":
        return
    with pytest.raises(ValueError):
        ControlCadence.from_spec(spec, fixed_period=5)


def test_resume_rejects_changed_cadence_but_accepts_same_state():
    cadence = ControlCadence.from_spec("0:5,1000:20", fixed_period=1)

    cadence.validate_resume(cadence.state_dict())
    cadence.validate_resume(None)
    with pytest.raises(ValueError, match="cadence checkpoint configuration mismatch"):
        cadence.validate_resume({"fixed_period": 1, "schedule": "0:5,1000:50"})
    with pytest.raises(ValueError, match="cadence checkpoint configuration mismatch"):
        cadence.validate_resume({"fixed_period": 5, "schedule": ""})


def test_full_fraction_selects_every_microbatch():
    assert select_probe_microbatch_indices(8, 1.0, step=17) == tuple(range(8))


def test_fraction_selects_evenly_spaced_rotating_microbatches():
    first = select_probe_microbatch_indices(8, 0.5, step=0)
    second = select_probe_microbatch_indices(8, 0.5, step=1)
    assert first == (0, 2, 4, 6)
    assert second == (1, 3, 5, 7)


def test_fraction_rounds_but_keeps_at_least_one_microbatch():
    assert len(select_probe_microbatch_indices(64, 0.25, step=3)) == 16
    assert len(select_probe_microbatch_indices(3, 0.01, step=3)) == 1


def test_fraction_validation():
    with pytest.raises(ValueError):
        select_probe_microbatch_indices(8, 0.0, step=0)
    with pytest.raises(ValueError):
        select_probe_microbatch_indices(8, 1.1, step=0)
    with pytest.raises(ValueError):
        select_probe_microbatch_indices(0, 0.5, step=0)
