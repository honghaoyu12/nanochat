"""CPU tests for scheduled Nanochat probe cadence."""

import pytest

from nanochat.control_cadence import ControlCadence, select_probe_microbatch_indices


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
