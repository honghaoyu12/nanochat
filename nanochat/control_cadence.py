"""Deterministic scheduled probe cadence for controlled Nanochat runs."""

from __future__ import annotations

from dataclasses import dataclass
import math


CadenceEntry = tuple[int, int]
ProbeFractionEntry = tuple[int, float]


def select_probe_microbatch_indices(
    num_microbatches: int, fraction: float, step: int
) -> tuple[int, ...]:
    """Select a deterministic rotating subset for a same-batch probe.

    The selected indices are spread across the accumulation window and rotate
    with the optimizer step, so fractional probes do not always observe the
    same microbatch position. Fraction is rounded to the nearest whole
    microbatch and at least one microbatch is retained.
    """
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be positive")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("probe fraction must be finite and in (0, 1]")
    count = max(1, min(num_microbatches, int(round(num_microbatches * fraction))))
    if count == num_microbatches:
        return tuple(range(num_microbatches))
    offset = int(step) % num_microbatches
    indices = {(offset + (i * num_microbatches) // count) % num_microbatches for i in range(count)}
    return tuple(sorted(indices))


@dataclass(frozen=True)
class ControlProbeFractionSchedule:
    """Resolve a fixed or phased probe fraction by controller-relative step."""

    fixed_fraction: float
    entries: tuple[ProbeFractionEntry, ...] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.fixed_fraction) or not 0.0 < self.fixed_fraction <= 1.0:
            raise ValueError("probe fraction must be finite and in (0, 1]")
        if self.entries is None:
            return
        if not self.entries or self.entries[0][0] != 0:
            raise ValueError("probe fraction schedule must start at step 0")
        previous_start = -1
        for start, fraction in self.entries:
            if start < 0 or not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
                raise ValueError(
                    "probe fraction schedule requires start >= 0 and fraction in (0, 1]"
                )
            if start <= previous_start:
                raise ValueError(
                    "probe fraction schedule start steps must be strictly increasing"
                )
            previous_start = start

    @classmethod
    def from_spec(
        cls, spec: str | None, *, fixed_fraction: float
    ) -> "ControlProbeFractionSchedule":
        if spec is None or not spec.strip():
            return cls(fixed_fraction=float(fixed_fraction))
        entries: list[ProbeFractionEntry] = []
        for raw_entry in spec.split(","):
            entry = raw_entry.strip()
            if not entry or ":" not in entry:
                raise ValueError(
                    "probe fraction schedule must use comma-separated "
                    "start_step:fraction entries"
                )
            raw_start, raw_fraction = entry.split(":", 1)
            try:
                start = int(raw_start.strip())
                fraction = float(raw_fraction.strip())
            except ValueError as exc:
                raise ValueError(
                    "probe fraction schedule contains an invalid start or fraction"
                ) from exc
            entries.append((start, fraction))
        return cls(fixed_fraction=float(fixed_fraction), entries=tuple(entries))

    @property
    def canonical_spec(self) -> str:
        if self.entries is None:
            return ""
        return ",".join(f"{start}:{fraction:g}" for start, fraction in self.entries)

    def active_entry(
        self, *, step: int, control_start_step: int
    ) -> tuple[int, int, float]:
        """Return phase index, relative phase start, and active fraction."""
        if self.entries is None:
            return 0, 0, self.fixed_fraction
        relative_step = int(step) - int(control_start_step)
        phase_index = 0
        phase_start, fraction = self.entries[0]
        for index, (start, candidate_fraction) in enumerate(self.entries):
            if relative_step < start:
                break
            phase_index = index
            phase_start = start
            fraction = candidate_fraction
        return phase_index, phase_start, fraction

    def active_fraction(self, *, step: int, control_start_step: int) -> float:
        return self.active_entry(
            step=step, control_start_step=control_start_step
        )[2]

    def state_dict(self) -> dict[str, object]:
        return {
            "fixed_fraction": self.fixed_fraction,
            "schedule": self.canonical_spec,
        }

    def validate_resume(self, saved_state: dict[str, object] | None) -> None:
        """Reject resumes that would silently change probe fractions."""
        if saved_state is None:
            return
        saved_fraction = float(saved_state.get("fixed_fraction", self.fixed_fraction))
        saved_schedule = str(saved_state.get("schedule", ""))
        if (
            saved_fraction != self.fixed_fraction
            or saved_schedule != self.canonical_spec
        ):
            raise ValueError("probe fraction checkpoint configuration mismatch")


@dataclass(frozen=True)
class ControlCadence:
    """Resolve fixed or phased probe cadence by controller-relative step."""

    fixed_period: int
    entries: tuple[CadenceEntry, ...] | None = None

    def __post_init__(self) -> None:
        if self.fixed_period <= 0:
            raise ValueError("control period must be positive")
        if self.entries is None:
            return
        if not self.entries or self.entries[0][0] != 0:
            raise ValueError("control cadence schedule must start at step 0")
        previous_start = -1
        for start, period in self.entries:
            if start < 0 or period <= 0:
                raise ValueError(
                    "control cadence schedule requires start >= 0 and period > 0"
                )
            if start <= previous_start:
                raise ValueError(
                    "control cadence schedule start steps must be strictly increasing"
                )
            previous_start = start

    @classmethod
    def from_spec(cls, spec: str | None, *, fixed_period: int) -> "ControlCadence":
        if spec is None or not spec.strip():
            return cls(fixed_period=int(fixed_period))
        entries: list[CadenceEntry] = []
        for raw_entry in spec.split(","):
            entry = raw_entry.strip()
            if not entry or ":" not in entry:
                raise ValueError(
                    "control cadence schedule must use comma-separated "
                    "start_step:period entries"
                )
            raw_start, raw_period = entry.split(":", 1)
            try:
                start = int(raw_start.strip())
                period = int(raw_period.strip())
            except ValueError as exc:
                raise ValueError(
                    "control cadence schedule contains a non-integer start or period"
                ) from exc
            entries.append((start, period))
        return cls(fixed_period=int(fixed_period), entries=tuple(entries))

    @property
    def canonical_spec(self) -> str:
        if self.entries is None:
            return ""
        return ",".join(f"{start}:{period}" for start, period in self.entries)

    def active_period(self, *, step: int, control_start_step: int) -> int:
        if self.entries is None:
            return self.fixed_period
        relative_step = int(step) - int(control_start_step)
        period = self.entries[0][1]
        for start, candidate_period in self.entries:
            if relative_step < start:
                break
            period = candidate_period
        return period

    def is_probe_step(self, *, step: int, control_start_step: int) -> bool:
        relative_step = int(step) - int(control_start_step)
        if relative_step < 0:
            return False
        if self.entries is None:
            return relative_step % self.fixed_period == 0
        phase_start = self.entries[0][0]
        period = self.entries[0][1]
        for start, candidate_period in self.entries:
            if relative_step < start:
                break
            phase_start = start
            period = candidate_period
        return (relative_step - phase_start) % period == 0

    def state_dict(self) -> dict[str, object]:
        return {"fixed_period": self.fixed_period, "schedule": self.canonical_spec}

    def validate_resume(self, saved_state: dict[str, object] | None) -> None:
        """Reject resumes that would silently change probe timing."""
        if saved_state is None:
            return
        saved_period = int(saved_state.get("fixed_period", self.fixed_period))
        saved_schedule = str(saved_state.get("schedule", ""))
        if saved_period != self.fixed_period or saved_schedule != self.canonical_spec:
            raise ValueError("control cadence checkpoint configuration mismatch")
