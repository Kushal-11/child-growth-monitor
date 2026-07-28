"""Shared, calendar-aware age calculation and clinical-range validation."""
from calendar import monthrange
from dataclasses import dataclass
from datetime import date

from config import CLINICAL_MAX_AGE_MONTHS, CLINICAL_MIN_AGE_MONTHS


AVERAGE_DAYS_PER_MONTH = 365.25 / 12


@dataclass(frozen=True)
class Age:
    """An age retaining precision needed by WHO growth references."""

    days: int
    months: float
    completed_months: int
    completed_years: int


class AgeService:
    """Stateless calendar-age calculator shared by request and service layers."""

    @staticmethod
    def _add_calendar_months(value: date, months: int) -> date:
        """Add months, using the last valid day for short target months."""
        month_index = value.year * 12 + value.month - 1 + months
        year, zero_based_month = divmod(month_index, 12)
        month = zero_based_month + 1
        day = min(value.day, monthrange(year, month)[1])
        return date(year, month, day)

    def calculate_age(self, date_of_birth: date, as_of: date | None = None) -> Age:
        """Calculate precise and calendar-completed age values without clamping.

        Fractional months preserve the exact day difference for WHO lookup,
        while completed months and years use calendar anniversaries (including
        leap-day birthdays) rather than averages.
        """
        as_of = as_of or date.today()
        days = (as_of - date_of_birth).days
        completed_months = (
            (as_of.year - date_of_birth.year) * 12
            + as_of.month
            - date_of_birth.month
        )
        if as_of < self._add_calendar_months(date_of_birth, completed_months):
            completed_months -= 1

        completed_years = as_of.year - date_of_birth.year
        if as_of < self._add_calendar_months(date_of_birth, completed_years * 12):
            completed_years -= 1
        return Age(
            days=days,
            months=days / AVERAGE_DAYS_PER_MONTH,
            completed_months=completed_months,
            completed_years=completed_years,
        )

    def validate_clinical_age(
        self, date_of_birth: date, as_of: date | None = None
    ) -> Age:
        """Return age when supported, otherwise raise a descriptive error.

        Bounds use calendar anniversaries. Unsupported ages are rejected and
        are never moved to the nearest WHO reference row.
        """
        as_of = as_of or date.today()
        if date_of_birth > as_of:
            raise ValueError("date_of_birth cannot be in the future")

        age = self.calculate_age(date_of_birth, as_of)
        lower_bound = self._add_calendar_months(
            date_of_birth, CLINICAL_MIN_AGE_MONTHS
        )
        upper_bound = self._add_calendar_months(
            date_of_birth, CLINICAL_MAX_AGE_MONTHS
        )
        if as_of < lower_bound or as_of >= upper_bound:
            raise ValueError(
                "age must be within the supported clinical range of 0 through 59 months"
            )
        return age
