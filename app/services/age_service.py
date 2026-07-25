"""Calendar-aware age calculations shared by assessment entry points.

Clinical reference tables indexed by age in months use *completed* calendar
months.  A fixed days-per-month divisor can move a child into the next row
early or late, especially around February and 30/31-day birthdays.
"""
from __future__ import annotations

import calendar
from datetime import date, datetime


def _as_date(value: date | datetime) -> date:
    return value.date() if isinstance(value, datetime) else value


def _monthly_anniversary(dob: date, months: int) -> date:
    """Return ``dob`` shifted by ``months``, clamping month-end birthdays."""
    month_index = dob.year * 12 + (dob.month - 1) + months
    year, month_zero_based = divmod(month_index, 12)
    month = month_zero_based + 1
    day = min(dob.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def completed_months(dob: date | datetime, assessment_date: date | datetime) -> int:
    """Return the number of completed calendar months at assessment.

    Month-end birthdays are handled by clamping to the last day of shorter
    months: a child born on 31 January completes one month on 28/29 February.
    """
    birth = _as_date(dob)
    assessed = _as_date(assessment_date)
    if assessed < birth:
        raise ValueError("assessment_date must not be before date_of_birth")

    months = (assessed.year - birth.year) * 12 + assessed.month - birth.month
    if assessed < _monthly_anniversary(birth, months):
        months -= 1
    return months


def age_months_at(dob: date | datetime, assessment_date: date | datetime) -> float:
    """Return calendar-aware fractional age in months.

    The integer portion is completed months.  The fractional portion is the
    elapsed fraction between the child's surrounding monthly anniversaries.
    This value is useful for range checks and transparent reporting; callers
    selecting an age-indexed WHO row must use :func:`completed_months`.
    """
    birth = _as_date(dob)
    assessed = _as_date(assessment_date)
    months = completed_months(birth, assessed)
    previous = _monthly_anniversary(birth, months)
    following = _monthly_anniversary(birth, months + 1)
    interval_days = (following - previous).days
    if interval_days <= 0:
        return float(months)
    return months + (assessed - previous).days / interval_days
