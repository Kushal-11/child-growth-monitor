/// Calendar-aware child age calculations.
///
/// Clinical tables use completed calendar months. A fixed days-per-month
/// divisor can select the next month too early or too late around February and
/// 30/31-day birthdays.
class AgeService {
  const AgeService._();

  static DateTime _dateOnly(DateTime value) =>
      DateTime(value.year, value.month, value.day);

  static DateTime _monthlyAnniversary(DateTime dateOfBirth, int months) {
    final monthIndex = dateOfBirth.year * 12 + (dateOfBirth.month - 1) + months;
    final year = monthIndex ~/ 12;
    final month = monthIndex % 12 + 1;
    final lastDay = DateTime(year, month + 1, 0).day;
    final day = dateOfBirth.day < lastDay ? dateOfBirth.day : lastDay;
    return DateTime(year, month, day);
  }

  static int completedMonths(
    DateTime dateOfBirth,
    DateTime assessmentDate,
  ) {
    final birth = _dateOnly(dateOfBirth);
    final assessed = _dateOnly(assessmentDate);
    if (assessed.isBefore(birth)) {
      throw ArgumentError(
        'assessmentDate must not be before dateOfBirth',
      );
    }

    var months =
        (assessed.year - birth.year) * 12 + assessed.month - birth.month;
    if (assessed.isBefore(_monthlyAnniversary(birth, months))) {
      months--;
    }
    return months;
  }

  static double ageMonthsAt(
    DateTime dateOfBirth,
    DateTime assessmentDate,
  ) {
    final birth = _dateOnly(dateOfBirth);
    final assessed = _dateOnly(assessmentDate);
    final months = completedMonths(birth, assessed);
    final previous = _monthlyAnniversary(birth, months);
    final following = _monthlyAnniversary(birth, months + 1);
    final intervalDays = following.difference(previous).inDays;
    if (intervalDays <= 0) return months.toDouble();
    return months + assessed.difference(previous).inDays / intervalDays;
  }
}
