import 'package:child_growth_monitor_app/services/age_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('month-end birthdays clamp to the shorter month', () {
    final dob = DateTime(2024, 1, 31);

    expect(AgeService.completedMonths(dob, DateTime(2024, 2, 27)), 0);
    expect(AgeService.completedMonths(dob, DateTime(2024, 2, 29)), 1);
    expect(AgeService.ageMonthsAt(dob, DateTime(2024, 2, 29)), 1);
  });

  test('fractional age is measured between calendar anniversaries', () {
    final dob = DateTime(2024, 1, 31);
    final age = AgeService.ageMonthsAt(dob, DateTime(2024, 2, 15));

    expect(age, closeTo(15 / 29, 0.000001));
  });

  test('the day before fifth birthday remains below 60 months', () {
    final dob = DateTime(2021, 7, 31);
    final beforeBirthday = AgeService.ageMonthsAt(dob, DateTime(2026, 7, 30));
    final birthday = AgeService.ageMonthsAt(dob, DateTime(2026, 7, 31));

    expect(beforeBirthday, lessThan(60));
    expect(beforeBirthday, greaterThanOrEqualTo(59));
    expect(birthday, 60);
  });

  test('assessment before birth date is rejected', () {
    expect(
      () => AgeService.ageMonthsAt(
        DateTime(2024, 2, 1),
        DateTime(2024, 1, 31),
      ),
      throwsArgumentError,
    );
  });
}
