/// Computes HAZ, WHZ, WAZ and BAZ z-scores using WHO reference data.
///
/// Ported from Python nutrition_service.py — logic is identical.
library;

import 'who_data_service.dart';

class NutritionService {
  final WhoDataService _who;
  NutritionService(this._who);

  /// Compute Height-for-Age Z-score (HAZ).
  ///
  /// Returns null if WHO reference data is unavailable for the given
  /// [sex] and [ageMonths].
  double? computeHaz(String sex, int ageMonths, double heightCm) {
    final lms = _who.getHazLms(sex, ageMonths);
    if (lms == null) return null;
    return WhoDataService.lmsZscore(heightCm, lms.$1, lms.$2, lms.$3);
  }

  /// Compute HAZ using exact decimal age derived from DOB and visit date.
  double? computeHazForAge(String sex, double ageMonths, double heightCm) {
    final lms = _who.getHazLmsForAge(sex, ageMonths);
    if (lms == null) return null;
    return WhoDataService.lmsZscore(heightCm, lms.$1, lms.$2, lms.$3);
  }

  /// Compute Weight-for-Height Z-score (WHZ).
  ///
  /// Uses LMS parameters from [WhoDataService.getWfhLms] and the standard
  /// WHO LMS formula via [WhoDataService.lmsZscore].
  ///
  /// Returns null if LMS data is unavailable for the given [sex] / [heightCm].
  double? computeWhz(
    String sex,
    double ageMonths,
    double heightCm,
    double weightKg,
  ) {
    final lms = _who.getWfhLms(sex, heightCm, ageMonths);
    if (lms == null) return null;
    return WhoDataService.lmsZscore(weightKg, lms.$1, lms.$2, lms.$3);
  }

  /// Compute Weight-for-Age Z-score (WAZ).
  double? computeWaz(String sex, double ageMonths, double weightKg) {
    final lms = _who.getWfaLms(sex, ageMonths);
    if (lms == null) return null;
    return WhoDataService.lmsZscore(weightKg, lms.$1, lms.$2, lms.$3);
  }

  /// Compute BMI-for-Age Z-score (BAZ).
  double? computeBaz(String sex, double ageMonths, double bmi) {
    final lms = _who.getBfaLms(sex, ageMonths);
    if (lms == null) return null;
    return WhoDataService.lmsZscore(bmi, lms.$1, lms.$2, lms.$3);
  }
}
