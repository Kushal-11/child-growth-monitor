/// Computes HAZ and WHZ z-scores using WHO reference data.
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
}
