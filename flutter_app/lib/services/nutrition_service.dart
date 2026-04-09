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
    final boundaries = _who.getHazBoundaries(sex, ageMonths);
    if (boundaries == null) return null;

    final zPoints = <(double, double)>[];
    for (final z in [-3, -2, -1, 0, 1, 2, 3]) {
      final v = boundaries[z];
      if (v != null) zPoints.add((z.toDouble(), v));
    }
    if (zPoints.isEmpty) return null;

    return _interpolateZscore(heightCm, zPoints);
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

  /// Interpolate a z-score from a value given a list of (z, reference_value)
  /// tuples.
  ///
  /// [zPoints] must be sorted by ascending reference value (second element).
  ///
  /// - Below lowest boundary: extrapolate left using first interval.
  /// - Above highest boundary: extrapolate right using last interval.
  /// - Between boundaries: linear interpolation.
  static double _interpolateZscore(
    double value,
    List<(double, double)> zPoints,
  ) {
    // Below lowest boundary — extrapolate left.
    if (value <= zPoints.first.$2) {
      final (z0, v0) = zPoints[0];
      final (_, v1) = zPoints[1];
      if (v1 == v0) return z0;
      return z0 - (v0 - value) / (v1 - v0);
    }

    // Above highest boundary — extrapolate right.
    if (value >= zPoints.last.$2) {
      final (_, v0) = zPoints[zPoints.length - 2];
      final (z1, v1) = zPoints[zPoints.length - 1];
      if (v1 == v0) return z1;
      return z1 + (value - v1) / (v1 - v0);
    }

    // Between boundaries — linear interpolation.
    for (int i = 0; i < zPoints.length - 1; i++) {
      final (zLow, vLow) = zPoints[i];
      final (zHigh, vHigh) = zPoints[i + 1];
      if (vLow <= value && value <= vHigh) {
        if (vHigh == vLow) return zLow;
        final fraction = (value - vLow) / (vHigh - vLow);
        return zLow + fraction * (zHigh - zLow);
      }
    }

    // Fallback (should not be reached given the guards above).
    return zPoints.last.$1;
  }
}
