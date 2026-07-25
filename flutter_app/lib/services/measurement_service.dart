import '../constants/config.dart';
import '../models/body_measurements.dart';
import 'who_data_service.dart';

/// Converts pixel-space body segments into cm measurements + body build.
/// No I/O — pure logic. Port of measurement_service.py height resolution
/// and body build classification.
class MeasurementService {
  final WhoDataService _who;
  MeasurementService(this._who);

  BodyMeasurements compute({
    required BodySegments segments,
    SideViewSegments? sideSegments,
    required double ageMonths,
    required String sex,
    double? manualHeightCm,
    required double poseConfidence,
  }) {
    final effectiveHeightCm = _resolveHeight(
      manualHeightCm: manualHeightCm,
      ageMonths: ageMonths,
      sex: sex,
    );
    final method = manualHeightCm != null ? 'manual' : 'who_statistical';

    final scale = _scale(segments, effectiveHeightCm);

    final shoulderCm = segments.shoulderWidthPx != null
        ? segments.shoulderWidthPx! * scale
        : _imputeShoulderCm(effectiveHeightCm, ageMonths);
    // Snyder ratio: hip width ≈ 0.88 × shoulder width.
    final hipCm = segments.hipWidthPx != null
        ? segments.hipWidthPx! * scale
        : shoulderCm * 0.88;
    // Snyder ratio: torso ≈ 0.30 × height.
    final torsoCm = segments.torsoLengthPx != null
        ? segments.torsoLengthPx! * scale
        : effectiveHeightCm * 0.30;
    final armCm = segments.upperArmLengthPx != null
        ? segments.upperArmLengthPx! * scale
        : _imputeArmCm(effectiveHeightCm, ageMonths);

    double? chestCm;
    double? abdCm;
    bool sideUsed = false;
    if (sideSegments != null && sideSegments.totalHeightPx != null) {
      final sideScale = effectiveHeightCm / sideSegments.totalHeightPx!;
      if (sideSegments.chestDepthPx != null) {
        final scaled = sideSegments.chestDepthPx! * sideScale;
        if (scaled >= 2.0 && scaled <= 50.0) chestCm = scaled;
      }
      if (sideSegments.abdDepthPx != null) {
        final scaled = sideSegments.abdDepthPx! * sideScale;
        if (scaled >= 2.0 && scaled <= 50.0) abdCm = scaled;
      }
      sideUsed = chestCm != null || abdCm != null;
    }

    final ratio = shoulderCm / effectiveHeightCm;
    final expected = expectedShoulderRatio(ageMonths);
    String build;
    int buildScore;
    if (ratio < expected - bodyBuildThresholdMl) {
      build = 'slender';
      buildScore = -1;
    } else if (ratio > expected + bodyBuildThresholdMl) {
      build = 'stocky';
      buildScore = 1;
    } else {
      build = 'average';
      buildScore = 0;
    }

    return BodyMeasurements(
      effectiveHeightCm: effectiveHeightCm,
      shoulderWidthCm: shoulderCm,
      hipWidthCm: hipCm,
      torsoLengthCm: torsoCm,
      upperArmLengthCm: armCm,
      chestDepthCm: chestCm,
      abdDepthCm: abdCm,
      bodyBuild: build,
      bodyBuildScore: buildScore,
      confidence: poseConfidence,
      estimationMethod: method,
      sideViewUsed: sideUsed,
    );
  }

  double _resolveHeight({
    required double? manualHeightCm,
    required double ageMonths,
    required String sex,
  }) {
    if (manualHeightCm != null && manualHeightCm > 0) return manualHeightCm;
    final median = _who.getMedianHeightForAge(sex, ageMonths.floor());
    if (median != null) return median;
    throw StateError(
      'WHO height-for-age median is unavailable for this child. '
      'Enter a measured height to continue.',
    );
  }

  double _scale(BodySegments segments, double heightCm) {
    final px = segments.totalHeightPx;
    if (px != null && px > 0) return heightCm / px;
    return 1.0;
  }

  double _imputeShoulderCm(double heightCm, double ageMonths) {
    if (ageMonths < 24) return heightCm * 0.200;
    if (ageMonths < 48) return heightCm * 0.210;
    return heightCm * 0.218;
  }

  double _imputeArmCm(double heightCm, double ageMonths) {
    if (ageMonths < 24) return heightCm * 0.150;
    if (ageMonths < 48) return heightCm * 0.158;
    return heightCm * 0.165;
  }
}
