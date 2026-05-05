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

    final shoulderCm = (segments.shoulderWidthPx ??
            _imputeShoulderPx(effectiveHeightCm, ageMonths)) *
        scale;
    final hipCm = (segments.hipWidthPx ?? shoulderCm * 0.88 / scale) * scale;
    final torsoCm =
        (segments.torsoLengthPx ?? effectiveHeightCm * 0.30 / scale) * scale;
    final armCm =
        (segments.upperArmLengthPx ??
                _imputeArmPx(effectiveHeightCm, ageMonths)) *
            scale;

    double? chestCm;
    double? abdCm;
    bool sideUsed = false;
    if (sideSegments != null && sideSegments.totalHeightPx != null) {
      final sideScale = effectiveHeightCm / sideSegments.totalHeightPx!;
      if (sideSegments.chestDepthPx != null) {
        chestCm = sideSegments.chestDepthPx! * sideScale;
      }
      if (sideSegments.abdDepthPx != null) {
        abdCm = sideSegments.abdDepthPx! * sideScale;
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
    final median = _who.getMedianHeightForAge(sex, ageMonths.round());
    if (median != null) return median;
    // Final fallback: WHO 24-month median to avoid /0; surfaces as low confidence upstream.
    return 87.1;
  }

  double _scale(BodySegments segments, double heightCm) {
    final px = segments.totalHeightPx;
    if (px != null && px > 0) return heightCm / px;
    return 1.0;
  }

  double _imputeShoulderPx(double heightCm, double ageMonths) {
    if (ageMonths < 24) return heightCm * 0.200;
    if (ageMonths < 48) return heightCm * 0.210;
    return heightCm * 0.218;
  }

  double _imputeArmPx(double heightCm, double ageMonths) {
    if (ageMonths < 24) return heightCm * 0.150;
    if (ageMonths < 48) return heightCm * 0.158;
    return heightCm * 0.165;
  }
}
