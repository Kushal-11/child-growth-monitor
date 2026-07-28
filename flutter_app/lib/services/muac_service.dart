import 'dart:math' as math;

import '../constants/config.dart';

class MuacResult {
  final double? muacCm;
  final String? muacStatus;
  final String muacMethod;
  final bool ageInRange;
  final double? confidence;
  final double? uncertaintyLowerCm;
  final double? uncertaintyUpperCm;
  final String? modelVersion;
  final String? calibrationVersion;
  final bool isDirectMeasurement;
  final bool requiresConfirmation;
  final String? referralGuidance;

  const MuacResult({
    this.muacCm,
    this.muacStatus,
    required this.muacMethod,
    required this.ageInRange,
    this.confidence,
    this.uncertaintyLowerCm,
    this.uncertaintyUpperCm,
    this.modelVersion,
    this.calibrationVersion,
    this.isDirectMeasurement = false,
    this.requiresConfirmation = false,
    this.referralGuidance,
  });
}

class MuacService {
  static const landmarkModelVersion = 'landmark-ratio-v1';
  static const landmarkCalibrationVersion = 'unvalidated-paired-tape-v0';

  static MuacResult estimate({
    required double ageMonths,
    required String sex,
    required double? whz,
    double? manualMuacCm,
    double? upperArmLengthCm,
    double? shoulderWidthCm,
    double? heightCm,
    double? landmarkVisibility,
    double? muacMedianCm,
  }) {
    final ageInRange = ageMonths >= 6.0 && ageMonths <= 59.9;

    if (manualMuacCm != null && manualMuacCm > 0) {
      return MuacResult(
        muacCm: double.parse(manualMuacCm.toStringAsFixed(1)),
        muacStatus: classifyMuac(manualMuacCm, ageInRange),
        muacMethod: 'manual',
        ageInRange: ageInRange,
        confidence: 1,
        uncertaintyLowerCm: double.parse(manualMuacCm.toStringAsFixed(1)),
        uncertaintyUpperCm: double.parse(manualMuacCm.toStringAsFixed(1)),
        calibrationVersion: 'direct-tape',
        isDirectMeasurement: true,
      );
    }

    if (upperArmLengthCm != null &&
        shoulderWidthCm != null &&
        heightCm != null) {
      final estimate = _estimateFromLandmarks(
        ageMonths: ageMonths,
        sex: sex,
        upperArmLengthCm: upperArmLengthCm,
        shoulderWidthCm: shoulderWidthCm,
        heightCm: heightCm,
        muacMedianCm: muacMedianCm,
      );
      if (estimate != null) {
        final confidence = (landmarkVisibility ?? 0.5).clamp(0.0, 1.0);
        final halfWidth = math.max(0.6, 2.0 * (1.0 - confidence));
        return MuacResult(
          muacCm: double.parse(estimate.toStringAsFixed(1)),
          // This pathway is useful as an app estimate but is not yet
          // validated against paired tape measurements for classification.
          muacStatus: null,
          muacMethod: 'landmark_estimated',
          ageInRange: ageInRange,
          confidence: double.parse(confidence.toStringAsFixed(2)),
          uncertaintyLowerCm: double.parse(
            (estimate - halfWidth).toStringAsFixed(1),
          ),
          uncertaintyUpperCm: double.parse(
            (estimate + halfWidth).toStringAsFixed(1),
          ),
          modelVersion: landmarkModelVersion,
          calibrationVersion: landmarkCalibrationVersion,
          requiresConfirmation: true,
          referralGuidance:
              'Photo-landmark MUAC estimate; confirm with a tape for clinical decisions.',
        );
      }
    }

    if (whz == null) {
      return MuacResult(
        muacCm: null,
        muacStatus: null,
        muacMethod: 'estimated_from_whz',
        ageInRange: ageInRange,
        modelVersion: 'whz-explanatory-v1',
        calibrationVersion: 'who-median-formula-v1',
        requiresConfirmation: true,
        referralGuidance: 'Obtain a direct tape MUAC measurement.',
      );
    }

    final median = medianForAge(ageMonths, sex);
    final whzClamped = whz.clamp(-3.0, 3.0);
    final muacCm = double.parse(
      (median * (1.0 + 0.087 * whzClamped)).toStringAsFixed(1),
    );
    return MuacResult(
      muacCm: muacCm,
      // This value is derived from WHZ and must not be treated as an
      // independent clinical MUAC classification.
      muacStatus: null,
      muacMethod: 'estimated_from_whz',
      ageInRange: ageInRange,
      confidence: 0.4,
      uncertaintyLowerCm: double.parse((muacCm - 1).toStringAsFixed(1)),
      uncertaintyUpperCm: double.parse((muacCm + 1).toStringAsFixed(1)),
      modelVersion: 'whz-explanatory-v1',
      calibrationVersion: 'who-median-formula-v1',
      requiresConfirmation: true,
      referralGuidance:
          'WHZ-derived MUAC is explanatory only; obtain a direct tape measurement.',
    );
  }

  static double medianForAge(double ageMonths, String sex) {
    final table = sex.toUpperCase() == 'M' ? muacBoys : muacGirls;
    if (ageMonths <= table.first.$1) return table.first.$2;
    if (ageMonths >= table.last.$1) return table.last.$2;
    for (int i = 0; i < table.length - 1; i++) {
      final (a0, m0) = table[i];
      final (a1, m1) = table[i + 1];
      if (a0 <= ageMonths && ageMonths <= a1) {
        final t = (ageMonths - a0) / (a1 - a0);
        return m0 + t * (m1 - m0);
      }
    }
    return table.last.$2;
  }

  static double? _estimateFromLandmarks({
    required double ageMonths,
    required String sex,
    required double upperArmLengthCm,
    required double shoulderWidthCm,
    required double heightCm,
    double? muacMedianCm,
  }) {
    if (upperArmLengthCm <= 0 || shoulderWidthCm <= 0 || heightCm <= 0) {
      return null;
    }

    final (armRatio, shoulderRatio) = switch (ageMonths) {
      < 12 => (0.150, 0.193),
      < 24 => (0.155, 0.207),
      < 48 => (0.160, 0.212),
      _ => (0.165, 0.218),
    };
    final expectedArm = heightCm * armRatio;
    final expectedShoulder = heightCm * shoulderRatio;
    final armFactor = math.pow(upperArmLengthCm / expectedArm, 0.30);
    final shoulderFactor = math.pow(
      shoulderWidthCm / expectedShoulder,
      0.50,
    );
    final median = muacMedianCm ?? medianForAge(ageMonths, sex);
    final estimate = median * armFactor * shoulderFactor;

    if (estimate < 7 || estimate > 22) return null;
    return estimate;
  }
}
