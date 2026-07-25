import 'dart:typed_data';

/// 14-feature vector for ML wasting prediction.
/// Feature order MUST match training - do not reorder.
class WastingFeatures {
  final double ageMonths;
  final int sexBinary; // 1 = Male, 0 = Female
  final double heightCm;
  final double shoulderWidthCm;
  final double hipWidthCm;
  final double torsoLengthCm;
  final double upperArmLengthCm;
  final double shoulderHeightRatio;
  final double hipHeightRatio;
  final int bodyBuildScore; // -1 = slender, 0 = average, 1 = stocky
  final double? chestDepthCm;
  final double? abdDepthCm;

  const WastingFeatures({
    required this.ageMonths,
    required this.sexBinary,
    required this.heightCm,
    required this.shoulderWidthCm,
    required this.hipWidthCm,
    required this.torsoLengthCm,
    required this.upperArmLengthCm,
    required this.shoulderHeightRatio,
    required this.hipHeightRatio,
    required this.bodyBuildScore,
    this.chestDepthCm,
    this.abdDepthCm,
  });

  /// Convert to 14-element Float32 array for TFLite inference.
  /// Imputes AP depth from lateral widths when side view unavailable (Snyder 1975).
  Float32List toArray() {
    final cd = chestDepthCm ?? shoulderWidthCm * 0.45;
    final ad = abdDepthCm ?? hipWidthCm * 0.50;
    final cdr = cd / heightCm;
    final adr = ad / heightCm;

    return Float32List.fromList([
      ageMonths,
      sexBinary.toDouble(),
      heightCm,
      shoulderWidthCm,
      hipWidthCm,
      torsoLengthCm,
      upperArmLengthCm,
      shoulderHeightRatio,
      hipHeightRatio,
      bodyBuildScore.toDouble(),
      cd,
      ad,
      cdr,
      adr,
    ]);
  }
}

/// ML prediction result
class WastingPrediction {
  final double? estimatedWeightKg;
  final double samProbability;
  final double mamProbability;
  final double normalProbability;
  final double riskProbability;
  final double overweightProbability;
  final String wastingStatus;
  final String? modelVersion;
  final String? trainingData;
  final bool? nonClinical;

  const WastingPrediction({
    this.estimatedWeightKg,
    required this.samProbability,
    required this.mamProbability,
    required this.normalProbability,
    required this.riskProbability,
    required this.overweightProbability,
    required this.wastingStatus,
    this.modelVersion,
    this.trainingData,
    this.nonClinical,
  });
}
