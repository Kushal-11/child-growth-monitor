/// Body segments measured from pose landmarks (in pixels)
class BodySegments {
  final double? headHeightPx;
  final double? torsoLengthPx;
  final double? legLengthPx;
  final double? shoulderWidthPx;
  final double? hipWidthPx;
  final double? upperArmLengthPx;
  final double? totalHeightPx;
  final double? headTopY;
  final double? chinY;
  final double? shoulderMidpointY;
  final double? hipMidpointY;
  final double? heelY;
  final double headConfidence;
  final double torsoConfidence;
  final double legConfidence;
  final double hipConfidence;
  final double armConfidence;

  const BodySegments({
    this.headHeightPx,
    this.torsoLengthPx,
    this.legLengthPx,
    this.shoulderWidthPx,
    this.hipWidthPx,
    this.upperArmLengthPx,
    this.totalHeightPx,
    this.headTopY,
    this.chinY,
    this.shoulderMidpointY,
    this.hipMidpointY,
    this.heelY,
    this.headConfidence = 0.0,
    this.torsoConfidence = 0.0,
    this.legConfidence = 0.0,
    this.hipConfidence = 0.0,
    this.armConfidence = 0.0,
  });
}

/// Side-view measurements
class SideViewSegments {
  final double? chestDepthPx;
  final double? abdDepthPx;
  final double? totalHeightPx;
  final double chestConfidence;
  final double abdConfidence;

  const SideViewSegments({
    this.chestDepthPx,
    this.abdDepthPx,
    this.totalHeightPx,
    this.chestConfidence = 0.0,
    this.abdConfidence = 0.0,
  });
}

/// Final measurement output: cm-scaled segments + height + body build.
/// Produced by MeasurementService from BodySegments (pixels) + WHO data.
class BodyMeasurements {
  final double effectiveHeightCm;
  final double shoulderWidthCm;
  final double hipWidthCm;
  final double torsoLengthCm;
  final double upperArmLengthCm;
  final double? chestDepthCm;
  final double? abdDepthCm;
  final String bodyBuild; // "slender" | "average" | "stocky"
  final int bodyBuildScore; // -1 | 0 | 1
  final double confidence; // 0.0 - 1.0
  final String estimationMethod; // "manual" | "who_statistical"
  final bool sideViewUsed;

  const BodyMeasurements({
    required this.effectiveHeightCm,
    required this.shoulderWidthCm,
    required this.hipWidthCm,
    required this.torsoLengthCm,
    required this.upperArmLengthCm,
    this.chestDepthCm,
    this.abdDepthCm,
    required this.bodyBuild,
    required this.bodyBuildScore,
    required this.confidence,
    required this.estimationMethod,
    required this.sideViewUsed,
  });
}

/// Full measurement output from the processing pipeline
class MeasurementOutput {
  final double? predictedHeightCm;
  final BodySegments? bodySegments;
  final String? bodyBuild;
  final double weightAdjustment;
  final double confidenceScore;
  final String estimationMethod;

  const MeasurementOutput({
    this.predictedHeightCm,
    this.bodySegments,
    this.bodyBuild,
    this.weightAdjustment = 1.0,
    this.confidenceScore = 0.0,
    this.estimationMethod = 'unknown',
  });
}
