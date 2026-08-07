const String fullArMethodV2 = 'arcore_guided_depth_v2';
const int fullArMinimumRamMb = 256;
const int fullArMinimumKeyframes = 12;

/// Native capability is advisory: a supported device can still fall back when
/// the Depth API startup or live multi-view quality gates fail.
class ArScanCapability {
  const ArScanCapability({
    required this.availability,
    required this.arSupported,
    required this.transient,
    required this.ramMb,
    this.method = fullArMethodV2,
  });

  factory ArScanCapability.fromMap(Map<Object?, Object?> map) =>
      ArScanCapability(
        availability: map['availability'] as String? ?? 'unknown',
        arSupported: map['arSupported'] == true,
        transient: map['transient'] == true,
        ramMb: (map['ramMb'] as num?)?.toInt() ?? 0,
        method: map['method'] as String? ?? 'unknown',
      );

  final String availability;
  final bool arSupported;
  final bool transient;
  final int ramMb;
  final String method;

  bool get shouldOfferFullScan =>
      arSupported &&
      !transient &&
      ramMb >= fullArMinimumRamMb &&
      method == fullArMethodV2;
}

class FullArScanResult {
  const FullArScanResult({
    required this.estimatedHeightCm,
    required this.uncertaintyCm,
    required this.acceptedKeyframes,
    required this.validDepthFraction,
    required this.meanDepthConfidence,
    required this.scanCoverageDegrees,
    required this.cameraTravelMeters,
    required this.floorStabilityCm,
    required this.capturedBodyPoints,
    required this.durationMs,
    required this.qualityScore,
    required this.depthMode,
  });

  factory FullArScanResult.fromMap(Map<Object?, Object?> map) {
    final method = map['method'] as String?;
    final height = _finiteDouble(map['estimatedHeightCm']);
    final uncertainty = _finiteDouble(map['uncertaintyCm']);
    final keyframes = (map['acceptedKeyframes'] as num?)?.toInt() ?? 0;
    final validDepthFraction = _finiteDouble(map['validDepthFraction']);
    final meanDepthConfidence = _finiteDouble(map['meanDepthConfidence']);
    final scanCoverageDegrees = _finiteDouble(map['scanCoverageDegrees']);
    final cameraTravelMeters = _finiteDouble(map['cameraTravelMeters']);
    final floorStabilityCm = _finiteDouble(map['floorStabilityCm']);
    final capturedBodyPoints =
        (map['capturedBodyPoints'] as num?)?.toInt() ?? 0;
    final durationMs = (map['durationMs'] as num?)?.toInt() ?? 0;
    final qualityScore = _finiteDouble(map['qualityScore']);
    if (method != fullArMethodV2 ||
        height == null ||
        height <= 0 ||
        uncertainty == null ||
        uncertainty < 0 ||
        keyframes < fullArMinimumKeyframes ||
        !_unitInterval(validDepthFraction) ||
        !_unitInterval(meanDepthConfidence) ||
        scanCoverageDegrees == null ||
        scanCoverageDegrees < 0 ||
        scanCoverageDegrees > 180 ||
        cameraTravelMeters == null ||
        cameraTravelMeters < 0 ||
        floorStabilityCm == null ||
        floorStabilityCm < 0 ||
        capturedBodyPoints <= 0 ||
        durationMs <= 0 ||
        !_unitInterval(qualityScore) ||
        map['clinicalMeasurementEligible'] != false) {
      throw const FormatException('Invalid full AR scan result');
    }
    return FullArScanResult(
      estimatedHeightCm: height,
      uncertaintyCm: uncertainty,
      acceptedKeyframes: keyframes,
      validDepthFraction: validDepthFraction!,
      meanDepthConfidence: meanDepthConfidence!,
      scanCoverageDegrees: scanCoverageDegrees,
      cameraTravelMeters: cameraTravelMeters,
      floorStabilityCm: floorStabilityCm,
      capturedBodyPoints: capturedBodyPoints,
      durationMs: durationMs,
      qualityScore: qualityScore!,
      depthMode: map['depthMode'] as String? ?? 'unknown',
    );
  }

  final double estimatedHeightCm;
  final double uncertaintyCm;
  final int acceptedKeyframes;
  final double validDepthFraction;
  final double meanDepthConfidence;
  final double scanCoverageDegrees;
  final double cameraTravelMeters;
  final double floorStabilityCm;
  final int capturedBodyPoints;
  final int durationMs;
  final double qualityScore;
  final String depthMode;

  Map<String, Object?> toJson() => {
        'method': fullArMethodV2,
        'estimated_height_cm': estimatedHeightCm,
        'uncertainty_cm': uncertaintyCm,
        'accepted_keyframes': acceptedKeyframes,
        'valid_depth_fraction': validDepthFraction,
        'mean_depth_confidence': meanDepthConfidence,
        'scan_coverage_degrees': scanCoverageDegrees,
        'camera_travel_meters': cameraTravelMeters,
        'floor_stability_cm': floorStabilityCm,
        'captured_body_points': capturedBodyPoints,
        'duration_ms': durationMs,
        'quality_score': qualityScore,
        'depth_mode': depthMode,
        'raw_media_retained': false,
        'clinical_measurement_eligible': false,
      };

  static double? _finiteDouble(Object? value) {
    final parsed = (value as num?)?.toDouble();
    return parsed != null && parsed.isFinite ? parsed : null;
  }

  static bool _unitInterval(double? value) =>
      value != null && value >= 0 && value <= 1;
}
