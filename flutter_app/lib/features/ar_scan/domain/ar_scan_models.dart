const String contactlessArMethodV3 = 'arcore_contactless_anthropometry_v3';
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
    this.method = contactlessArMethodV3,
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
      method == contactlessArMethodV3;
}

class FullArScanResult {
  const FullArScanResult({
    this.method = contactlessArMethodV3,
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
    this.shoulderWidthCm,
    this.hipWidthCm,
    this.torsoLengthCm,
    this.upperArmLengthCm,
    this.chestDepthCm,
    this.abdomenDepthCm,
    this.estimatedMuacCm,
    this.muacUncertaintyCm,
    this.poseQualityScore,
    this.geometryQualityScore,
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
    if ((method != contactlessArMethodV3 && method != fullArMethodV2) ||
        height == null ||
        height <= 0 ||
        (method == contactlessArMethodV3 && (height < 35 || height > 145)) ||
        uncertainty == null ||
        uncertainty < 0 ||
        uncertainty > 6 ||
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
        map['clinicalMeasurementEligible'] != false ||
        (method == contactlessArMethodV3 && map['isEstimate'] != true)) {
      throw const FormatException('Invalid full AR scan result');
    }
    final shoulderWidth = _positiveDouble(map['shoulderWidthCm']);
    final hipWidth = _positiveDouble(map['hipWidthCm']);
    final torsoLength = _positiveDouble(map['torsoLengthCm']);
    final upperArmLength = _positiveDouble(map['upperArmLengthCm']);
    final chestDepth = _positiveDouble(map['chestDepthCm']);
    final abdomenDepth = _positiveDouble(map['abdomenDepthCm']);
    final muac = _positiveDouble(map['estimatedMuacCm']);
    final muacUncertainty = _positiveDouble(map['muacUncertaintyCm']);
    final poseQuality = _finiteDouble(map['poseQualityScore']);
    final geometryQuality = _finiteDouble(map['geometryQualityScore']);
    final geometryValues = [
      shoulderWidth,
      hipWidth,
      torsoLength,
      upperArmLength,
      chestDepth,
      abdomenDepth,
    ];
    final geometryPresent = geometryValues.any((value) => value != null);
    if ((geometryPresent && geometryValues.any((value) => value == null)) ||
        (shoulderWidth != null && (shoulderWidth < 5 || shoulderWidth > 45)) ||
        (hipWidth != null && (hipWidth < 5 || hipWidth > 45)) ||
        (torsoLength != null && (torsoLength < 8 || torsoLength > 60)) ||
        (upperArmLength != null &&
            (upperArmLength < 5 || upperArmLength > 35)) ||
        (chestDepth != null && (chestDepth < 3 || chestDepth > 35)) ||
        (abdomenDepth != null && (abdomenDepth < 3 || abdomenDepth > 35)) ||
        (poseQuality != null && !_unitInterval(poseQuality)) ||
        (method == contactlessArMethodV3 &&
            geometryPresent &&
            poseQuality == null) ||
        (geometryQuality != null && !_unitInterval(geometryQuality)) ||
        (method == contactlessArMethodV3 &&
            geometryPresent &&
            geometryQuality == null) ||
        (muac == null) != (muacUncertainty == null) ||
        (muac != null && (muac < 7 || muac > 24)) ||
        (muacUncertainty != null && muacUncertainty > 6) ||
        (muac != null && !geometryPresent)) {
      throw const FormatException('Invalid contactless geometry result');
    }
    return FullArScanResult(
      method: method!,
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
      shoulderWidthCm: shoulderWidth,
      hipWidthCm: hipWidth,
      torsoLengthCm: torsoLength,
      upperArmLengthCm: upperArmLength,
      chestDepthCm: chestDepth,
      abdomenDepthCm: abdomenDepth,
      estimatedMuacCm: muac,
      muacUncertaintyCm: muacUncertainty,
      poseQualityScore: poseQuality,
      geometryQualityScore: geometryQuality,
    );
  }

  factory FullArScanResult.fromJson(Map<String, dynamic> json) =>
      FullArScanResult.fromMap({
        'method': json['method'],
        'estimatedHeightCm': json['estimated_height_cm'],
        'uncertaintyCm': json['uncertainty_cm'],
        'acceptedKeyframes': json['accepted_keyframes'],
        'validDepthFraction': json['valid_depth_fraction'],
        'meanDepthConfidence': json['mean_depth_confidence'],
        'scanCoverageDegrees': json['scan_coverage_degrees'],
        'cameraTravelMeters': json['camera_travel_meters'],
        'floorStabilityCm': json['floor_stability_cm'],
        'capturedBodyPoints': json['captured_body_points'],
        'durationMs': json['duration_ms'],
        'qualityScore': json['quality_score'],
        'depthMode': json['depth_mode'],
        'shoulderWidthCm': json['shoulder_width_cm'],
        'hipWidthCm': json['hip_width_cm'],
        'torsoLengthCm': json['torso_length_cm'],
        'upperArmLengthCm': json['upper_arm_length_cm'],
        'chestDepthCm': json['chest_depth_cm'],
        'abdomenDepthCm': json['abdomen_depth_cm'],
        'estimatedMuacCm': json['estimated_muac_cm'],
        'muacUncertaintyCm': json['muac_uncertainty_cm'],
        'poseQualityScore': json['pose_quality_score'],
        'geometryQualityScore': json['geometry_quality_score'],
        'clinicalMeasurementEligible': json['clinical_measurement_eligible'],
        'isEstimate': json['is_estimate'],
      });

  final String method;
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
  final double? shoulderWidthCm;
  final double? hipWidthCm;
  final double? torsoLengthCm;
  final double? upperArmLengthCm;
  final double? chestDepthCm;
  final double? abdomenDepthCm;
  final double? estimatedMuacCm;
  final double? muacUncertaintyCm;
  final double? poseQualityScore;
  final double? geometryQualityScore;

  bool get hasWeightGeometry =>
      shoulderWidthCm != null &&
      hipWidthCm != null &&
      torsoLengthCm != null &&
      upperArmLengthCm != null &&
      chestDepthCm != null &&
      abdomenDepthCm != null;

  double get heightRangeLowerCm => estimatedHeightCm - uncertaintyCm;
  double get heightRangeUpperCm => estimatedHeightCm + uncertaintyCm;
  double? get muacRangeLowerCm => estimatedMuacCm != null
      ? estimatedMuacCm! - (muacUncertaintyCm ?? 0)
      : null;
  double? get muacRangeUpperCm => estimatedMuacCm != null
      ? estimatedMuacCm! + (muacUncertaintyCm ?? 0)
      : null;

  Map<String, Object?> toJson() => {
        'method': method,
        'estimated_height_cm': estimatedHeightCm,
        'uncertainty_cm': uncertaintyCm,
        'height_range_lower_cm': heightRangeLowerCm,
        'height_range_upper_cm': heightRangeUpperCm,
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
        if (hasWeightGeometry) ...{
          'shoulder_width_cm': shoulderWidthCm,
          'hip_width_cm': hipWidthCm,
          'torso_length_cm': torsoLengthCm,
          'upper_arm_length_cm': upperArmLengthCm,
          'chest_depth_cm': chestDepthCm,
          'abdomen_depth_cm': abdomenDepthCm,
        },
        if (estimatedMuacCm != null) ...{
          'estimated_muac_cm': estimatedMuacCm,
          'muac_uncertainty_cm': muacUncertaintyCm,
          'muac_range_lower_cm': muacRangeLowerCm,
          'muac_range_upper_cm': muacRangeUpperCm,
        },
        if (geometryQualityScore != null)
          'geometry_quality_score': geometryQualityScore,
        if (poseQualityScore != null) 'pose_quality_score': poseQualityScore,
        'raw_media_retained': false,
        'clinical_measurement_eligible': false,
        'is_estimate': true,
      };

  static double? _finiteDouble(Object? value) {
    final parsed = (value as num?)?.toDouble();
    return parsed != null && parsed.isFinite ? parsed : null;
  }

  static double? _positiveDouble(Object? value) {
    final parsed = _finiteDouble(value);
    return parsed != null && parsed > 0 ? parsed : null;
  }

  static bool _unitInterval(double? value) =>
      value != null && value >= 0 && value <= 1;
}
