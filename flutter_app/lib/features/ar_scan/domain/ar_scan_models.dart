const String sparseArMethodV1 = 'arcore_sparse_depth_v1';
const int sparseArMinimumRamMb = 192;

/// Native capability is deliberately advisory: a supported device can still
/// fall back when Depth API startup or live quality checks fail.
class ArScanCapability {
  const ArScanCapability({
    required this.availability,
    required this.arSupported,
    required this.transient,
    required this.ramMb,
  });

  factory ArScanCapability.fromMap(Map<Object?, Object?> map) => ArScanCapability(
        availability: map['availability'] as String? ?? 'unknown',
        arSupported: map['arSupported'] == true,
        transient: map['transient'] == true,
        ramMb: (map['ramMb'] as num?)?.toInt() ?? 0,
      );

  final String availability;
  final bool arSupported;
  final bool transient;
  final int ramMb;

  bool get shouldOfferSparseScan =>
      arSupported && !transient && ramMb >= sparseArMinimumRamMb;
}

class SparseArScanResult {
  const SparseArScanResult({
    required this.estimatedHeightCm,
    required this.uncertaintyCm,
    required this.acceptedKeyframes,
    required this.validDepthFraction,
    required this.depthMode,
  });

  factory SparseArScanResult.fromMap(Map<Object?, Object?> map) {
    final height = (map['estimatedHeightCm'] as num?)?.toDouble();
    final uncertainty = (map['uncertaintyCm'] as num?)?.toDouble();
    final keyframes = (map['acceptedKeyframes'] as num?)?.toInt() ?? 0;
    if (height == null || !height.isFinite || height <= 0 ||
        uncertainty == null || !uncertainty.isFinite || uncertainty < 0 ||
        keyframes <= 0 || map['clinicalMeasurementEligible'] != false) {
      throw const FormatException('Invalid sparse AR scan result');
    }
    return SparseArScanResult(
      estimatedHeightCm: height,
      uncertaintyCm: uncertainty,
      acceptedKeyframes: keyframes,
      validDepthFraction:
          ((map['validDepthFraction'] as num?)?.toDouble() ?? 0)
              .clamp(0.0, 1.0)
              .toDouble(),
      depthMode: map['depthMode'] as String? ?? 'unknown',
    );
  }

  final double estimatedHeightCm;
  final double uncertaintyCm;
  final int acceptedKeyframes;
  final double validDepthFraction;
  final String depthMode;

  Map<String, Object?> toJson() => {
        'method': sparseArMethodV1,
        'estimated_height_cm': estimatedHeightCm,
        'uncertainty_cm': uncertaintyCm,
        'accepted_keyframes': acceptedKeyframes,
        'valid_depth_fraction': validDepthFraction,
        'depth_mode': depthMode,
        'clinical_measurement_eligible': false,
      };
}
