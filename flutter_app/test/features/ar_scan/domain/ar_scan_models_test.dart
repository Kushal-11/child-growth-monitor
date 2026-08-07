import 'package:child_growth_monitor_app/features/ar_scan/domain/ar_scan_models.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('compatible device offers sparse scan', () {
    const capability = ArScanCapability(
      availability: 'supported_installed',
      arSupported: true,
      transient: false,
      ramMb: sparseArMinimumRamMb,
    );
    expect(capability.shouldOfferSparseScan, isTrue);
  });

  test('unsupported and transient devices use fallback', () {
    const unsupported = ArScanCapability(
      availability: 'unsupported_device_not_capable',
      arSupported: false,
      transient: false,
      ramMb: 512,
    );
    const transient = ArScanCapability(
      availability: 'unknown_checking',
      arSupported: true,
      transient: true,
      ramMb: 512,
    );
    expect(unsupported.shouldOfferSparseScan, isFalse);
    expect(transient.shouldOfferSparseScan, isFalse);
  });

  test('result rejects clinical eligibility and invalid measurements', () {
    expect(
      () => SparseArScanResult.fromMap({
        'estimatedHeightCm': 84.2,
        'uncertaintyCm': 0.8,
        'acceptedKeyframes': 8,
        'clinicalMeasurementEligible': true,
      }),
      throwsFormatException,
    );
  });

  test('valid result remains explicitly experimental', () {
    final result = SparseArScanResult.fromMap({
      'estimatedHeightCm': 84.2,
      'uncertaintyCm': 0.8,
      'acceptedKeyframes': 8,
      'validDepthFraction': 0.7,
      'depthMode': 'automatic',
      'clinicalMeasurementEligible': false,
    });
    expect(result.toJson()['clinical_measurement_eligible'], isFalse);
    expect(result.toJson()['method'], sparseArMethodV1);
  });
}
