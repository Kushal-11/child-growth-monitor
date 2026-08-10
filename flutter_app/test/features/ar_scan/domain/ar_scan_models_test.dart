import 'package:child_growth_monitor_app/features/ar_scan/domain/ar_scan_models.dart';
import 'package:flutter_test/flutter_test.dart';

Map<String, Object?> validResultMap() => {
      'method': contactlessArMethodV3,
      'estimatedHeightCm': 84.2,
      'uncertaintyCm': 0.8,
      'acceptedKeyframes': 20,
      'validDepthFraction': 0.42,
      'meanDepthConfidence': 0.81,
      'scanCoverageDegrees': 39.0,
      'cameraTravelMeters': 0.62,
      'floorStabilityCm': 1.4,
      'capturedBodyPoints': 4200,
      'durationMs': 12500,
      'qualityScore': 0.87,
      'depthMode': 'raw_depth_with_confidence',
      'shoulderWidthCm': 20.2,
      'hipWidthCm': 17.4,
      'torsoLengthCm': 25.3,
      'upperArmLengthCm': 13.5,
      'chestDepthCm': 8.1,
      'abdomenDepthCm': 8.4,
      'estimatedMuacCm': 12.3,
      'muacUncertaintyCm': 0.5,
      'poseQualityScore': 0.86,
      'geometryQualityScore': 0.83,
      'clinicalMeasurementEligible': false,
      'isEstimate': true,
    };

void main() {
  test('compatible device offers full scan', () {
    const capability = ArScanCapability(
      availability: 'supported_installed',
      arSupported: true,
      transient: false,
      ramMb: fullArMinimumRamMb,
    );
    expect(capability.shouldOfferFullScan, isTrue);
  });

  test('unsupported, transient, low-memory, and old method use fallback', () {
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
    const lowMemory = ArScanCapability(
      availability: 'supported_installed',
      arSupported: true,
      transient: false,
      ramMb: fullArMinimumRamMb - 1,
    );
    const oldMethod = ArScanCapability(
      availability: 'supported_installed',
      arSupported: true,
      transient: false,
      ramMb: 512,
      method: fullArMethodV2,
    );
    expect(unsupported.shouldOfferFullScan, isFalse);
    expect(transient.shouldOfferFullScan, isFalse);
    expect(lowMemory.shouldOfferFullScan, isFalse);
    expect(oldMethod.shouldOfferFullScan, isFalse);
  });

  test('result rejects clinical eligibility and insufficient coverage', () {
    final clinicallyEligible = validResultMap()
      ..['clinicalMeasurementEligible'] = true;
    final insufficientFrames = validResultMap()
      ..['acceptedKeyframes'] = fullArMinimumKeyframes - 1;
    expect(
      () => FullArScanResult.fromMap(clinicallyEligible),
      throwsFormatException,
    );
    expect(
      () => FullArScanResult.fromMap(insufficientFrames),
      throwsFormatException,
    );
  });

  test('valid result serializes complete experimental provenance', () {
    final result = FullArScanResult.fromMap(validResultMap());
    final json = result.toJson();
    expect(result.acceptedKeyframes, 20);
    expect(json['method'], contactlessArMethodV3);
    expect(json['clinical_measurement_eligible'], isFalse);
    expect(json['raw_media_retained'], isFalse);
    expect(json['mean_depth_confidence'], 0.81);
    expect(json['estimated_muac_cm'], 12.3);
    expect(json['shoulder_width_cm'], 20.2);
    expect(json['is_estimate'], isTrue);
  });

  test('partial contactless result keeps height when geometry is absent', () {
    final map = validResultMap()
      ..remove('shoulderWidthCm')
      ..remove('hipWidthCm')
      ..remove('torsoLengthCm')
      ..remove('upperArmLengthCm')
      ..remove('chestDepthCm')
      ..remove('abdomenDepthCm')
      ..remove('estimatedMuacCm')
      ..remove('muacUncertaintyCm')
      ..remove('poseQualityScore')
      ..remove('geometryQualityScore');
    final result = FullArScanResult.fromMap(map);
    expect(result.estimatedHeightCm, 84.2);
    expect(result.hasWeightGeometry, isFalse);
    expect(result.estimatedMuacCm, isNull);
  });
}
