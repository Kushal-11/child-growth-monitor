import 'package:child_growth_monitor_app/features/ar_scan/domain/ar_scan_models.dart';
import 'package:child_growth_monitor_app/features/ar_scan/services/ar_scan_platform.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('test/ar_scan');

  tearDown(() => TestDefaultBinaryMessengerBinding
      .instance.defaultBinaryMessenger
      .setMockMethodCallHandler(channel, null));

  test('missing native plugin fails safely to fallback capability', () async {
    const platform = MethodChannelArScanPlatform(channel: channel);
    final capability = await platform.checkCapability();
    expect(capability.shouldOfferFullScan, isFalse);
  });

  test('uses full scan method and parses complete native evidence', () async {
    MethodCall? receivedCall;
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      receivedCall = call;
      return <String, Object?>{
        'method': contactlessArMethodV3,
        'estimatedHeightCm': 88.1,
        'uncertaintyCm': 0.6,
        'acceptedKeyframes': 20,
        'validDepthFraction': 0.45,
        'meanDepthConfidence': 0.82,
        'scanCoverageDegrees': 41.0,
        'cameraTravelMeters': 0.7,
        'floorStabilityCm': 1.2,
        'capturedBodyPoints': 5000,
        'durationMs': 14000,
        'qualityScore': 0.9,
        'depthMode': 'raw_depth_with_confidence',
        'shoulderWidthCm': 20.2,
        'hipWidthCm': 17.4,
        'torsoLengthCm': 26.4,
        'upperArmLengthCm': 14.1,
        'chestDepthCm': 8.0,
        'abdomenDepthCm': 8.4,
        'estimatedMuacCm': 12.2,
        'muacUncertaintyCm': 0.5,
        'poseQualityScore': 0.88,
        'geometryQualityScore': 0.84,
        'clinicalMeasurementEligible': false,
        'isEstimate': true,
      };
    });
    final result = await const MethodChannelArScanPlatform(channel: channel)
        .startFullScan(ageMonths: 30, sex: 'F');
    expect(receivedCall?.method, 'startContactlessScan');
    expect(receivedCall?.arguments, {'ageMonths': 30.0, 'sex': 'F'});
    expect(result?.estimatedHeightCm, 88.1);
    expect(result?.acceptedKeyframes, 20);
    expect(result?.meanDepthConfidence, 0.82);
    expect(result?.estimatedMuacCm, 12.2);
  });
}
