import 'package:child_growth_monitor_app/features/ar_scan/services/ar_scan_platform.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('test/ar_scan');

  tearDown(() => TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
      .setMockMethodCallHandler(channel, null));

  test('missing native plugin fails safely to fallback capability', () async {
    const platform = MethodChannelArScanPlatform(channel: channel);
    final capability = await platform.checkCapability();
    expect(capability.shouldOfferSparseScan, isFalse);
  });

  test('parses a native sparse scan result', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      if (call.method == 'startSparseScan') {
        return <String, Object?>{
          'estimatedHeightCm': 88.1,
          'uncertaintyCm': 0.6,
          'acceptedKeyframes': 8,
          'validDepthFraction': 0.75,
          'depthMode': 'automatic',
          'clinicalMeasurementEligible': false,
        };
      }
      return null;
    });
    final result = await const MethodChannelArScanPlatform(channel: channel)
        .startSparseScan();
    expect(result?.estimatedHeightCm, 88.1);
    expect(result?.acceptedKeyframes, 8);
  });
}
