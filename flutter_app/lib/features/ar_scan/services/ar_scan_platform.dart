import 'package:flutter/services.dart';

import '../domain/ar_scan_models.dart';

abstract interface class ArScanPlatform {
  Future<ArScanCapability> checkCapability();
  Future<FullArScanResult?> startFullScan({
    double? ageMonths,
    String? sex,
  });
}

class MethodChannelArScanPlatform implements ArScanPlatform {
  const MethodChannelArScanPlatform({
    MethodChannel channel =
        const MethodChannel('org.childgrowthmonitor/ar_scan'),
  }) : _channel = channel;

  final MethodChannel _channel;

  @override
  Future<ArScanCapability> checkCapability() async {
    try {
      final response = await _channel.invokeMapMethod<Object?, Object?>(
        'checkCapability',
      );
      if (response == null) return _unsupported;
      return ArScanCapability.fromMap(response);
    } on MissingPluginException {
      return _unsupported;
    } on PlatformException {
      return _unsupported;
    }
  }

  @override
  Future<FullArScanResult?> startFullScan({
    double? ageMonths,
    String? sex,
  }) async {
    final response = await _channel.invokeMapMethod<Object?, Object?>(
      'startContactlessScan',
      <String, Object?>{
        if (ageMonths != null) 'ageMonths': ageMonths,
        if (sex != null) 'sex': sex,
      },
    );
    if (response == null) return null;
    return FullArScanResult.fromMap(response);
  }

  static const _unsupported = ArScanCapability(
    availability: 'unavailable',
    arSupported: false,
    transient: false,
    ramMb: 0,
  );
}
