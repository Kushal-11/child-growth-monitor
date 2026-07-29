import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:sensors_plus/sensors_plus.dart';

class DeviceTiltSample {
  const DeviceTiltSample(this.x, this.y, this.z);

  final double x;
  final double y;
  final double z;
}

abstract interface class DeviceTiltSource {
  Stream<DeviceTiltSample> get samples;
}

class SensorsPlusDeviceTiltSource implements DeviceTiltSource {
  const SensorsPlusDeviceTiltSource();

  @override
  Stream<DeviceTiltSample> get samples => accelerometerEventStream(
        samplingPeriod: SensorInterval.uiInterval,
      ).map((event) => DeviceTiltSample(event.x, event.y, event.z));
}

class NoopDeviceTiltSource implements DeviceTiltSource {
  const NoopDeviceTiltSource();

  @override
  Stream<DeviceTiltSample> get samples => const Stream.empty();
}

/// Maintains optional phone-tilt data and produces deterministic metadata JSON.
///
/// Sensor errors deliberately clear the current tilt instead of blocking
/// capture. Devices without an accelerometer therefore use the remaining
/// quality gates.
class DeviceMetadataService {
  DeviceMetadataService({
    DeviceTiltSource tiltSource = const SensorsPlusDeviceTiltSource(),
    String Function()? platformDescription,
  })  : _tiltSource = tiltSource,
        _platformDescription = platformDescription ?? _defaultPlatform;

  final DeviceTiltSource _tiltSource;
  final String Function() _platformDescription;
  StreamSubscription<DeviceTiltSample>? _subscription;
  double? _tiltDegrees;
  bool _sensorAvailable = false;

  double? get tiltDegrees => _tiltDegrees;

  void start() {
    if (_subscription != null) return;
    try {
      _subscription = _tiltSource.samples.listen(
        (sample) {
          final magnitude = math.sqrt(
            sample.x * sample.x + sample.y * sample.y + sample.z * sample.z,
          );
          if (magnitude <= 0) {
            _tiltDegrees = null;
            return;
          }
          final cosine = (sample.y.abs() / magnitude).clamp(0.0, 1.0);
          _tiltDegrees = math.acos(cosine) * 180 / math.pi;
          _sensorAvailable = true;
        },
        onError: (_) {
          _tiltDegrees = null;
          _sensorAvailable = false;
        },
      );
    } catch (_) {
      _subscription = null;
      _tiltDegrees = null;
      _sensorAvailable = false;
    }
  }

  Future<void> stop() async {
    await _subscription?.cancel();
    _subscription = null;
  }

  String snapshotJson({
    required int displayOrientationDegrees,
    required String cameraIdentifier,
    required String lensDirection,
    required int sensorOrientationDegrees,
  }) {
    return jsonEncode({
      'platform': _platformDescription(),
      'tilt_sensor_available': _sensorAvailable,
      'tilt_degrees': _tiltDegrees,
      'display_orientation_degrees': displayOrientationDegrees,
      'camera_identifier': cameraIdentifier,
      'lens_direction': lensDirection,
      'sensor_orientation_degrees': sensorOrientationDegrees,
    });
  }

  Future<void> dispose() => stop();

  static String _defaultPlatform() =>
      '${Platform.operatingSystem}:${Platform.operatingSystemVersion}';
}
