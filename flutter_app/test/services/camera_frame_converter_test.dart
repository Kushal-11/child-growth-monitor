import 'dart:ui' show Size;

import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/services/camera_frame_converter.dart';

void main() {
  group('computeRotationCompensation', () {
    // Standard ML Kit rotation formula:
    // back camera: (sensor - device + 360) % 360, front: (sensor + device) % 360.
    test('typical Android back camera, portrait device', () {
      expect(
        computeRotationCompensation(
          sensorOrientation: 90,
          deviceOrientationDegrees: 0,
          isFrontFacing: false,
        ),
        90,
      );
    });

    test('back camera, landscape device cancels sensor rotation', () {
      expect(
        computeRotationCompensation(
          sensorOrientation: 90,
          deviceOrientationDegrees: 90,
          isFrontFacing: false,
        ),
        0,
      );
    });

    test('back camera wraps negative differences into 0..359', () {
      expect(
        computeRotationCompensation(
          sensorOrientation: 0,
          deviceOrientationDegrees: 270,
          isFrontFacing: false,
        ),
        90,
      );
    });

    test('swaps frame dimensions for 90/270 rotations', () {
      // ML Kit returns landmark coordinates in the upright (rotated) frame,
      // so a 720x480 sensor frame rotated 90° must be read as 480x720.
      expect(uprightFrameSize(720, 480, 90), const Size(480, 720));
      expect(uprightFrameSize(720, 480, 270), const Size(480, 720));
      expect(uprightFrameSize(720, 480, 0), const Size(720, 480));
      expect(uprightFrameSize(720, 480, 180), const Size(720, 480));
    });

    test('front camera adds device rotation', () {
      expect(
        computeRotationCompensation(
          sensorOrientation: 270,
          deviceOrientationDegrees: 0,
          isFrontFacing: true,
        ),
        270,
      );
      expect(
        computeRotationCompensation(
          sensorOrientation: 270,
          deviceOrientationDegrees: 90,
          isFrontFacing: true,
        ),
        0,
      );
    });
  });
}
