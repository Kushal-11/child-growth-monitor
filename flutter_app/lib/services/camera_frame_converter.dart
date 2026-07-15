import 'dart:ui' show Size;

import 'package:camera/camera.dart';
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

/// Degrees the sensor frame must be rotated so ML Kit sees it upright.
/// Standard ML Kit formula: back camera subtracts the device rotation from
/// the sensor orientation, front camera adds it (mirrored sensor).
int computeRotationCompensation({
  required int sensorOrientation,
  required int deviceOrientationDegrees,
  required bool isFrontFacing,
}) {
  if (isFrontFacing) {
    return (sensorOrientation + deviceOrientationDegrees) % 360;
  }
  return (sensorOrientation - deviceOrientationDegrees + 360) % 360;
}

/// ML Kit returns landmark coordinates in the upright (post-rotation) frame:
/// width and height swap for 90°/270° rotations.
Size uprightFrameSize(int width, int height, int rotationDegrees) {
  if (rotationDegrees == 90 || rotationDegrees == 270) {
    return Size(height.toDouble(), width.toDouble());
  }
  return Size(width.toDouble(), height.toDouble());
}

const Map<DeviceOrientation, int> _orientationDegrees = {
  DeviceOrientation.portraitUp: 0,
  DeviceOrientation.landscapeLeft: 90,
  DeviceOrientation.portraitDown: 180,
  DeviceOrientation.landscapeRight: 270,
};

/// A stream frame converted for ML Kit, plus the upright size its landmark
/// coordinates are expressed in.
class ConvertedFrame {
  const ConvertedFrame(this.inputImage, this.uprightSize);

  final InputImage inputImage;
  final Size uprightSize;
}

/// Convert a camera-plugin stream frame into an ML Kit [InputImage].
///
/// Android-only: the controller must be configured with
/// [ImageFormatGroup.nv21] so frames arrive as a single NV21 plane. Returns
/// null for frames ML Kit cannot consume (unexpected format or orientation) —
/// callers simply skip those frames.
ConvertedFrame? convertCameraFrame(
  CameraImage image,
  CameraDescription camera,
  DeviceOrientation deviceOrientation,
) {
  final deviceDegrees = _orientationDegrees[deviceOrientation];
  if (deviceDegrees == null) return null;

  final rotationDegrees = computeRotationCompensation(
    sensorOrientation: camera.sensorOrientation,
    deviceOrientationDegrees: deviceDegrees,
    isFrontFacing: camera.lensDirection == CameraLensDirection.front,
  );
  final rotation = InputImageRotationValue.fromRawValue(rotationDegrees);
  if (rotation == null) return null;

  final format = InputImageFormatValue.fromRawValue(image.format.raw);
  if (format == null ||
      format != InputImageFormat.nv21 ||
      image.planes.length != 1) {
    return null;
  }

  final plane = image.planes.first;
  final inputImage = InputImage.fromBytes(
    bytes: plane.bytes,
    metadata: InputImageMetadata(
      size: Size(image.width.toDouble(), image.height.toDouble()),
      rotation: rotation,
      format: format,
      bytesPerRow: plane.bytesPerRow,
    ),
  );
  return ConvertedFrame(
    inputImage,
    uprightFrameSize(image.width, image.height, rotationDegrees),
  );
}
