import 'dart:math' as math;
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import '../models/body_measurements.dart';

/// Wraps google_mlkit_pose_detection for on-device pose estimation and
/// extracts body segments from 33 MediaPipe landmarks.
class PoseService {
  PoseService() {
    _poseDetector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.single,
        model: PoseDetectionModel.accurate,
      ),
    );
  }

  late final PoseDetector _poseDetector;

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /// Detect pose landmarks from an image file path.
  /// Returns the list of [PoseLandmark] objects (33 landmarks).
  Future<List<PoseLandmark>> detectPose(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final poses = await _poseDetector.processImage(inputImage);
    if (poses.isEmpty) return [];
    return poses.first.landmarks.values.toList();
  }

  /// Extract body segments from pose landmarks (front-view).
  ///
  /// [landmarks] is the list from [detectPose].
  /// [imageWidth] and [imageHeight] are the pixel dimensions of the image.
  BodySegments extractSegments(
    List<PoseLandmark> landmarks,
    double imageWidth,
    double imageHeight,
  ) {
    final lm = _landmarkMap(landmarks);

    // --- head ---
    final nose = lm[PoseLandmarkType.nose];
    final leftEye = lm[PoseLandmarkType.leftEye];
    final rightEye = lm[PoseLandmarkType.rightEye];
    final leftEar = lm[PoseLandmarkType.leftEar];
    final rightEar = lm[PoseLandmarkType.rightEar];
    final leftMouth = lm[PoseLandmarkType.leftMouth];
    final rightMouth = lm[PoseLandmarkType.rightMouth];

    double? headTopY;
    double? chinY;
    double noseToEye = 0;

    if (nose != null && leftEye != null && rightEye != null) {
      noseToEye = ((nose.y - leftEye.y).abs() + (nose.y - rightEye.y).abs()) / 2;
      headTopY = estimateHeadTopY(
        noseY: nose.y * imageHeight,
        leftEyeY: leftEye.y * imageHeight,
        rightEyeY: rightEye.y * imageHeight,
        leftEarY: leftEar != null ? leftEar.y * imageHeight : null,
        rightEarY: rightEar != null ? rightEar.y * imageHeight : null,
      );

      double? mouthY;
      if (leftMouth != null && rightMouth != null) {
        mouthY = ((leftMouth.y + rightMouth.y) / 2) * imageHeight;
      }

      chinY = estimateChinY(
        noseY: nose.y * imageHeight,
        noseToEye: noseToEye * imageHeight,
        mouthY: mouthY,
      );
    }

    // head confidence: visible head landmarks out of 5
    // (nose, leftEye, rightEye, leftEar, rightEar)
    final headLandmarks = [nose, leftEye, rightEye, leftEar, rightEar];
    final headVisible = headLandmarks.where((l) => l != null && l.likelihood >= 0.5).length;
    final headConfidence = headVisible / 5.0;

    // --- shoulders ---
    final leftShoulder = lm[PoseLandmarkType.leftShoulder];
    final rightShoulder = lm[PoseLandmarkType.rightShoulder];

    double? shoulderWidthPx;
    double? shoulderMidpointY;
    double torsoConfidence = 0;

    if (leftShoulder != null && rightShoulder != null) {
      shoulderWidthPx = (leftShoulder.x - rightShoulder.x).abs() * imageWidth;
      shoulderMidpointY = ((leftShoulder.y + rightShoulder.y) / 2) * imageHeight;

      if (leftShoulder.likelihood >= 0.5 && rightShoulder.likelihood >= 0.5) {
        torsoConfidence += 0.5;
      }
    }

    // --- hips ---
    final leftHip = lm[PoseLandmarkType.leftHip];
    final rightHip = lm[PoseLandmarkType.rightHip];

    double? hipWidthPx;
    double? hipMidpointY;
    double hipConfidence = 0;

    if (leftHip != null && rightHip != null) {
      hipWidthPx = (leftHip.x - rightHip.x).abs() * imageWidth;
      hipMidpointY = ((leftHip.y + rightHip.y) / 2) * imageHeight;

      if (leftHip.likelihood >= 0.5 && rightHip.likelihood >= 0.5) {
        hipConfidence = 1.0;
        torsoConfidence += 0.5;
      }
    }

    // --- torso ---
    double? torsoLengthPx;
    if (shoulderMidpointY != null && hipMidpointY != null) {
      torsoLengthPx = (hipMidpointY - shoulderMidpointY).abs();
    }

    // --- upper arm (left side) ---
    final leftElbow = lm[PoseLandmarkType.leftElbow];

    double? upperArmLengthPx;
    double armConfidence = 0;

    if (leftShoulder != null && leftElbow != null) {
      final dx = (leftShoulder.x - leftElbow.x) * imageWidth;
      final dy = (leftShoulder.y - leftElbow.y) * imageHeight;
      upperArmLengthPx = math.sqrt(dx * dx + dy * dy);

      if (leftShoulder.likelihood >= 0.5 && leftElbow.likelihood >= 0.5) {
        armConfidence = 1.0;
      }
    }

    // --- feet / heel ---
    final leftHeel = lm[PoseLandmarkType.leftHeel];
    final rightHeel = lm[PoseLandmarkType.rightHeel];
    final leftFootIndex = lm[PoseLandmarkType.leftFootIndex];
    final rightFootIndex = lm[PoseLandmarkType.rightFootIndex];
    final leftAnkle = lm[PoseLandmarkType.leftAnkle];
    final rightAnkle = lm[PoseLandmarkType.rightAnkle];

    // heelY = max y of all foot landmark y coords (lowest point in image)
    final footLandmarks = [leftHeel, rightHeel, leftFootIndex, rightFootIndex, leftAnkle, rightAnkle];
    final visibleFootYs = footLandmarks
        .where((l) => l != null && l.likelihood >= 0.5)
        .map((l) => l!.y * imageHeight)
        .toList();

    double? heelY;
    double legConfidence = 0;

    if (visibleFootYs.isNotEmpty) {
      heelY = visibleFootYs.reduce(math.max);
      legConfidence = math.min(visibleFootYs.length / 4.0, 1.0);
    }

    // --- leg length ---
    double? legLengthPx;
    if (heelY != null && hipMidpointY != null) {
      legLengthPx = (heelY - hipMidpointY).abs();
    }

    // --- total height ---
    double? totalHeightPx;
    if (heelY != null && headTopY != null) {
      totalHeightPx = (heelY - headTopY).abs();
    }

    // --- head height ---
    double? headHeightPx;
    if (headTopY != null && chinY != null) {
      headHeightPx = (chinY - headTopY).abs();
    }

    return BodySegments(
      headHeightPx: headHeightPx,
      torsoLengthPx: torsoLengthPx,
      legLengthPx: legLengthPx,
      shoulderWidthPx: shoulderWidthPx,
      hipWidthPx: hipWidthPx,
      upperArmLengthPx: upperArmLengthPx,
      totalHeightPx: totalHeightPx,
      headTopY: headTopY,
      chinY: chinY,
      shoulderMidpointY: shoulderMidpointY,
      hipMidpointY: hipMidpointY,
      heelY: heelY,
      headConfidence: headConfidence,
      torsoConfidence: torsoConfidence,
      legConfidence: legConfidence,
      hipConfidence: hipConfidence,
      armConfidence: armConfidence,
    );
  }

  /// Extract side-view segments from pose landmarks.
  ///
  /// [heightCm] is the known or estimated height used to compute scale.
  SideViewSegments extractSideSegments(
    List<PoseLandmark> landmarks,
    double heightCm,
  ) {
    final lm = _landmarkMap(landmarks);

    // compute scale: heightCm / totalHeightPx (nose to lowest heel)
    final nose = lm[PoseLandmarkType.nose];
    final leftHeel = lm[PoseLandmarkType.leftHeel];
    final rightHeel = lm[PoseLandmarkType.rightHeel];

    double? scale;
    double? totalHeightPx;

    if (nose != null) {
      final heelYs = <double>[];
      if (leftHeel != null && leftHeel.likelihood >= 0.5) heelYs.add(leftHeel.y);
      if (rightHeel != null && rightHeel.likelihood >= 0.5) heelYs.add(rightHeel.y);

      if (heelYs.isNotEmpty) {
        final maxHeelY = heelYs.reduce(math.max);
        totalHeightPx = (maxHeelY - nose.y).abs();
        if (totalHeightPx > 0) {
          scale = heightCm / totalHeightPx;
        }
      }
    }

    // chest depth: max(x) - min(x) of shoulder/elbow landmarks with visibility >= 0.5
    final chestLandmarks = [
      lm[PoseLandmarkType.leftShoulder],
      lm[PoseLandmarkType.rightShoulder],
      lm[PoseLandmarkType.leftElbow],
      lm[PoseLandmarkType.rightElbow],
    ].where((l) => l != null && l.likelihood >= 0.5).cast<PoseLandmark>().toList();

    double? chestDepthPx;
    double chestConfidence = 0;

    if (chestLandmarks.length >= 2) {
      final xs = chestLandmarks.map((l) => l.x).toList();
      final depthPx = xs.reduce(math.max) - xs.reduce(math.min);
      final depthCm = scale != null ? depthPx * scale : null;

      if (depthCm == null || (depthCm >= 2.0 && depthCm <= 50.0)) {
        chestDepthPx = depthPx;
        chestConfidence = chestLandmarks.length / 4.0;
      }
    }

    // abd depth: max(x) - min(x) of hip/knee landmarks with visibility >= 0.5
    final abdLandmarks = [
      lm[PoseLandmarkType.leftHip],
      lm[PoseLandmarkType.rightHip],
      lm[PoseLandmarkType.leftKnee],
      lm[PoseLandmarkType.rightKnee],
    ].where((l) => l != null && l.likelihood >= 0.5).cast<PoseLandmark>().toList();

    double? abdDepthPx;
    double abdConfidence = 0;

    if (abdLandmarks.length >= 2) {
      final xs = abdLandmarks.map((l) => l.x).toList();
      final depthPx = xs.reduce(math.max) - xs.reduce(math.min);
      final depthCm = scale != null ? depthPx * scale : null;

      if (depthCm == null || (depthCm >= 2.0 && depthCm <= 50.0)) {
        abdDepthPx = depthPx;
        abdConfidence = abdLandmarks.length / 4.0;
      }
    }

    return SideViewSegments(
      chestDepthPx: chestDepthPx,
      abdDepthPx: abdDepthPx,
      totalHeightPx: totalHeightPx,
      chestConfidence: chestConfidence,
      abdConfidence: abdConfidence,
    );
  }

  /// Compute overall confidence score from landmark visibility.
  ///
  /// Averages visibility of: nose, leftShoulder, rightShoulder, leftHip,
  /// rightHip, leftHeel, rightHeel.
  /// If [postureValid] is false, the result is multiplied by 0.8.
  double computeConfidence(
    List<PoseLandmark> landmarks, {
    bool postureValid = true,
  }) {
    final lm = _landmarkMap(landmarks);

    final keyLandmarks = [
      lm[PoseLandmarkType.nose],
      lm[PoseLandmarkType.leftShoulder],
      lm[PoseLandmarkType.rightShoulder],
      lm[PoseLandmarkType.leftHip],
      lm[PoseLandmarkType.rightHip],
      lm[PoseLandmarkType.leftHeel],
      lm[PoseLandmarkType.rightHeel],
    ];

    final totalVisibility = keyLandmarks.fold<double>(
      0.0,
      (sum, l) => sum + (l?.likelihood ?? 0.0),
    );
    final avgVisibility = totalVisibility / keyLandmarks.length;

    return postureValid ? avgVisibility : avgVisibility * 0.8;
  }

  /// Release the underlying pose detector resources.
  void dispose() {
    _poseDetector.close();
  }

  // -------------------------------------------------------------------------
  // Static helpers (testable without device)
  // -------------------------------------------------------------------------

  /// Estimate head top Y coordinate from nose and eye positions.
  ///
  /// Method 1 (always): nose_y - nose_to_eye * 2.5
  /// Method 2 (with ears): ear_avg_y - nose_to_eye * 3.0
  /// If both are available: average of method 1 and 2.
  ///
  /// All Y coordinates must be in the same unit (e.g., pixels).
  static double estimateHeadTopY({
    required double noseY,
    required double leftEyeY,
    required double rightEyeY,
    double? leftEarY,
    double? rightEarY,
  }) {
    final eyeAvgY = (leftEyeY + rightEyeY) / 2;
    final noseToEye = (noseY - eyeAvgY).abs();

    final method1 = noseY - noseToEye * 2.5;

    if (leftEarY != null && rightEarY != null) {
      final earAvgY = (leftEarY + rightEarY) / 2;
      final method2 = earAvgY - noseToEye * 3.0;
      return (method1 + method2) / 2;
    }

    return method1;
  }

  /// Estimate chin Y from nose position and optional mouth landmark.
  ///
  /// With mouth: mouth_y + nose_to_eye * 0.5
  /// Without mouth: nose_y + nose_to_eye * 1.5
  ///
  /// All Y coordinates must be in the same unit (e.g., pixels).
  static double estimateChinY({
    required double noseY,
    required double noseToEye,
    double? mouthY,
  }) {
    if (mouthY != null) {
      return mouthY + noseToEye * 0.5;
    }
    return noseY + noseToEye * 1.5;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  Map<PoseLandmarkType, PoseLandmark> _landmarkMap(List<PoseLandmark> landmarks) {
    return {for (final l in landmarks) l.type: l};
  }
}
