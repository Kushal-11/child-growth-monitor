import 'dart:math' as math;

import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import '../features/guided_capture/domain/capture_models.dart';
import '../features/guided_capture/domain/capture_thresholds.dart';
import 'pose_service.dart';

/// Why a live frame is not yet good enough to capture, in priority order.
/// Each issue maps to one actionable on-screen instruction.
enum CaptureIssue {
  noPose,
  multiplePoses,
  wrongOrientation,
  cutOffTop,
  cutOffBottom,
  missingRequiredLandmark,
  tooFar,
  offCenter,
  lowVisibility,
  excessiveTilt,
}

/// Normalized live-frame component scores plus one actionable issue.
class CaptureQuality {
  const CaptureQuality.ok()
      : issue = null,
        poseScore = 1,
        coverageScore = 1,
        orientationScore = 1,
        overallScore = 1;

  const CaptureQuality.accepted({
    required this.poseScore,
    required this.coverageScore,
    required this.orientationScore,
  })  : issue = null,
        overallScore = (poseScore + coverageScore + orientationScore) / 3;

  const CaptureQuality.blocked(
    this.issue, {
    this.poseScore = 0,
    this.coverageScore = 0,
    this.orientationScore = 0,
  }) : overallScore = (poseScore + coverageScore + orientationScore) / 3;

  final CaptureIssue? issue;
  final double poseScore;
  final double coverageScore;
  final double orientationScore;
  final double overallScore;

  bool get ready => issue == null;
}

/// Evaluate one upright live frame without performing any IO.
///
/// Pose cardinality is checked before individual landmarks so a second person
/// can never be hidden by evaluating only the first detected pose.
CaptureQuality evaluateCaptureQuality(
  List<PoseLandmark> landmarks, {
  required int poseCount,
  required CaptureAssetRole role,
  required double imageWidth,
  required double imageHeight,
  required double? tiltDegrees,
}) {
  if (poseCount <= 0 || landmarks.isEmpty) {
    return const CaptureQuality.blocked(CaptureIssue.noPose);
  }
  if (poseCount > 1) {
    return const CaptureQuality.blocked(CaptureIssue.multiplePoses);
  }
  final lm = {for (final landmark in landmarks) landmark.type: landmark};

  final nose = lm[PoseLandmarkType.nose];
  final leftEye = lm[PoseLandmarkType.leftEye];
  final rightEye = lm[PoseLandmarkType.rightEye];

  double? headTopY;
  if (nose != null && leftEye != null && rightEye != null) {
    headTopY = PoseService.estimateHeadTopY(
      noseY: nose.y,
      leftEyeY: leftEye.y,
      rightEyeY: rightEye.y,
      leftEarY: lm[PoseLandmarkType.leftEar]?.y,
      rightEarY: lm[PoseLandmarkType.rightEar]?.y,
    );
  }

  final heels = [
    lm[PoseLandmarkType.leftHeel],
    lm[PoseLandmarkType.rightHeel],
  ].whereType<PoseLandmark>();
  double? heelY;
  for (final heel in heels) {
    heelY = heelY == null || heel.y > heelY ? heel.y : heelY;
  }

  final bodyTopY = headTopY ?? nose?.y;
  final bodySpan = bodyTopY != null && heelY != null
      ? math.max(heelY - bodyTopY, 1.0)
      : null;
  final shoulderWidth = _pairWidth(
    lm[PoseLandmarkType.leftShoulder],
    lm[PoseLandmarkType.rightShoulder],
  );
  final hipWidth = _pairWidth(
    lm[PoseLandmarkType.leftHip],
    lm[PoseLandmarkType.rightHip],
  );
  final visibleWidth = shoulderWidth == null
      ? hipWidth
      : hipWidth == null
          ? shoulderWidth
          : math.max(shoulderWidth, hipWidth);
  final widthToBody =
      visibleWidth != null && bodySpan != null ? visibleWidth / bodySpan : null;
  final isProfileRole =
      role == CaptureAssetRole.side || role == CaptureAssetRole.armSide;
  final orientationScore = _orientationScore(widthToBody, isProfileRole);

  if (widthToBody != null) {
    if (isProfileRole && widthToBody > captureSideMaxWidthToBodyFraction) {
      return CaptureQuality.blocked(
        CaptureIssue.wrongOrientation,
        orientationScore: orientationScore,
      );
    }
    if (!isProfileRole && widthToBody < captureFrontMinWidthToBodyFraction) {
      return CaptureQuality.blocked(
        CaptureIssue.wrongOrientation,
        orientationScore: orientationScore,
      );
    }
  }

  final edgeMargin = captureEdgeMarginFraction * imageHeight;
  if (headTopY != null && headTopY < edgeMargin) {
    return CaptureQuality.blocked(
      CaptureIssue.cutOffTop,
      orientationScore: orientationScore,
    );
  }
  if (heelY != null && heelY > imageHeight - edgeMargin) {
    return CaptureQuality.blocked(
      CaptureIssue.cutOffBottom,
      orientationScore: orientationScore,
    );
  }

  final requiredLandmarks = _requiredLandmarks(lm, role);
  if (requiredLandmarks == null) {
    return CaptureQuality.blocked(
      CaptureIssue.missingRequiredLandmark,
      orientationScore: orientationScore,
    );
  }

  var coverageScore = 0.0;
  if (headTopY != null && heelY != null) {
    final bodyFraction = (heelY - headTopY) / imageHeight;
    final bodyCoverageScore = ((bodyFraction - captureMinBodyCoverageFraction) /
            (captureTargetBodyCoverageFraction -
                captureMinBodyCoverageFraction))
        .clamp(0.0, 1.0);
    if (bodyFraction < captureMinBodyCoverageFraction) {
      return CaptureQuality.blocked(
        CaptureIssue.tooFar,
        coverageScore: bodyCoverageScore,
        orientationScore: orientationScore,
      );
    }
    coverageScore = bodyCoverageScore;
  }

  final torsoXs = [
    lm[PoseLandmarkType.leftShoulder],
    lm[PoseLandmarkType.rightShoulder],
    lm[PoseLandmarkType.leftHip],
    lm[PoseLandmarkType.rightHip],
  ].whereType<PoseLandmark>().map((landmark) => landmark.x).toList();
  if (torsoXs.isNotEmpty) {
    final midX = torsoXs.reduce((left, right) => left + right) / torsoXs.length;
    final offset = (midX - imageWidth / 2).abs();
    final maxOffset = captureMaxCenterOffsetFraction * imageWidth;
    final centerScore = (1 - offset / maxOffset).clamp(0.0, 1.0);
    coverageScore = math.min(coverageScore, centerScore);
    if (offset > maxOffset) {
      return CaptureQuality.blocked(
        CaptureIssue.offCenter,
        coverageScore: coverageScore,
        orientationScore: orientationScore,
      );
    }
  }

  final poseScore = requiredLandmarks
          .map((landmark) => landmark.likelihood.clamp(0.0, 1.0))
          .reduce((left, right) => left + right) /
      requiredLandmarks.length;
  if (requiredLandmarks.any(
    (landmark) => landmark.likelihood < captureMinLandmarkLikelihood,
  )) {
    return CaptureQuality.blocked(
      CaptureIssue.lowVisibility,
      poseScore: poseScore,
      coverageScore: coverageScore,
      orientationScore: orientationScore,
    );
  }

  if (tiltDegrees != null && tiltDegrees.abs() > captureMaxTiltDegrees) {
    return CaptureQuality.blocked(
      CaptureIssue.excessiveTilt,
      poseScore: poseScore,
      coverageScore: coverageScore,
      orientationScore: orientationScore,
    );
  }

  return CaptureQuality.accepted(
    poseScore: poseScore,
    coverageScore: coverageScore,
    orientationScore: orientationScore,
  );
}

double? _pairWidth(PoseLandmark? left, PoseLandmark? right) {
  if (left == null || right == null) return null;
  return (left.x - right.x).abs();
}

double _orientationScore(double? widthToBody, bool isProfileRole) {
  if (widthToBody == null) return 0;
  if (isProfileRole) {
    return ((captureSideMaxWidthToBodyFraction - widthToBody) /
            (captureSideMaxWidthToBodyFraction -
                captureSideTargetWidthToBodyFraction))
        .clamp(0.0, 1.0);
  }
  return ((widthToBody - captureFrontMinWidthToBodyFraction) /
          (captureFrontTargetWidthToBodyFraction -
              captureFrontMinWidthToBodyFraction))
      .clamp(0.0, 1.0);
}

List<PoseLandmark>? _requiredLandmarks(
  Map<PoseLandmarkType, PoseLandmark> landmarks,
  CaptureAssetRole role,
) {
  PoseLandmark? at(PoseLandmarkType type) => landmarks[type];

  final head = [
    at(PoseLandmarkType.nose),
    at(PoseLandmarkType.leftEye),
    at(PoseLandmarkType.rightEye),
  ];
  if (head.any((landmark) => landmark == null)) return null;

  final isProfileRole =
      role == CaptureAssetRole.side || role == CaptureAssetRole.armSide;
  if (isProfileRole) {
    final leftChain = [
      at(PoseLandmarkType.leftShoulder),
      at(PoseLandmarkType.leftHip),
      at(PoseLandmarkType.leftHeel),
    ];
    final rightChain = [
      at(PoseLandmarkType.rightShoulder),
      at(PoseLandmarkType.rightHip),
      at(PoseLandmarkType.rightHeel),
    ];
    final chain = leftChain.every((landmark) => landmark != null)
        ? leftChain
        : rightChain.every((landmark) => landmark != null)
            ? rightChain
            : null;
    if (chain == null) return null;
    return [
      ...head.whereType<PoseLandmark>(),
      ...chain.whereType<PoseLandmark>(),
    ];
  }

  final body = [
    at(PoseLandmarkType.leftShoulder),
    at(PoseLandmarkType.rightShoulder),
    at(PoseLandmarkType.leftHip),
    at(PoseLandmarkType.rightHip),
    at(PoseLandmarkType.leftHeel),
    at(PoseLandmarkType.rightHeel),
  ];
  if (body.any((landmark) => landmark == null)) return null;
  return [
    ...head.whereType<PoseLandmark>(),
    ...body.whereType<PoseLandmark>(),
  ];
}

/// Debounces auto-capture and fires once per stable quality streak.
class AutoCaptureGate {
  AutoCaptureGate({this.requiredGoodFrames = 8});

  final int requiredGoodFrames;
  int _streak = 0;

  double get progress => (_streak / requiredGoodFrames).clamp(0.0, 1.0);

  bool onFrame(bool good) {
    if (!good) {
      _streak = 0;
      return false;
    }
    _streak++;
    if (_streak >= requiredGoodFrames) {
      _streak = 0;
      return true;
    }
    return false;
  }

  void reset() => _streak = 0;
}
