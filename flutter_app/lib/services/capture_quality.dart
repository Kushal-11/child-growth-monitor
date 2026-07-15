import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import 'pose_service.dart';

/// Why a live frame is not yet good enough to capture, in priority order.
/// Each issue maps to one actionable on-screen instruction.
enum CaptureIssue {
  /// No pose detected in the frame at all.
  noPose,

  /// The estimated head top is at/beyond the top edge of the frame.
  cutOffTop,

  /// The heels are at/beyond the bottom edge of the frame.
  cutOffBottom,

  /// A key landmark is inside the frame but below the likelihood threshold
  /// (e.g. occluded or blurred).
  lowVisibility,

  /// The body is fully visible but fills too little of the frame for a
  /// reliable pixel-ratio measurement.
  tooFar,

  /// The torso midline is too far from the horizontal center.
  offCenter,
}

/// Result of evaluating one live camera frame against the capture gate.
class CaptureQuality {
  const CaptureQuality._(this.issue);

  const CaptureQuality.ok() : this._(null);
  const CaptureQuality.blocked(CaptureIssue issue) : this._(issue);

  /// The highest-priority framing problem, or null when the frame is good.
  final CaptureIssue? issue;

  bool get ready => issue == null;
}

/// Minimum ML Kit likelihood for a key landmark to count as clearly visible.
/// Matches the threshold PoseService uses for segment extraction.
const double captureMinLandmarkLikelihood = 0.5;

/// Head top / heels must stay this fraction of the frame away from the edge.
const double captureEdgeMarginFrac = 0.02;

/// Head-to-heel span must cover at least this fraction of the frame height.
const double captureMinBodyFrac = 0.5;

/// Torso midline may deviate at most this fraction of the frame width from
/// the horizontal center.
const double captureMaxCenterOffsetFrac = 0.20;

/// Evaluate a single frame's pose landmarks (pixel coordinates in the upright
/// frame) against the live capture gate. Pure — unit-testable without ML Kit.
///
/// Coordinate-based cut-off checks run BEFORE the likelihood check: ML Kit
/// infers out-of-frame landmarks with low likelihood but out-of-bounds
/// coordinates, and "move back" is more actionable than "body not visible".
CaptureQuality evaluateCaptureQuality(
  List<PoseLandmark> landmarks, {
  required double imageWidth,
  required double imageHeight,
}) {
  if (landmarks.isEmpty) {
    return const CaptureQuality.blocked(CaptureIssue.noPose);
  }
  final lm = {for (final l in landmarks) l.type: l};

  final nose = lm[PoseLandmarkType.nose];
  final leftEye = lm[PoseLandmarkType.leftEye];
  final rightEye = lm[PoseLandmarkType.rightEye];

  double? headTopY;
  if (nose != null && leftEye != null && rightEye != null) {
    final leftEar = lm[PoseLandmarkType.leftEar];
    final rightEar = lm[PoseLandmarkType.rightEar];
    headTopY = PoseService.estimateHeadTopY(
      noseY: nose.y,
      leftEyeY: leftEye.y,
      rightEyeY: rightEye.y,
      leftEarY: leftEar?.y,
      rightEarY: rightEar?.y,
    );
  }

  final edgeMargin = captureEdgeMarginFrac * imageHeight;
  if (headTopY != null && headTopY < edgeMargin) {
    return const CaptureQuality.blocked(CaptureIssue.cutOffTop);
  }

  final heels = [
    lm[PoseLandmarkType.leftHeel],
    lm[PoseLandmarkType.rightHeel],
  ].whereType<PoseLandmark>().toList();
  double? heelY;
  for (final heel in heels) {
    heelY = heelY == null || heel.y > heelY ? heel.y : heelY;
  }
  if (heelY != null && heelY > imageHeight - edgeMargin) {
    return const CaptureQuality.blocked(CaptureIssue.cutOffBottom);
  }

  // Same seven key landmarks PoseService.computeConfidence averages.
  final keyLandmarks = [
    nose,
    lm[PoseLandmarkType.leftShoulder],
    lm[PoseLandmarkType.rightShoulder],
    lm[PoseLandmarkType.leftHip],
    lm[PoseLandmarkType.rightHip],
    lm[PoseLandmarkType.leftHeel],
    lm[PoseLandmarkType.rightHeel],
  ];
  final allClear = keyLandmarks.every(
    (l) => l != null && l.likelihood >= captureMinLandmarkLikelihood,
  );
  if (!allClear) {
    return const CaptureQuality.blocked(CaptureIssue.lowVisibility);
  }

  if (headTopY != null && heelY != null) {
    final bodyFrac = (heelY - headTopY) / imageHeight;
    if (bodyFrac < captureMinBodyFrac) {
      return const CaptureQuality.blocked(CaptureIssue.tooFar);
    }
  }

  final torsoXs = [
    lm[PoseLandmarkType.leftShoulder],
    lm[PoseLandmarkType.rightShoulder],
    lm[PoseLandmarkType.leftHip],
    lm[PoseLandmarkType.rightHip],
  ].whereType<PoseLandmark>().map((l) => l.x).toList();
  if (torsoXs.isNotEmpty) {
    final midX = torsoXs.reduce((a, b) => a + b) / torsoXs.length;
    final offset = (midX - imageWidth / 2).abs();
    if (offset > captureMaxCenterOffsetFrac * imageWidth) {
      return const CaptureQuality.blocked(CaptureIssue.offCenter);
    }
  }

  return const CaptureQuality.ok();
}

/// Debounces auto-capture: fires once after [requiredGoodFrames] consecutive
/// good frames, then requires a fresh streak. Pure state machine — no timers.
class AutoCaptureGate {
  AutoCaptureGate({this.requiredGoodFrames = 8});

  final int requiredGoodFrames;
  int _streak = 0;

  /// Current streak as a 0..1 fraction, for a progress ring in the UI.
  double get progress => (_streak / requiredGoodFrames).clamp(0.0, 1.0);

  /// Feed one frame's gate verdict; returns true when capture should fire.
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
