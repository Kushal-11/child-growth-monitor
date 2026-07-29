import 'package:flutter_test/flutter_test.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/services/capture_quality.dart';

/// Frame size used across tests (portrait 720x1280).
const double w = 720;
const double h = 1280;

PoseLandmark lm(PoseLandmarkType type, double x, double y,
    {double likelihood = 0.9}) {
  return PoseLandmark(type: type, x: x, y: y, z: 0, likelihood: likelihood);
}

/// A well-framed standing child:
/// head top ≈ y=170, heels at y=1150 → body fills ~77% of frame, centered.
List<PoseLandmark> goodPose({
  double dx = 0,
  double dy = 0,
  double shoulderHalfWidth = 80,
  double hipHalfWidth = 60,
  double heelLikelihood = 0.9,
  double heelY = 1150,
}) {
  return [
    lm(PoseLandmarkType.nose, 360 + dx, 220 + dy),
    lm(PoseLandmarkType.leftEye, 380 + dx, 200 + dy),
    lm(PoseLandmarkType.rightEye, 340 + dx, 200 + dy),
    lm(PoseLandmarkType.leftShoulder, 360 + shoulderHalfWidth + dx, 350 + dy),
    lm(PoseLandmarkType.rightShoulder, 360 - shoulderHalfWidth + dx, 350 + dy),
    lm(PoseLandmarkType.leftHip, 360 + hipHalfWidth + dx, 600 + dy),
    lm(PoseLandmarkType.rightHip, 360 - hipHalfWidth + dx, 600 + dy),
    lm(PoseLandmarkType.leftHeel, 390 + dx, heelY + dy,
        likelihood: heelLikelihood),
    lm(PoseLandmarkType.rightHeel, 330 + dx, heelY + dy,
        likelihood: heelLikelihood),
  ];
}

CaptureQuality evaluate(
  List<PoseLandmark> landmarks, {
  int poseCount = 1,
  CaptureAssetRole role = CaptureAssetRole.front,
  double? tiltDegrees,
}) {
  return evaluateCaptureQuality(
    landmarks,
    poseCount: poseCount,
    role: role,
    imageWidth: w,
    imageHeight: h,
    tiltDegrees: tiltDegrees,
  );
}

void main() {
  group('evaluateCaptureQuality', () {
    test('reports noPose when no landmarks detected', () {
      final q = evaluate(const [], poseCount: 0);
      expect(q.ready, isFalse);
      expect(q.issue, CaptureIssue.noPose);
    });

    test('reports multiplePoses before evaluating landmarks', () {
      final q = evaluate(goodPose(), poseCount: 2);
      expect(q.ready, isFalse);
      expect(q.issue, CaptureIssue.multiplePoses);
    });

    test('is ready for a well-framed, fully visible pose', () {
      final q = evaluate(goodPose());
      expect(q.issue, isNull);
      expect(q.ready, isTrue);
      expect(q.poseScore, inInclusiveRange(0, 1));
      expect(q.coverageScore, inInclusiveRange(0, 1));
      expect(q.orientationScore, inInclusiveRange(0, 1));
      expect(q.overallScore, inInclusiveRange(0, 1));
    });

    test('rejects a narrow profile pose for the front role', () {
      final q = evaluate(
        goodPose(shoulderHalfWidth: 12, hipHalfWidth: 10),
      );
      expect(q.issue, CaptureIssue.wrongOrientation);
    });

    test('accepts a narrow profile pose for the side role', () {
      final q = evaluate(
        goodPose(shoulderHalfWidth: 12, hipHalfWidth: 10),
        role: CaptureAssetRole.side,
      );
      expect(q.ready, isTrue);
    });

    test('rejects a frontal pose for the side role', () {
      final q = evaluate(goodPose(), role: CaptureAssetRole.side);
      expect(q.issue, CaptureIssue.wrongOrientation);
    });

    test('reports cutOffBottom when heels are inferred beyond bottom edge', () {
      // ML Kit infers out-of-frame landmarks with low likelihood; their
      // coordinates land outside the image. That's a framing problem, not a
      // generic visibility problem.
      final q = evaluate(
        goodPose(heelY: 1290, heelLikelihood: 0.3),
      );
      expect(q.ready, isFalse);
      expect(q.issue, CaptureIssue.cutOffBottom);
    });

    test('reports cutOffTop when estimated head top leaves the frame', () {
      // Eyes at y=30, nose at y=50 → estimated head top = 50 - 20*2.5 = 0.
      final q = evaluate(goodPose(dy: -170));
      expect(q.ready, isFalse);
      expect(q.issue, CaptureIssue.cutOffTop);
    });

    test('reports missingRequiredLandmark for incomplete front anatomy', () {
      final pose = goodPose()
          .where((landmark) => landmark.type != PoseLandmarkType.rightHip)
          .toList();
      final q = evaluate(pose);
      expect(q.issue, CaptureIssue.missingRequiredLandmark);
    });

    test('reports lowVisibility when a key landmark is in-frame but unclear',
        () {
      final q = evaluate(goodPose(heelLikelihood: 0.2));
      expect(q.ready, isFalse);
      expect(q.issue, CaptureIssue.lowVisibility);
    });

    test('reports tooFar when the body fills too little of the frame', () {
      // Head top ≈ 470, heels at 700+300=1000 → 530px of 1280 ≈ 41% < 50%.
      final q = evaluate(goodPose(dy: 300, heelY: 700));
      expect(q.ready, isFalse);
      expect(q.issue, CaptureIssue.tooFar);
    });

    test('coverage instruction wins over low landmark confidence', () {
      final q = evaluate(
        goodPose(dy: 300, heelY: 700, heelLikelihood: 0.2),
      );
      expect(q.issue, CaptureIssue.tooFar);
    });

    test('reports offCenter when the torso midline drifts sideways', () {
      // Torso mid-x shifts from 360 to 560; offset 200 > 20% of width (144).
      final q = evaluate(goodPose(dx: 200));
      expect(q.ready, isFalse);
      expect(q.issue, CaptureIssue.offCenter);
    });

    test('rejects excessive phone tilt when sensor data exists', () {
      final q = evaluate(goodPose(), tiltDegrees: 18);
      expect(q.issue, CaptureIssue.excessiveTilt);
    });

    test('missing tilt data does not block an otherwise good frame', () {
      final q = evaluate(goodPose(), tiltDegrees: null);
      expect(q.ready, isTrue);
    });

    test('returns only the highest-priority actionable issue', () {
      final q = evaluate(
        goodPose(dx: 250, heelY: 1290),
        poseCount: 2,
        tiltDegrees: 30,
      );
      expect(q.issue, CaptureIssue.multiplePoses);
    });
  });

  group('AutoCaptureGate', () {
    test('fires only after the required number of consecutive good frames', () {
      final gate = AutoCaptureGate(requiredGoodFrames: 3);
      expect(gate.onFrame(true), isFalse);
      expect(gate.onFrame(true), isFalse);
      expect(gate.onFrame(true), isTrue);
    });

    test('a bad frame resets the streak', () {
      final gate = AutoCaptureGate(requiredGoodFrames: 3);
      gate.onFrame(true);
      gate.onFrame(true);
      expect(gate.onFrame(false), isFalse);
      expect(gate.onFrame(true), isFalse);
      expect(gate.onFrame(true), isFalse);
      expect(gate.onFrame(true), isTrue);
    });

    test('after firing, a fresh streak is required to fire again', () {
      final gate = AutoCaptureGate(requiredGoodFrames: 2);
      gate.onFrame(true);
      expect(gate.onFrame(true), isTrue);
      expect(gate.onFrame(true), isFalse);
      expect(gate.onFrame(true), isTrue);
    });

    test('progress reflects the current streak fraction', () {
      final gate = AutoCaptureGate(requiredGoodFrames: 4);
      expect(gate.progress, 0);
      gate.onFrame(true);
      expect(gate.progress, closeTo(0.25, 1e-9));
      gate.onFrame(false);
      expect(gate.progress, 0);
    });
  });
}
