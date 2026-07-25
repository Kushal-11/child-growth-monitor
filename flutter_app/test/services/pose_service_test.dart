import 'package:child_growth_monitor_app/models/body_measurements.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/pose_service.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';

class _RawSidePose extends PoseService {
  double? receivedHeight;

  @override
  Future<List<PoseLandmark>> detectPose(String imagePath) async => [
        PoseLandmark(
          type: PoseLandmarkType.nose,
          x: 0.5,
          y: 0.1,
          z: 0,
          likelihood: 1,
        ),
      ];

  @override
  SideViewSegments extractSideSegments(
    List<PoseLandmark> landmarks, [
    double? heightCm,
  ]) {
    receivedHeight = heightCm;
    return const SideViewSegments(
      chestDepthPx: 0.1,
      abdDepthPx: 0.12,
      totalHeightPx: 0.8,
    );
  }
}

void main() {
  group('estimateHeadTopY', () {
    test('computes head top from nose and eyes', () {
      final y = PoseService.estimateHeadTopY(
        noseY: 100,
        leftEyeY: 90,
        rightEyeY: 90,
        leftEarY: null,
        rightEarY: null,
      );
      expect(y, closeTo(75.0, 0.1));
    });

    test('averages with ear method when ears visible', () {
      final y = PoseService.estimateHeadTopY(
        noseY: 100,
        leftEyeY: 90,
        rightEyeY: 90,
        leftEarY: 88,
        rightEarY: 88,
      );
      expect(y, closeTo(66.5, 0.1));
    });
  });

  group('estimateChinY', () {
    test('estimates from nose without mouth', () {
      expect(
        PoseService.estimateChinY(noseY: 100, noseToEye: 10, mouthY: null),
        closeTo(115, 0.1),
      );
    });

    test('uses mouth when available', () {
      expect(
        PoseService.estimateChinY(noseY: 100, noseToEye: 10, mouthY: 110),
        closeTo(115, 0.1),
      );
    });
  });

  test('side adapter returns raw segments without dummy height scaling',
      () async {
    final pose = _RawSidePose();
    final source = PoseServiceSource(pose);
    final segments = await source.sideSegmentsFor('side.jpg');
    expect(segments?.chestDepthPx, 0.1);
    expect(pose.receivedHeight, isNull);
  });
}
