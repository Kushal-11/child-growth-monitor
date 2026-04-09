import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/pose_service.dart';

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
}
