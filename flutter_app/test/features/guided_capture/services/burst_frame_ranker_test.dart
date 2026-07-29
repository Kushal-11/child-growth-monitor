import 'package:child_growth_monitor_app/features/guided_capture/services/burst_frame_ranker.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/frame_quality_service.dart';
import 'package:child_growth_monitor_app/services/capture_quality.dart';
import 'package:flutter_test/flutter_test.dart';

CaptureQuality live(double overall) => CaptureQuality.accepted(
      poseScore: overall,
      coverageScore: overall,
      orientationScore: overall,
    );

FrameQualityResult still(double overall, {bool accepted = true}) =>
    FrameQualityResult(
      brightnessScore: overall,
      contrastScore: overall,
      lightingScore: overall,
      sharpnessScore: overall,
      overallScore: overall,
      accepted: accepted,
      rejectionReason: accepted ? null : StillQualityIssue.blurred,
    );

void main() {
  const ranker = BurstFrameRanker();

  test('orders accepted frames by deterministic combined quality', () {
    final ranked = ranker.rank([
      BurstFrameCandidate(
        value: 'middle',
        captureIndex: 0,
        liveQuality: live(0.7),
        stillQuality: still(0.7),
      ),
      BurstFrameCandidate(
        value: 'best',
        captureIndex: 1,
        liveQuality: live(0.9),
        stillQuality: still(0.9),
      ),
      BurstFrameCandidate(
        value: 'rejected',
        captureIndex: 2,
        liveQuality: live(0.95),
        stillQuality: still(0.95, accepted: false),
      ),
    ]);

    expect(ranked.map((frame) => frame.value), ['best', 'middle']);
    expect(ranked.map((frame) => frame.rank), [1, 2]);
  });

  test('uses capture index as the explicit stable tie-breaker', () {
    final input = [
      BurstFrameCandidate(
        value: 'later',
        captureIndex: 2,
        liveQuality: live(0.8),
        stillQuality: still(0.8),
      ),
      BurstFrameCandidate(
        value: 'earlier',
        captureIndex: 1,
        liveQuality: live(0.8),
        stillQuality: still(0.8),
      ),
    ];

    expect(
      ranker.rank(input).map((frame) => frame.value),
      ['earlier', 'later'],
    );
    expect(
      ranker.rank(input).map((frame) => frame.value),
      ranker.rank(input).map((frame) => frame.value),
    );
  });

  test('rejects the whole burst when no frame passes minimum quality', () {
    final ranked = ranker.rank([
      BurstFrameCandidate(
        value: 'bad-live',
        captureIndex: 0,
        liveQuality: const CaptureQuality.blocked(CaptureIssue.noPose),
        stillQuality: still(0.9),
      ),
      BurstFrameCandidate(
        value: 'bad-still',
        captureIndex: 1,
        liveQuality: live(0.9),
        stillQuality: still(0.3, accepted: false),
      ),
    ]);

    expect(ranked, isEmpty);
  });
}
