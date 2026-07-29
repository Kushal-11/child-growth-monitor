import '../../../services/capture_quality.dart';
import '../domain/capture_thresholds.dart';
import 'frame_quality_service.dart';

class BurstFrameCandidate<T> {
  const BurstFrameCandidate({
    required this.value,
    required this.captureIndex,
    required this.liveQuality,
    required this.stillQuality,
  });

  final T value;
  final int captureIndex;
  final CaptureQuality liveQuality;
  final FrameQualityResult stillQuality;
}

class RankedBurstFrame<T> {
  const RankedBurstFrame({
    required this.value,
    required this.captureIndex,
    required this.rank,
    required this.overallScore,
    required this.liveQuality,
    required this.stillQuality,
  });

  final T value;
  final int captureIndex;
  final int rank;
  final double overallScore;
  final CaptureQuality liveQuality;
  final FrameQualityResult stillQuality;
}

/// Selects only quality-approved frames using a stable, versioned ordering.
class BurstFrameRanker {
  const BurstFrameRanker();

  List<RankedBurstFrame<T>> rank<T>(List<BurstFrameCandidate<T>> frames) {
    final accepted = frames
        .where(
          (frame) => frame.liveQuality.ready && frame.stillQuality.accepted,
        )
        .map((frame) {
          final overall =
              captureLiveScoreWeight * frame.liveQuality.overallScore +
                  captureStillScoreWeight * frame.stillQuality.overallScore;
          return (frame: frame, overall: overall.clamp(0.0, 1.0));
        })
        .where((item) => item.overall >= captureMinBurstOverallScore)
        .toList()
      ..sort((left, right) {
        final qualityOrder = right.overall.compareTo(left.overall);
        if (qualityOrder != 0) return qualityOrder;
        return left.frame.captureIndex.compareTo(right.frame.captureIndex);
      });

    return List.unmodifiable([
      for (var index = 0; index < accepted.length; index++)
        RankedBurstFrame(
          value: accepted[index].frame.value,
          captureIndex: accepted[index].frame.captureIndex,
          rank: index + 1,
          overallScore: accepted[index].overall,
          liveQuality: accepted[index].frame.liveQuality,
          stillQuality: accepted[index].frame.stillQuality,
        ),
    ]);
  }
}
