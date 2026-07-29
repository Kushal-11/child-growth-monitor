import 'package:child_growth_monitor_app/features/guided_capture/services/frame_quality_service.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

img.Image checkerboard(int low, int high, {int size = 32}) {
  final image = img.Image(width: size, height: size);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final value = (x + y).isEven ? low : high;
      image.setPixelRgb(x, y, value, value, value);
    }
  }
  return image;
}

img.Image smoothGradient({int size = 32}) {
  final image = img.Image(width: size, height: size);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final value = 55 + ((140 * x) / (size - 1)).round();
      image.setPixelRgb(x, y, value, value, value);
    }
  }
  return image;
}

void main() {
  const service = FrameQualityService();

  test('rejects a dark image with an actionable reason', () {
    final result = service.evaluateImage(checkerboard(4, 24));
    expect(result.accepted, isFalse);
    expect(result.rejectionReason, StillQualityIssue.tooDark);
  });

  test('rejects an overexposed image', () {
    final result = service.evaluateImage(checkerboard(238, 255));
    expect(result.accepted, isFalse);
    expect(result.rejectionReason, StillQualityIssue.overexposed);
  });

  test('rejects low contrast even when mean luminance is acceptable', () {
    final result = service.evaluateImage(checkerboard(122, 130));
    expect(result.accepted, isFalse);
    expect(result.rejectionReason, StillQualityIssue.lowContrast);
  });

  test('rejects a smooth blurred gradient', () {
    final result = service.evaluateImage(smoothGradient());
    expect(result.accepted, isFalse);
    expect(result.rejectionReason, StillQualityIssue.blurred);
  });

  test('accepts an evenly lit sharp edge pattern', () {
    final result = service.evaluateImage(checkerboard(45, 210));
    expect(result.accepted, isTrue);
    expect(result.rejectionReason, isNull);
  });

  test('scores are normalized and deterministic', () {
    final image = checkerboard(45, 210);
    final first = service.evaluateImage(image);
    final second = service.evaluateImage(image);

    for (final score in [
      first.brightnessScore,
      first.contrastScore,
      first.lightingScore,
      first.sharpnessScore,
      first.overallScore,
    ]) {
      expect(score, inInclusiveRange(0, 1));
    }
    expect(second, first);
  });
}
