import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../domain/capture_thresholds.dart';

enum StillQualityIssue {
  decodeFailed,
  tooDark,
  overexposed,
  lowContrast,
  blurred,
}

class FrameQualityResult {
  const FrameQualityResult({
    required this.brightnessScore,
    required this.contrastScore,
    required this.lightingScore,
    required this.sharpnessScore,
    required this.overallScore,
    required this.accepted,
    required this.rejectionReason,
  });

  const FrameQualityResult.decodeFailed()
      : brightnessScore = 0,
        contrastScore = 0,
        lightingScore = 0,
        sharpnessScore = 0,
        overallScore = 0,
        accepted = false,
        rejectionReason = StillQualityIssue.decodeFailed;

  final double brightnessScore;
  final double contrastScore;
  final double lightingScore;
  final double sharpnessScore;
  final double overallScore;
  final bool accepted;
  final StillQualityIssue? rejectionReason;

  @override
  bool operator ==(Object other) =>
      other is FrameQualityResult &&
      other.brightnessScore == brightnessScore &&
      other.contrastScore == contrastScore &&
      other.lightingScore == lightingScore &&
      other.sharpnessScore == sharpnessScore &&
      other.overallScore == overallScore &&
      other.accepted == accepted &&
      other.rejectionReason == rejectionReason;

  @override
  int get hashCode => Object.hash(
        brightnessScore,
        contrastScore,
        lightingScore,
        sharpnessScore,
        overallScore,
        accepted,
        rejectionReason,
      );
}

/// Deterministic post-capture lighting, contrast, and sharpness evaluation.
class FrameQualityService {
  const FrameQualityService();

  Future<FrameQualityResult> evaluateFile(String path) async {
    try {
      return evaluateBytes(await File(path).readAsBytes());
    } on FileSystemException {
      return const FrameQualityResult.decodeFailed();
    }
  }

  FrameQualityResult evaluateBytes(Uint8List bytes) {
    final image = img.decodeImage(bytes);
    if (image == null) return const FrameQualityResult.decodeFailed();
    return evaluateImage(image);
  }

  FrameQualityResult evaluateImage(img.Image image) {
    if (image.width < 3 || image.height < 3) {
      return const FrameQualityResult.decodeFailed();
    }

    final luminance = List<double>.filled(image.width * image.height, 0);
    var sum = 0.0;
    var index = 0;
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        final pixel = image.getPixel(x, y);
        final value =
            (0.2126 * pixel.r + 0.7152 * pixel.g + 0.0722 * pixel.b) / 255;
        luminance[index++] = value;
        sum += value;
      }
    }

    final mean = sum / luminance.length;
    var squaredDeviation = 0.0;
    for (final value in luminance) {
      final delta = value - mean;
      squaredDeviation += delta * delta;
    }
    final contrast = math.sqrt(squaredDeviation / luminance.length);

    var laplacianEnergy = 0.0;
    var interiorPixels = 0;
    for (var y = 1; y < image.height - 1; y++) {
      for (var x = 1; x < image.width - 1; x++) {
        final center = luminance[y * image.width + x];
        final laplacian = 4 * center -
            luminance[y * image.width + x - 1] -
            luminance[y * image.width + x + 1] -
            luminance[(y - 1) * image.width + x] -
            luminance[(y + 1) * image.width + x];
        laplacianEnergy += laplacian * laplacian;
        interiorPixels++;
      }
    }
    final sharpness = math.sqrt(laplacianEnergy / math.max(interiorPixels, 1));

    final brightnessScore = _brightnessScore(mean);
    final contrastScore = (contrast / captureTargetContrast).clamp(0.0, 1.0);
    final lightingScore = math.min(brightnessScore, contrastScore);
    final sharpnessScore = (sharpness / captureTargetSharpness).clamp(0.0, 1.0);
    final overallScore = (captureStillScoreWeight * sharpnessScore +
            (1 - captureStillScoreWeight) * lightingScore)
        .clamp(0.0, 1.0);

    final rejectionReason = switch ((mean, contrast, sharpness)) {
      (final value, _, _) when value < captureMinMeanLuminance =>
        StillQualityIssue.tooDark,
      (final value, _, _) when value > captureMaxMeanLuminance =>
        StillQualityIssue.overexposed,
      (_, final value, _) when value < captureMinContrast =>
        StillQualityIssue.lowContrast,
      (_, _, final value) when value < captureMinSharpness =>
        StillQualityIssue.blurred,
      _ => null,
    };

    return FrameQualityResult(
      brightnessScore: brightnessScore,
      contrastScore: contrastScore,
      lightingScore: lightingScore,
      sharpnessScore: sharpnessScore,
      overallScore: overallScore,
      accepted: rejectionReason == null,
      rejectionReason: rejectionReason,
    );
  }

  double _brightnessScore(double mean) {
    if (mean < captureMinMeanLuminance) {
      return (mean / captureMinMeanLuminance).clamp(0.0, 1.0);
    }
    if (mean > captureMaxMeanLuminance) {
      return ((1 - mean) / (1 - captureMaxMeanLuminance)).clamp(0.0, 1.0);
    }
    return 1;
  }
}
