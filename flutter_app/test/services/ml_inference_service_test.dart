import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/models/wasting_features.dart';
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final goldenFile = [
    File('../ml/runtime_golden_cases.json'),
    File('ml/runtime_golden_cases.json'),
  ].firstWhere((file) => file.existsSync());
  final golden =
      jsonDecode(goldenFile.readAsStringSync()) as Map<String, dynamic>;
  final runDeviceTflite =
      Platform.environment['CGM_RUN_DEVICE_TFLITE_TEST'] == '1';

  test('exact TFLite assets match shared inference goldens',
      skip: runDeviceTflite
          ? false
          : 'requires device TFLite runtime; set '
              'CGM_RUN_DEVICE_TFLITE_TEST=1 on an Android test device',
      () async {
    final svc = MlInferenceService();
    await svc.load();
    for (final dynamic rawCase in golden['inference_cases'] as List) {
      final testCase = rawCase as Map<String, dynamic>;
      final rawFeatures = testCase['features'] as Map<String, dynamic>;
      final expected = testCase['expected'] as Map<String, dynamic>;
      final prediction = svc.predict(WastingFeatures(
        ageMonths: (rawFeatures['age_months'] as num).toDouble(),
        sexBinary: rawFeatures['sex_binary'] as int,
        heightCm: (rawFeatures['height_cm'] as num).toDouble(),
        shoulderWidthCm: (rawFeatures['shoulder_width_cm'] as num).toDouble(),
        hipWidthCm: (rawFeatures['hip_width_cm'] as num).toDouble(),
        torsoLengthCm: (rawFeatures['torso_length_cm'] as num).toDouble(),
        upperArmLengthCm:
            (rawFeatures['upper_arm_length_cm'] as num).toDouble(),
        shoulderHeightRatio:
            (rawFeatures['shoulder_height_ratio'] as num).toDouble(),
        hipHeightRatio: (rawFeatures['hip_height_ratio'] as num).toDouble(),
        bodyBuildScore: rawFeatures['body_build_score'] as int,
        chestDepthCm: (rawFeatures['chest_depth_cm'] as num).toDouble(),
        abdDepthCm: (rawFeatures['abd_depth_cm'] as num).toDouble(),
      ));
      final expectedProbabilities = (expected['probabilities'] as List)
          .map((value) => (value as num).toDouble())
          .toList();
      final actualProbabilities = [
        prediction.mamProbability,
        prediction.normalProbability,
        prediction.overweightProbability,
        prediction.riskProbability,
        prediction.samProbability,
      ];
      expect(
        prediction.estimatedWeightKg,
        closeTo((expected['estimated_weight_kg'] as num).toDouble(), 1e-5),
        reason: testCase['name'] as String,
      );
      for (var index = 0; index < expectedProbabilities.length; index++) {
        expect(
          actualProbabilities[index],
          closeTo(expectedProbabilities[index], 1e-5),
          reason: testCase['name'] as String,
        );
      }
      expect(
        prediction.wastingStatus,
        expected['wasting_status'],
        reason: testCase['name'] as String,
      );
    }
    svc.dispose();
  });

  test('postprocessing matches shared Python/Dart golden cases', () {
    for (final dynamic rawCase in golden['postprocessing_cases'] as List) {
      final testCase = rawCase as Map<String, dynamic>;
      final probabilities = (testCase['probabilities'] as List)
          .map((value) => (value as num).toDouble())
          .toList();
      if (testCase['valid'] as bool) {
        final prediction = MlInferenceService.postprocessRawOutputs(
          estimatedWeightKg:
              (testCase['estimated_weight_kg'] as num).toDouble(),
          probabilities: probabilities,
        );
        expect(
          prediction.estimatedWeightKg,
          (testCase['estimated_weight_kg'] as num).toDouble(),
          reason: testCase['name'] as String,
        );
        expect(
          prediction.wastingStatus,
          testCase['expected_status'],
          reason: testCase['name'] as String,
        );
      } else {
        expect(
          () => MlInferenceService.postprocessRawOutputs(
            estimatedWeightKg:
                (testCase['estimated_weight_kg'] as num).toDouble(),
            probabilities: probabilities,
          ),
          throwsStateError,
          reason: testCase['name'] as String,
        );
      }
    }
  });

  test('postprocessing rejects non-finite model output', () {
    for (final invalid in [
      double.nan,
      double.infinity,
      double.negativeInfinity,
    ]) {
      expect(
        () => MlInferenceService.postprocessRawOutputs(
          estimatedWeightKg: invalid,
          probabilities: const [0.1, 0.2, 0.3, 0.15, 0.25],
        ),
        throwsStateError,
      );
      expect(
        () => MlInferenceService.postprocessRawOutputs(
          estimatedWeightKg: 9,
          probabilities: [0.1, 0.2, invalid, 0.35, 0.35],
        ),
        throwsStateError,
      );
    }
  });

  // weightWithinBounds is pure math — no native lib needed, runs on host.
  test('weight bound check rejects values outside 45–180% of WHO median', () {
    final svc = MlInferenceService();
    expect(
        svc.weightWithinBounds(predictedKg: 12.0, whoMedianKg: 12.0), isTrue);
    expect(
        svc.weightWithinBounds(predictedKg: 4.0, whoMedianKg: 12.0), isFalse);
    expect(
        svc.weightWithinBounds(predictedKg: 25.0, whoMedianKg: 12.0), isFalse);
    expect(
        svc.weightWithinBounds(predictedKg: 21.6, whoMedianKg: 12.0), isTrue);
    expect(svc.weightWithinBounds(predictedKg: 5.4, whoMedianKg: 12.0),
        isTrue); // 45% exact lower bound
    expect(svc.weightWithinBounds(predictedKg: 12.0, whoMedianKg: 0), isFalse);
    expect(svc.weightWithinBounds(predictedKg: 12.0, whoMedianKg: -1), isFalse);
    expect(
      svc.weightWithinBounds(
        predictedKg: double.nan,
        whoMedianKg: 12,
      ),
      isFalse,
    );
    expect(
      svc.weightWithinBounds(
        predictedKg: double.infinity,
        whoMedianKg: 12,
      ),
      isFalse,
    );
    svc.dispose();
  });

  test('throws StateError when predict called before load', () {
    final svc = MlInferenceService();
    expect(
      () => svc.predict(const WastingFeatures(
        ageMonths: 24,
        sexBinary: 1,
        heightCm: 87.1,
        shoulderWidthCm: 18,
        hipWidthCm: 15.5,
        torsoLengthCm: 26.5,
        upperArmLengthCm: 13.7,
        shoulderHeightRatio: 0.207,
        hipHeightRatio: 0.178,
        bodyBuildScore: 0,
      )),
      throwsStateError,
    );
  });
}
