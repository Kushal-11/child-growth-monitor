import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/models/wasting_features.dart';

void main() {
  test('feature expansion matches the shared Python/Flutter fixture', () {
    final cases = jsonDecode(
      File('../shared/ml_parity_cases.json').readAsStringSync(),
    ) as List<dynamic>;

    for (final rawCase in cases) {
      final testCase = rawCase as Map<String, dynamic>;
      final f = testCase['features'] as Map<String, dynamic>;
      final features = WastingFeatures(
        ageMonths: (f['age_months'] as num).toDouble(),
        sexBinary: f['sex_binary'] as int,
        heightCm: (f['height_cm'] as num).toDouble(),
        shoulderWidthCm: (f['shoulder_width_cm'] as num).toDouble(),
        hipWidthCm: (f['hip_width_cm'] as num).toDouble(),
        torsoLengthCm: (f['torso_length_cm'] as num).toDouble(),
        upperArmLengthCm: (f['upper_arm_length_cm'] as num).toDouble(),
        shoulderHeightRatio:
            (f['shoulder_height_ratio'] as num).toDouble(),
        hipHeightRatio: (f['hip_height_ratio'] as num).toDouble(),
        bodyBuildScore: f['body_build_score'] as int,
        chestDepthCm: (f['chest_depth_cm'] as num?)?.toDouble(),
        abdDepthCm: (f['abd_depth_cm'] as num?)?.toDouble(),
        chestDepthRatio: (f['chest_depth_ratio'] as num?)?.toDouble(),
        abdDepthRatio: (f['abd_depth_ratio'] as num?)?.toDouble(),
      );
      final expected = (testCase['expected_raw_features'] as List<dynamic>)
          .map((value) => (value as num).toDouble())
          .toList();
      final actual = features.toArray();
      expect(actual, hasLength(expected.length), reason: testCase['name'] as String);
      for (var index = 0; index < expected.length; index++) {
        expect(
          actual[index],
          closeTo(expected[index], 1e-5),
          reason: '${testCase['name']} feature $index',
        );
      }
    }
  });
}
