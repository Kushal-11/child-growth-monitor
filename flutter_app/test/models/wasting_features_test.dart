import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/models/wasting_features.dart';

void main() {
  test('toArray returns 14-element vector in correct order', () {
    const f = WastingFeatures(
      ageMonths: 24.0,
      sexBinary: 1,
      heightCm: 85.0,
      shoulderWidthCm: 18.0,
      hipWidthCm: 15.0,
      torsoLengthCm: 25.5,
      upperArmLengthCm: 13.0,
      shoulderHeightRatio: 18.0 / 85.0,
      hipHeightRatio: 15.0 / 85.0,
      bodyBuildScore: 0,
      chestDepthCm: 10.0,
      abdDepthCm: 8.0,
    );
    final arr = f.toArray();
    expect(arr.length, 14);
    expect(arr[0], 24.0); // age_months
    expect(arr[1], 1.0); // sex_binary
    expect(arr[2], 85.0); // height_cm
    expect(arr[10], 10.0); // chest_depth_cm (provided)
    expect(arr[12], closeTo(10.0 / 85.0, 0.001)); // chest_depth_ratio
  });

  test('toArray imputes AP depth when not provided', () {
    const f = WastingFeatures(
      ageMonths: 24.0,
      sexBinary: 0,
      heightCm: 85.0,
      shoulderWidthCm: 18.0,
      hipWidthCm: 15.0,
      torsoLengthCm: 25.5,
      upperArmLengthCm: 13.0,
      shoulderHeightRatio: 18.0 / 85.0,
      hipHeightRatio: 15.0 / 85.0,
      bodyBuildScore: -1,
    );
    final arr = f.toArray();
    expect(arr[10], closeTo(18.0 * 0.45, 0.01)); // chest = shoulder * 0.45
    expect(arr[11], closeTo(15.0 * 0.50, 0.01)); // abd = hip * 0.50
  });

  test('feature vector matches the shared Python/mobile parity fixture',
      () async {
    final fixture = jsonDecode(
      await File('../tests/fixtures/wasting_features_parity.json')
          .readAsString(),
    ) as Map<String, dynamic>;
    final input = fixture['input'] as Map<String, dynamic>;
    final expected = (fixture['expected_vector'] as List<dynamic>)
        .map((value) => (value as num).toDouble())
        .toList(growable: false);
    final actual = WastingFeatures(
      ageMonths: (input['age_months'] as num).toDouble(),
      sexBinary: input['sex_binary'] as int,
      heightCm: (input['height_cm'] as num).toDouble(),
      shoulderWidthCm: (input['shoulder_width_cm'] as num).toDouble(),
      hipWidthCm: (input['hip_width_cm'] as num).toDouble(),
      torsoLengthCm: (input['torso_length_cm'] as num).toDouble(),
      upperArmLengthCm: (input['upper_arm_length_cm'] as num).toDouble(),
      shoulderHeightRatio:
          (input['shoulder_height_ratio'] as num).toDouble(),
      hipHeightRatio: (input['hip_height_ratio'] as num).toDouble(),
      bodyBuildScore: input['body_build_score'] as int,
      chestDepthCm: (input['chest_depth_cm'] as num).toDouble(),
      abdDepthCm: (input['abd_depth_cm'] as num).toDouble(),
    ).toArray();

    expect(actual, hasLength(expected.length));
    for (var index = 0; index < expected.length; index++) {
      expect(actual[index], closeTo(expected[index], 1e-6));
    }
  });
}
