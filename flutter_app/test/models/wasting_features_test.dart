import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/models/wasting_features.dart';

void main() {
  test('toArray returns 14-element vector in correct order', () {
    final f = WastingFeatures(
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
    expect(arr[1], 1.0);  // sex_binary
    expect(arr[2], 85.0); // height_cm
    expect(arr[10], 10.0); // chest_depth_cm (provided)
    expect(arr[12], closeTo(10.0 / 85.0, 0.001)); // chest_depth_ratio
  });

  test('toArray imputes AP depth when not provided', () {
    final f = WastingFeatures(
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
}
