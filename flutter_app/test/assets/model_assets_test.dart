import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('feature_scaler.json has 14-feature mean and scale arrays', () async {
    final jsonStr = await rootBundle.loadString('assets/models/feature_scaler.json');
    final data = jsonDecode(jsonStr) as Map<String, dynamic>;
    expect((data['mean'] as List).length, 14);
    expect((data['scale'] as List).length, 14);
  });

  test('weight_estimator.tflite is bundled and non-trivial', () async {
    final bytes = await rootBundle.load('assets/models/weight_estimator.tflite');
    expect(bytes.lengthInBytes, greaterThan(2000));
  });

  test('wasting_classifier.tflite is bundled and non-trivial', () async {
    final bytes = await rootBundle.load('assets/models/wasting_classifier.tflite');
    expect(bytes.lengthInBytes, greaterThan(10000));
  });
}
