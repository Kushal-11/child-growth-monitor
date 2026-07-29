import 'dart:convert';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('feature_scaler.json has 14-feature mean and scale arrays', () async {
    final jsonStr =
        await rootBundle.loadString('assets/models/feature_scaler.json');
    final data = jsonDecode(jsonStr) as Map<String, dynamic>;
    expect((data['mean'] as List).length, 14);
    expect((data['scale'] as List).length, 14);
  });

  test('weight_estimator.tflite is bundled and non-trivial', () async {
    final bytes =
        await rootBundle.load('assets/models/weight_estimator.tflite');
    expect(bytes.lengthInBytes, greaterThan(2000));
  });

  test('wasting_classifier.tflite is bundled and non-trivial', () async {
    final bytes =
        await rootBundle.load('assets/models/wasting_classifier.tflite');
    expect(bytes.lengthInBytes, greaterThan(10000));
  });

  test('camera model manifest is research-only and hash-binds every asset',
      () async {
    final manifestBytes =
        await rootBundle.load('assets/models/model_manifest.json');
    final manifest = jsonDecode(
      utf8.decode(
        manifestBytes.buffer.asUint8List(
          manifestBytes.offsetInBytes,
          manifestBytes.lengthInBytes,
        ),
      ),
    ) as Map<String, dynamic>;

    expect(manifest['non_clinical'], isTrue);
    expect(manifest['training_data_label'], 'synthetic_who_research_only');
    final files = manifest['files'] as Map<String, dynamic>;
    for (final fileName in [
      'weight_estimator.tflite',
      'wasting_classifier.tflite',
      'feature_scaler.json',
    ]) {
      final data = await rootBundle.load('assets/models/$fileName');
      final bytes = data.buffer.asUint8List(
        data.offsetInBytes,
        data.lengthInBytes,
      );
      final record = files[fileName] as Map<String, dynamic>;
      expect(bytes.length, record['size_bytes'], reason: fileName);
      expect(sha256.convert(bytes).toString(), record['sha256'],
          reason: fileName);
    }
  });
}
