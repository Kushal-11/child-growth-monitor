import 'dart:convert';

import 'package:child_growth_monitor_app/constants/config.dart';
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
    expect(data['feature_names'], featureNames);
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

  test('manifest versions and hashes every promoted runtime artifact',
      () async {
    final manifestJson =
        await rootBundle.loadString('assets/models/model_manifest.json');
    final manifest = jsonDecode(manifestJson) as Map<String, dynamic>;
    expect(manifest['model_version'], 'cgm-wasting-14f-synth-v1');
    expect(manifest['feature_schema_version'], 1);
    expect(manifest['feature_count'], 14);
    expect(manifest['feature_names'], featureNames);
    expect(manifest['labels'], wastingLabels);
    expect(manifest['training_data'], 'synthetic');
    expect(
      (manifest['evaluation'] as Map<String, dynamic>)['non_clinical'],
      isTrue,
    );
    final evaluation = manifest['evaluation'] as Map<String, dynamic>;
    expect(evaluation['engine'], 'tensorflow_lite');
    expect(evaluation['evaluation_contract_version'], 2);
    expect(evaluation['sam_sample_count'], greaterThan(0));
    expect(evaluation['invalid_prediction_count'], 0);
    expect(evaluation['sam_recall_floor_met'], isTrue);
    expect(
      (evaluation['dataset'] as Map<String, dynamic>)['sha256'],
      hasLength(64),
    );

    final artifacts = manifest['artifacts'] as Map<String, dynamic>;
    for (final filename in [
      'weight_estimator.tflite',
      'wasting_classifier.tflite',
      'feature_scaler.json',
      'label_encoder.json',
    ]) {
      final data = await rootBundle.load('assets/models/$filename');
      final bytes = data.buffer.asUint8List(
        data.offsetInBytes,
        data.lengthInBytes,
      );
      final record = artifacts[filename] as Map<String, dynamic>;
      expect(bytes.length, record['size_bytes']);
      expect(sha256.convert(bytes).toString(), record['sha256']);
      expect(
        (evaluation['evaluated_artifacts'] as Map<String, dynamic>)[filename],
        record,
      );
    }
  });
}
