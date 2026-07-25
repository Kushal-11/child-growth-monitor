import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../constants/config.dart';
import '../models/wasting_features.dart';

/// Runs the on-device weight estimator + wasting classifier.
/// Fails loudly if assets are missing — caller must catch and trigger fallback.
class MlInferenceService {
  Interpreter? _weight;
  Interpreter? _classifier;
  List<double>? _mean;
  List<double>? _scale;
  String? _modelVersion;
  String? _trainingData;
  bool? _nonClinical;

  static const _weightAsset = 'assets/models/weight_estimator.tflite';
  static const _classifierAsset = 'assets/models/wasting_classifier.tflite';
  static const _scalerAsset = 'assets/models/feature_scaler.json';
  static const _labelsAsset = 'assets/models/label_encoder.json';
  static const _manifestAsset = 'assets/models/model_manifest.json';
  static const double _probabilitySumTolerance = 1e-3;

  static const double _lowerBound = mlWeightLowerBound; // 0.45
  static const double _upperBound = mlWeightUpperBound; // 1.80

  bool get isLoaded => _weight != null && _classifier != null && _mean != null;
  String? get modelVersion => _modelVersion;
  String? get trainingData => _trainingData;
  bool? get nonClinical => _nonClinical;

  Future<void> load() async {
    try {
      final manifestJson = await rootBundle.loadString(_manifestAsset);
      final manifest = jsonDecode(manifestJson) as Map<String, dynamic>;
      _validateManifest(manifest);

      final weightBytes = _asBytes(await rootBundle.load(_weightAsset));
      final classifierBytes = _asBytes(await rootBundle.load(_classifierAsset));
      final scalerBytes = _asBytes(await rootBundle.load(_scalerAsset));
      final labelsBytes = _asBytes(await rootBundle.load(_labelsAsset));
      _validateArtifactHash(
        manifest,
        'weight_estimator.tflite',
        weightBytes,
      );
      _validateArtifactHash(
        manifest,
        'wasting_classifier.tflite',
        classifierBytes,
      );
      _validateArtifactHash(manifest, 'feature_scaler.json', scalerBytes);
      _validateArtifactHash(manifest, 'label_encoder.json', labelsBytes);

      final labelData =
          jsonDecode(utf8.decode(labelsBytes)) as Map<String, dynamic>;
      final labelClasses =
          (labelData['classes'] as List).map((v) => v as String).toList();
      if (!_sameStrings(labelClasses, wastingLabels)) {
        throw StateError(
          'label_encoder.json label order differs from app wastingLabels',
        );
      }
      if (labelData['model_version'] != manifest['model_version']) {
        throw StateError(
          'label_encoder.json model version differs from the manifest',
        );
      }

      _weight = Interpreter.fromBuffer(weightBytes);
      _classifier = Interpreter.fromBuffer(classifierBytes);

      final scalerJson = utf8.decode(scalerBytes);
      final data = jsonDecode(scalerJson) as Map<String, dynamic>;
      _mean = (data['mean'] as List).map((v) => (v as num).toDouble()).toList();
      _scale =
          (data['scale'] as List).map((v) => (v as num).toDouble()).toList();
      if (_mean!.length != 14 || _scale!.length != 14) {
        throw StateError(
          'feature_scaler.json must contain 14-element mean and scale arrays',
        );
      }
      final scalerFeatures =
          (data['feature_names'] as List).map((v) => v as String).toList();
      if (!_sameStrings(scalerFeatures, featureNames)) {
        throw StateError(
          'feature_scaler.json feature order differs from app featureNames',
        );
      }
      if (_scale!.any((v) => !v.isFinite || v <= 0) ||
          _mean!.any((v) => !v.isFinite)) {
        throw StateError('feature_scaler.json contains invalid numeric values');
      }

      final wOut = _weight!.getOutputTensor(0).shape;
      final cOut = _classifier!.getOutputTensor(0).shape;
      final wIn = _weight!.getInputTensor(0).shape;
      final cIn = _classifier!.getInputTensor(0).shape;
      if (wIn.length != 2 || wIn[0] != 1 || wIn[1] != 14) {
        throw StateError(
            'weight_estimator input shape must be [1,14], got $wIn');
      }
      if (cIn.length != 2 || cIn[0] != 1 || cIn[1] != 14) {
        throw StateError(
            'wasting_classifier input shape must be [1,14], got $cIn');
      }
      if (wOut.length != 2 || wOut[0] != 1 || wOut[1] != 1) {
        throw StateError(
            'weight_estimator output shape must be [1,1], got $wOut');
      }
      if (cOut.length != 2 || cOut[0] != 1 || cOut[1] != 5) {
        throw StateError(
            'wasting_classifier output shape must be [1,5], got $cOut');
      }
      _modelVersion = manifest['model_version'] as String;
      _trainingData = manifest['training_data'] as String;
      _nonClinical = (manifest['evaluation']
          as Map<String, dynamic>)['non_clinical'] as bool;
    } catch (_) {
      dispose();
      rethrow;
    }
  }

  Uint8List _asBytes(ByteData data) {
    return data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  bool _sameStrings(List<String> left, List<String> right) {
    if (left.length != right.length) return false;
    for (var i = 0; i < left.length; i++) {
      if (left[i] != right[i]) return false;
    }
    return true;
  }

  void _validateManifest(Map<String, dynamic> manifest) {
    final version = manifest['model_version'];
    if (version is! String || version.isEmpty) {
      throw StateError('model_manifest.json has no model_version');
    }
    if (manifest['feature_schema_version'] != 1 ||
        manifest['feature_count'] != 14) {
      throw StateError(
        'Unsupported ML feature schema in model_manifest.json',
      );
    }
    final manifestFeatures =
        (manifest['feature_names'] as List).map((v) => v as String).toList();
    if (!_sameStrings(manifestFeatures, featureNames)) {
      throw StateError(
        'model_manifest.json feature order differs from app featureNames',
      );
    }
    final labels =
        (manifest['labels'] as List).map((v) => v as String).toList();
    if (!_sameStrings(labels, wastingLabels)) {
      throw StateError(
        'model_manifest.json label order differs from app wastingLabels',
      );
    }
    if (manifest['training_data'] is! String) {
      throw StateError('model_manifest.json has no training_data provenance');
    }
    final evaluation = manifest['evaluation'];
    if (evaluation is! Map<String, dynamic> ||
        evaluation['non_clinical'] != true) {
      throw StateError(
        'model_manifest.json must declare evaluation.non_clinical=true',
      );
    }
    if (evaluation['evaluation_contract_version'] != 2 ||
        evaluation['engine'] != 'tensorflow_lite' ||
        evaluation['sam_recall_floor_met'] != true ||
        evaluation['sam_sample_count'] is! int ||
        (evaluation['sam_sample_count'] as int) <= 0 ||
        evaluation['invalid_prediction_count'] != 0) {
      throw StateError(
        'model_manifest.json has no passing TFLite safety evaluation',
      );
    }
    for (final metricName in [
      'weight_mae_kg',
      'classification_accuracy',
      'sam_recall',
      'mam_recall',
      'mam_precision',
      'sam_recall_floor',
    ]) {
      final metric = evaluation[metricName];
      if (metric is! num || !metric.toDouble().isFinite) {
        throw StateError(
          'model_manifest.json evaluation.$metricName is not finite',
        );
      }
    }
    final samRecall = (evaluation['sam_recall'] as num).toDouble();
    final samFloor = (evaluation['sam_recall_floor'] as num).toDouble();
    if (samFloor != 0.80 || samRecall < samFloor) {
      throw StateError(
        'model_manifest.json SAM recall is below its declared floor',
      );
    }
    final artifacts = manifest['artifacts'];
    final evaluatedArtifacts = evaluation['evaluated_artifacts'];
    if (artifacts is! Map<String, dynamic> ||
        evaluatedArtifacts is! Map<String, dynamic>) {
      throw StateError(
        'model_manifest.json does not bind evaluation to runtime artifacts',
      );
    }
    for (final filename in [
      'weight_estimator.tflite',
      'wasting_classifier.tflite',
      'feature_scaler.json',
      'label_encoder.json',
    ]) {
      if (jsonEncode(evaluatedArtifacts[filename]) !=
          jsonEncode(artifacts[filename])) {
        throw StateError(
          'model_manifest.json evaluation is not bound to $filename',
        );
      }
    }
  }

  void _validateArtifactHash(
    Map<String, dynamic> manifest,
    String filename,
    Uint8List bytes,
  ) {
    final artifacts = manifest['artifacts'];
    if (artifacts is! Map<String, dynamic>) {
      throw StateError('model_manifest.json has no artifacts map');
    }
    final record = artifacts[filename];
    if (record is! Map<String, dynamic>) {
      throw StateError('model_manifest.json has no record for $filename');
    }
    final expected = record['sha256'];
    final expectedSize = record['size_bytes'];
    if (expected is! String || expected.length != 64) {
      throw StateError('model_manifest.json has invalid SHA-256 for $filename');
    }
    if (expectedSize is! int || expectedSize != bytes.length) {
      throw StateError('$filename size differs from model_manifest.json');
    }
    final actual = sha256.convert(bytes).toString();
    if (actual != expected) {
      throw StateError('$filename SHA-256 differs from model_manifest.json');
    }
  }

  WastingPrediction predict(WastingFeatures features) {
    if (!isLoaded) {
      throw StateError('MlInferenceService.predict called before load()');
    }
    final rawFeatures = features.toArray();
    if (rawFeatures.any((value) => !value.isFinite)) {
      throw StateError('ML feature vector contains non-finite values');
    }
    final scaled = _scale14(rawFeatures);
    if (scaled.any((value) => !value.isFinite)) {
      throw StateError('Scaled ML feature vector contains non-finite values');
    }

    final weightOut = List.filled(1, List<double>.filled(1, 0.0));
    _weight!.run([scaled], weightOut);
    final weightKg = weightOut[0][0];

    final probsOut = List.filled(1, List<double>.filled(5, 0.0));
    _classifier!.run([scaled], probsOut);
    final probs = probsOut[0];

    return postprocessRawOutputs(
      estimatedWeightKg: weightKg,
      probabilities: probs,
      modelVersion: _modelVersion,
      trainingData: _trainingData,
      nonClinical: _nonClinical,
    );
  }

  /// Validate raw model output without clamping, normalization, or rounding.
  ///
  /// Public so host-side golden tests can enforce parity with Python without
  /// loading the platform TFLite shared library.
  static WastingPrediction postprocessRawOutputs({
    required double estimatedWeightKg,
    required List<double> probabilities,
    String? modelVersion,
    String? trainingData,
    bool? nonClinical,
  }) {
    if (!estimatedWeightKg.isFinite ||
        estimatedWeightKg < minPlausibleWeightKg ||
        estimatedWeightKg > maxPlausibleWeightKg) {
      throw StateError(
        'Weight model output is outside the plausible '
        '$minPlausibleWeightKg-$maxPlausibleWeightKg kg range',
      );
    }
    if (probabilities.length != wastingLabels.length) {
      throw StateError(
        'Classifier output length differs from label count',
      );
    }
    if (probabilities.any(
      (value) => !value.isFinite || value < 0.0 || value > 1.0,
    )) {
      throw StateError(
        'Classifier output contains invalid probabilities',
      );
    }
    final probabilitySum =
        probabilities.fold<double>(0.0, (sum, value) => sum + value);
    if ((probabilitySum - 1.0).abs() > _probabilitySumTolerance) {
      throw StateError(
        'Classifier probabilities do not sum to 1 within '
        '$_probabilitySumTolerance',
      );
    }

    var argmax = 0;
    for (var i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[argmax]) argmax = i;
    }

    return WastingPrediction(
      estimatedWeightKg: estimatedWeightKg,
      samProbability: probabilities[wastingLabels.indexOf('SAM')],
      mamProbability: probabilities[wastingLabels.indexOf('MAM')],
      normalProbability: probabilities[wastingLabels.indexOf('Normal')],
      riskProbability: probabilities[wastingLabels.indexOf('Risk_Overweight')],
      overweightProbability: probabilities[wastingLabels.indexOf('Overweight')],
      wastingStatus: wastingLabels[argmax],
      modelVersion: modelVersion,
      trainingData: trainingData,
      nonClinical: nonClinical,
    );
  }

  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  }) {
    if (!predictedKg.isFinite ||
        !whoMedianKg.isFinite ||
        predictedKg < minPlausibleWeightKg ||
        predictedKg > maxPlausibleWeightKg ||
        whoMedianKg <= 0) {
      return false;
    }
    final ratio = predictedKg / whoMedianKg;
    return ratio >= _lowerBound && ratio <= _upperBound;
  }

  List<double> _scale14(Float32List raw) {
    final out = List<double>.filled(14, 0);
    for (var i = 0; i < 14; i++) {
      out[i] = (raw[i] - _mean![i]) / _scale![i];
    }
    return out;
  }

  void dispose() {
    _weight?.close();
    _classifier?.close();
    _weight = null;
    _classifier = null;
    _mean = null;
    _scale = null;
    _modelVersion = null;
    _trainingData = null;
    _nonClinical = null;
  }
}
