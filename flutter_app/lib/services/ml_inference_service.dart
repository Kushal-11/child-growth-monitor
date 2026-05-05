import 'dart:convert';
import 'dart:typed_data';

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

  static const _weightAsset = 'assets/models/weight_estimator.tflite';
  static const _classifierAsset = 'assets/models/wasting_classifier.tflite';
  static const _scalerAsset = 'assets/models/feature_scaler.json';

  static const double _lowerBound = mlWeightLowerBound; // 0.45
  static const double _upperBound = mlWeightUpperBound; // 1.80

  bool get isLoaded => _weight != null && _classifier != null && _mean != null;

  Future<void> load() async {
    try {
      _weight = await Interpreter.fromAsset(_weightAsset);
      _classifier = await Interpreter.fromAsset(_classifierAsset);

      final scalerJson = await rootBundle.loadString(_scalerAsset);
      final data = jsonDecode(scalerJson) as Map<String, dynamic>;
      _mean = (data['mean'] as List).map((v) => (v as num).toDouble()).toList();
      _scale = (data['scale'] as List).map((v) => (v as num).toDouble()).toList();
      if (_mean!.length != 14 || _scale!.length != 14) {
        throw StateError(
          'feature_scaler.json must contain 14-element mean and scale arrays',
        );
      }

      final wOut = _weight!.getOutputTensor(0).shape;
      final cOut = _classifier!.getOutputTensor(0).shape;
      if (wOut.length != 2 || wOut[0] != 1 || wOut[1] != 1) {
        throw StateError('weight_estimator output shape must be [1,1], got $wOut');
      }
      if (cOut.length != 2 || cOut[0] != 1 || cOut[1] != 5) {
        throw StateError('wasting_classifier output shape must be [1,5], got $cOut');
      }
    } catch (_) {
      dispose();
      rethrow;
    }
  }

  WastingPrediction predict(WastingFeatures features) {
    if (!isLoaded) {
      throw StateError('MlInferenceService.predict called before load()');
    }
    final scaled = _scale14(features.toArray());

    final weightOut = List.filled(1, List<double>.filled(1, 0.0));
    _weight!.run([scaled], weightOut);
    final weightKg = weightOut[0][0];

    final probsOut = List.filled(1, List<double>.filled(5, 0.0));
    _classifier!.run([scaled], probsOut);
    final probs = probsOut[0];

    int argmax = 0;
    for (var i = 1; i < probs.length; i++) {
      if (probs[i] > probs[argmax]) argmax = i;
    }
    final label = wastingLabels[argmax];

    return WastingPrediction(
      estimatedWeightKg: weightKg,
      samProbability: probs[wastingLabels.indexOf('SAM')],
      mamProbability: probs[wastingLabels.indexOf('MAM')],
      normalProbability: probs[wastingLabels.indexOf('Normal')],
      riskProbability: probs[wastingLabels.indexOf('Risk_Overweight')],
      overweightProbability: probs[wastingLabels.indexOf('Overweight')],
      wastingStatus: label,
    );
  }

  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  }) {
    if (whoMedianKg <= 0) return false;
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
  }
}
