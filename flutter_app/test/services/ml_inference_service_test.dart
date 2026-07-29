import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/models/wasting_features.dart';
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('predicts wasting class for typical 24-month boy features',
      skip:
          'requires device: libtensorflowlite_c-linux.so not present on host; integration covered in Task 14',
      () async {
    final svc = MlInferenceService();
    await svc.load();
    const features = WastingFeatures(
      ageMonths: 24,
      sexBinary: 1,
      heightCm: 87.1,
      shoulderWidthCm: 18.0,
      hipWidthCm: 15.5,
      torsoLengthCm: 26.5,
      upperArmLengthCm: 13.7,
      shoulderHeightRatio: 0.207,
      hipHeightRatio: 0.178,
      bodyBuildScore: 0,
    );
    final prediction = svc.predict(features);
    expect(prediction.estimatedWeightKg, isNotNull);
    expect(prediction.estimatedWeightKg!, inInclusiveRange(2.0, 30.0));
    expect(
      ['SAM', 'MAM', 'Normal', 'Risk_Overweight', 'Overweight']
          .contains(prediction.wastingStatus),
      isTrue,
    );
    final probSum = prediction.samProbability +
        prediction.mamProbability +
        prediction.normalProbability +
        prediction.riskProbability +
        prediction.overweightProbability;
    expect(probSum, closeTo(1.0, 0.01));
    svc.dispose();
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
