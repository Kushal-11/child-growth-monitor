import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:child_growth_monitor_app/models/wasting_features.dart';
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('bundled TFLite models create interpreters and run on device',
      (tester) async {
    final service = MlInferenceService();
    addTearDown(service.dispose);

    await service.load();

    final prediction = service.predict(
      const WastingFeatures(
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
      ),
    );

    expect(prediction.estimatedWeightKg, inInclusiveRange(2, 30));
    expect(
      prediction.samProbability +
          prediction.mamProbability +
          prediction.normalProbability +
          prediction.riskProbability +
          prediction.overweightProbability,
      closeTo(1, 0.01),
    );
  });
}
