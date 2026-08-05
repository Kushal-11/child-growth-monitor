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

    // Locked against shared/ml_parity_cases.json and Python's exact TFLite
    // interpreter. This catches model, scaler, feature-order, and platform
    // inference drift rather than merely checking that inference returns.
    expect(prediction.estimatedWeightKg, closeTo(11.794523239135742, 1e-5));
    expect(prediction.mamProbability, closeTo(0.004753005690872669, 1e-6));
    expect(prediction.normalProbability, closeTo(0.9375147223472595, 1e-6));
    expect(prediction.overweightProbability,
        closeTo(0.0010749729117378592, 1e-6));
    expect(prediction.riskProbability,
        closeTo(0.05527637526392937, 1e-6));
    expect(prediction.samProbability,
        closeTo(0.0013809206429868937, 1e-6));
  });
}
