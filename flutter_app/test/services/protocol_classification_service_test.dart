import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/protocol_classification_service.dart';
import 'package:child_growth_monitor_app/models/assessment_result.dart';

void main() {
  for (final testCase in [
    ('M', 12.99, 'SAM'), ('M', 13.0, 'MAM'), ('M', 13.7, 'Normal'),
    ('F', 12.79, 'SAM'), ('F', 12.8, 'MAM'), ('F', 13.5, 'Normal'),
  ]) {
    test('${testCase.$1} BMI boundary ${testCase.$2}', () {
      expect(ProtocolClassificationService.classify(testCase.$2, 100, testCase.$1, null).bmiStatus, testCase.$3);
    });
  }

  test('missing inputs remain insufficient', () {
    expect(ProtocolClassificationService.classify(null, null, 'M', null).finalStatus, 'Insufficient data');
  });

  test('either indicator escalates using severity ordering', () {
    expect(ProtocolClassificationService.classify(14, 100, 'M', 'SAM').finalStatus, 'SAM');
    expect(ProtocolClassificationService.classify(12, 100, 'M', 'At Risk (MAM)').finalStatus, 'SAM');
  });

  test('API protocol fields deserialize for offline use', () {
    final result = AssessmentResult.fromJson({
      'child_name': 'A', 'sex': 'F', 'age_months': 24, 'summary': 'MAM',
      'measurement': <String, dynamic>{},
      'nutrition': <String, dynamic>{},
      'bmi_value': 12.8, 'bmi_status': 'MAM', 'protocol_status': 'MAM',
      'triggered_indicators': ['bmi'],
      'measurement_methods': {'height': 'manual', 'weight': 'manual'},
    });
    expect(result.bmiValue, 12.8);
    expect(result.triggeredIndicators, ['bmi']);
    expect(result.measurementMethods['weight'], 'manual');
  });
}
