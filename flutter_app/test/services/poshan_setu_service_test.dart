import 'dart:convert';
import 'dart:io';

import 'package:child_growth_monitor_app/services/poshan_setu_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  const service = PoshanSetuService();

  test('matches every shared Poshan Setu v1 parity case', () {
    final fixture = [
      File('../shared/poshan_setu_v1_cases.json'),
      File('shared/poshan_setu_v1_cases.json'),
    ].firstWhere((file) => file.existsSync());
    final cases = (jsonDecode(fixture.readAsStringSync()) as List)
        .cast<Map<String, dynamic>>();
    for (final testCase in cases) {
      final height = (testCase['height_cm'] as num?)?.toDouble();
      final weight = (testCase['weight_kg'] as num?)?.toDouble();
      final muac = (testCase['muac_cm'] as num?)?.toDouble();
      final result = service.classify(
        sex: testCase['sex'] as String,
        ageMonths: (testCase['age_months'] as num).toDouble(),
        heightCm: height,
        heightSource: height == null ? 'unavailable' : 'manual',
        weightKg: weight,
        weightSource: weight == null ? 'unavailable' : 'manual',
        muacCm: muac,
        muacSource: muac == null ? 'unavailable' : 'manual',
      );
      expect(result.bmiStatus, testCase['expected_bmi_status'],
          reason: testCase['name'] as String);
      expect(result.muacStatus, testCase['expected_muac_status'],
          reason: testCase['name'] as String);
      expect(result.finalStatus, testCase['expected_final_status'],
          reason: testCase['name'] as String);
    }
  });

  test('estimated values cannot certify Normal', () {
    final result = service.classify(
      sex: 'F',
      ageMonths: 36,
      heightCm: 100,
      heightSource: 'who_median_estimated',
      weightKg: 14,
      weightSource: 'ml_estimated',
      muacCm: 14,
      muacSource: 'estimated_from_whz',
    );
    expect(result.bmiStatus, 'Indeterminate');
    expect(result.muacStatus, 'Indeterminate');
    expect(result.finalStatus, 'Indeterminate');
  });

  test('MAM needs both components while SAM survives missing MUAC', () {
    final mam = service.classify(
      sex: 'M',
      ageMonths: 36,
      heightCm: 100,
      heightSource: 'manual',
      weightKg: 13.2,
      weightSource: 'manual',
      muacCm: null,
      muacSource: 'unavailable',
    );
    final sam = service.classify(
      sex: 'M',
      ageMonths: 36,
      heightCm: 100,
      heightSource: 'manual',
      weightKg: 12.9,
      weightSource: 'manual',
      muacCm: null,
      muacSource: 'unavailable',
    );
    expect(mam.finalStatus, 'Indeterminate');
    expect(mam.triggeredBy, ['bmi']);
    expect(sam.finalStatus, 'SAM');
  });
}
