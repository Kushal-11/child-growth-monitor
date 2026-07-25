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
      expect(
        result.bmiStatus,
        testCase['expected_bmi_status'],
        reason: testCase['name'] as String,
      );
      expect(
        result.muacStatus,
        testCase['expected_muac_status'],
        reason: testCase['name'] as String,
      );
      expect(
        result.finalStatus,
        testCase['expected_final_status'],
        reason: testCase['name'] as String,
      );
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

  test('non-finite values are ineligible', () {
    final result = service.classify(
      sex: 'M',
      ageMonths: 36,
      heightCm: double.nan,
      heightSource: 'manual',
      weightKg: double.infinity,
      weightSource: 'manual',
      muacCm: double.negativeInfinity,
      muacSource: 'manual',
    );
    expect(result.bmi, isNull);
    expect(result.finalStatus, 'Indeterminate');
  });

  test('very low finite measured values remain eligible', () {
    final result = service.classify(
      sex: 'M',
      ageMonths: 6,
      heightCm: 30,
      heightSource: ' manual ',
      weightKg: 0.5,
      weightSource: 'manual',
      muacCm: 5,
      muacSource: 'tape',
    );
    expect(result.bmiStatus, 'SAM');
    expect(result.muacStatus, 'SAM');
    expect(result.finalStatus, 'SAM');
  });

  test('known MAM remains triggered when final is Indeterminate', () {
    final result = service.classify(
      sex: 'M',
      ageMonths: 36,
      heightCm: 100,
      heightSource: 'manual',
      weightKg: 13,
      weightSource: 'reference_object',
      muacCm: null,
      muacSource: 'unavailable',
    );
    expect(result.finalStatus, 'Indeterminate');
    expect(result.triggeredBy, ['bmi']);
    expect(result.rationale, contains('MAM'));
  });

  test('source canonicalisation mirrors the Python contract', () {
    expect(
      PoshanSetuService.normalizeSource('who_median_estimated'),
      'who_statistical',
    );
    expect(
      PoshanSetuService.normalizeSource('estimated_from_whz'),
      'whz_derived',
    );
    expect(
      PoshanSetuService.normalizeSource('anthropometric'),
      'landmark_estimated',
    );
    expect(PoshanSetuService.normalizeSource('made_up'), 'unavailable');
    expect(PoshanSetuService.normalizeSource('tape'), 'unavailable');
    expect(PoshanSetuService.normalizeMuacSource('tape'), 'manual');
  });

  test('MUAC remains eligible until the fifth birthday', () {
    final beforeFive = service.classify(
      sex: 'F',
      ageMonths: 59.999,
      heightCm: null,
      heightSource: 'unavailable',
      weightKg: null,
      weightSource: 'unavailable',
      muacCm: 11,
      muacSource: 'manual',
    );
    final atFive = service.classify(
      sex: 'F',
      ageMonths: 60,
      heightCm: null,
      heightSource: 'unavailable',
      weightKg: null,
      weightSource: 'unavailable',
      muacCm: 11,
      muacSource: 'manual',
    );

    expect(beforeFive.muacStatus, 'SAM');
    expect(beforeFive.finalStatus, 'SAM');
    expect(atFive.muacStatus, 'Indeterminate');
  });
}
