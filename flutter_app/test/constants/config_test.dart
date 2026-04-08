import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/constants/config.dart';

void main() {
  group('getAnthropometricRatios', () {
    test('returns 0-12 month ratios for age 6', () {
      final r = getAnthropometricRatios(6);
      expect(r.headRatio, 0.28);
      expect(r.torsoRatio, 0.32);
      expect(r.legRatio, 0.40);
    });

    test('returns 12-24 month ratios for age 18', () {
      final r = getAnthropometricRatios(18);
      expect(r.headRatio, 0.25);
    });

    test('returns 48-60 month ratios for age 55', () {
      final r = getAnthropometricRatios(55);
      expect(r.headRatio, 0.20);
      expect(r.legRatio, 0.50);
    });
  });

  group('classifyHaz', () {
    test('z < -3 is Severely Stunted', () {
      expect(classifyHaz(-3.5), 'Severely Stunted');
    });
    test('z = -2.5 is Stunted', () {
      expect(classifyHaz(-2.5), 'Stunted');
    });
    test('z = 0 is Normal', () {
      expect(classifyHaz(0), 'Normal');
    });
    test('z = 2.5 is Tall', () {
      expect(classifyHaz(2.5), 'Tall');
    });
  });

  group('classifyWhz', () {
    test('z < -3 is SAM', () {
      expect(classifyWhz(-3.5), 'Severe Acute Malnutrition (SAM)');
    });
    test('z = -2.5 is MAM', () {
      expect(classifyWhz(-2.5), 'Moderate Acute Malnutrition (MAM)');
    });
    test('z = 0 is Normal', () {
      expect(classifyWhz(0), 'Normal');
    });
    test('z = 1.5 is Risk of Overweight', () {
      expect(classifyWhz(1.5), 'Possible Risk of Overweight');
    });
    test('z = 2.5 is Overweight', () {
      expect(classifyWhz(2.5), 'Overweight');
    });
    test('z = 3.5 is Obese', () {
      expect(classifyWhz(3.5), 'Obese');
    });
  });

  group('classifyMuac', () {
    test('< 11.5 is SAM when age in range', () {
      expect(classifyMuac(11.0, true), 'SAM');
    });
    test('11.5-12.5 is At Risk (MAM)', () {
      expect(classifyMuac(12.0, true), 'At Risk (MAM)');
    });
    test('>= 12.5 is Normal', () {
      expect(classifyMuac(13.0, true), 'Normal');
    });
    test('returns null when age not in range', () {
      expect(classifyMuac(11.0, false), isNull);
    });
  });
}
