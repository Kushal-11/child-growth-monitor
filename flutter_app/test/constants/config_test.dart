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

  // --- Exact WHO boundary regression net --------------------------------
  // Lock the precise threshold operators so an off-by-one flip (< vs <=)
  // that would misclassify a malnourished child as healthy is caught in CI.
  group('classifyWhz exact boundaries', () {
    test('z == -3.0 is MAM (not SAM)', () {
      expect(classifyWhz(-3.0), 'Moderate Acute Malnutrition (MAM)');
    });
    test('z just below -3 is SAM', () {
      expect(classifyWhz(-3.01), 'Severe Acute Malnutrition (SAM)');
    });
    test('z == -2.0 is Normal (not MAM)', () {
      expect(classifyWhz(-2.0), 'Normal');
    });
    test('z just below -2 is MAM', () {
      expect(classifyWhz(-2.01), 'Moderate Acute Malnutrition (MAM)');
    });
    test('z == 1.0 is Possible Risk of Overweight', () {
      expect(classifyWhz(1.0), 'Possible Risk of Overweight');
    });
    test('z == 2.0 is Overweight', () {
      expect(classifyWhz(2.0), 'Overweight');
    });
    test('z == 3.0 is Obese', () {
      expect(classifyWhz(3.0), 'Obese');
    });
  });

  group('classifyHaz exact boundaries', () {
    test('z == -3.0 is Stunted (not Severely Stunted)', () {
      expect(classifyHaz(-3.0), 'Stunted');
    });
    test('z just below -3 is Severely Stunted', () {
      expect(classifyHaz(-3.01), 'Severely Stunted');
    });
    test('z == -2.0 is Normal (not Stunted)', () {
      expect(classifyHaz(-2.0), 'Normal');
    });
    test('z just below -2 is Stunted', () {
      expect(classifyHaz(-2.01), 'Stunted');
    });
    test('z == 2.0 is Tall', () {
      expect(classifyHaz(2.0), 'Tall');
    });
  });

  group('classifyMuac exact WHO boundaries', () {
    test('11.5 is At Risk (MAM), not SAM', () {
      expect(classifyMuac(11.5, true), 'At Risk (MAM)');
    });
    test('just below 11.5 is SAM', () {
      expect(classifyMuac(11.49, true), 'SAM');
    });
    test('12.5 is Normal, not MAM', () {
      expect(classifyMuac(12.5, true), 'Normal');
    });
    test('just below 12.5 is At Risk (MAM)', () {
      expect(classifyMuac(12.49, true), 'At Risk (MAM)');
    });
  });

  // --- WHO 2009/2013 CMAM OR-rule: most-severe of WHZ/MUAC/ML wins -------
  group('combineNutritionStatus', () {
    test('MUAC SAM escalates over Normal WHZ (false-negative guard)', () {
      expect(
        combineNutritionStatus(whzStatus: 'Normal', muacStatus: 'SAM'),
        'SAM',
      );
    });
    test('WHZ SAM escalates over Normal MUAC', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'Severe Acute Malnutrition (SAM)',
          muacStatus: 'Normal',
        ),
        'SAM',
      );
    });
    test('ML SAM escalates over Normal WHZ with no MUAC', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'Normal',
          muacStatus: null,
          mlStatus: 'SAM',
        ),
        'SAM',
      );
    });
    test('MUAC At Risk (MAM) escalates over Normal WHZ', () {
      expect(
        combineNutritionStatus(
            whzStatus: 'Normal', muacStatus: 'At Risk (MAM)'),
        'MAM',
      );
    });
    test('WHZ MAM over Normal MUAC', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'Moderate Acute Malnutrition (MAM)',
          muacStatus: 'Normal',
        ),
        'MAM',
      );
    });
    test('ML MAM escalates over Normal WHZ', () {
      expect(
        combineNutritionStatus(
            whzStatus: 'Normal', muacStatus: null, mlStatus: 'MAM'),
        'MAM',
      );
    });
    test('SAM beats MAM when sources disagree', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'Moderate Acute Malnutrition (MAM)',
          muacStatus: 'SAM',
        ),
        'SAM',
      );
    });
    test('MUAC SAM escalates even when WHZ could not be computed (null)', () {
      expect(
        combineNutritionStatus(whzStatus: null, muacStatus: 'SAM'),
        'SAM',
      );
    });
    test('all Normal is Normal', () {
      expect(
        combineNutritionStatus(
            whzStatus: 'Normal', muacStatus: 'Normal', mlStatus: 'Normal'),
        'Normal',
      );
    });
    test('all null is Unknown', () {
      expect(
        combineNutritionStatus(whzStatus: null, muacStatus: null),
        'Unknown',
      );
    });
    test('WHZ overweight surfaces when nothing more severe', () {
      expect(
        combineNutritionStatus(whzStatus: 'Overweight', muacStatus: 'Normal'),
        'Overweight',
      );
    });
  });
}
