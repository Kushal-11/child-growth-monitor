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
      expect(classifyWhz(-3.5), 'SAM');
    });
    test('z = -2.5 is MAM', () {
      expect(classifyWhz(-2.5), 'MAM');
    });
    test('z = 0 is Normal', () {
      expect(classifyWhz(0), 'NORMAL');
    });
    test('z = 1.5 is Risk of Overweight', () {
      expect(classifyWhz(1.5), 'RISK_OVERWEIGHT');
    });
    test('z = 2.5 is Overweight', () {
      expect(classifyWhz(2.5), 'OVERWEIGHT');
    });
    test('z = 3.5 is Obese', () {
      expect(classifyWhz(3.5), 'OBESE');
    });
  });

  group('classifyMuac', () {
    test('< 11.5 is SAM when age in range', () {
      expect(classifyMuac(11.0, true), 'SAM');
    });
    test('11.5-12.5 is At Risk (MAM)', () {
      expect(classifyMuac(12.0, true), 'MAM');
    });
    test('>= 12.5 is Normal', () {
      expect(classifyMuac(13.0, true), 'NORMAL');
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
      expect(classifyWhz(-3.0), 'MAM');
    });
    test('z just below -3 is SAM', () {
      expect(classifyWhz(-3.01), 'SAM');
    });
    test('z == -2.0 is Normal (not MAM)', () {
      expect(classifyWhz(-2.0), 'NORMAL');
    });
    test('z just below -2 is MAM', () {
      expect(classifyWhz(-2.01), 'MAM');
    });
    test('z == 1.0 is Possible Risk of Overweight', () {
      expect(classifyWhz(1.0), 'RISK_OVERWEIGHT');
    });
    test('z == 2.0 is Overweight', () {
      expect(classifyWhz(2.0), 'OVERWEIGHT');
    });
    test('z == 3.0 is Obese', () {
      expect(classifyWhz(3.0), 'OBESE');
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
      expect(classifyMuac(11.5, true), 'MAM');
    });
    test('just below 11.5 is SAM', () {
      expect(classifyMuac(11.49, true), 'SAM');
    });
    test('12.5 is Normal, not MAM', () {
      expect(classifyMuac(12.5, true), 'NORMAL');
    });
    test('just below 12.5 is At Risk (MAM)', () {
      expect(classifyMuac(12.49, true), 'MAM');
    });
  });

  // --- WHO 2009/2013 CMAM OR-rule: direct MUAC or WHZ wins ---------------
  group('combineNutritionStatus', () {
    test('MUAC SAM escalates over Normal WHZ (false-negative guard)', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'NORMAL',
          muacStatus: 'SAM',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'SAM',
      );
    });
    test('WHZ SAM escalates over Normal MUAC', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'Severe Acute Malnutrition (SAM)',
          muacStatus: 'NORMAL',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'SAM',
      );
    });
    test('WHZ-derived MUAC cannot create an independent SAM arm', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'NORMAL',
          muacStatus: 'SAM',
          muacMethod: 'estimated_from_whz',
          isDirectMeasurement: false,
        ),
        'NORMAL',
      );
    });
    test('MUAC At Risk (MAM) escalates over Normal WHZ', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'NORMAL',
          muacStatus: 'MAM',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'MAM',
      );
    });
    test('WHZ MAM over Normal MUAC', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'Moderate Acute Malnutrition (MAM)',
          muacStatus: 'NORMAL',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'MAM',
      );
    });
    test('SAM beats MAM when sources disagree', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'Moderate Acute Malnutrition (MAM)',
          muacStatus: 'SAM',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'SAM',
      );
    });
    test('MUAC SAM escalates even when WHZ could not be computed (null)', () {
      expect(
        combineNutritionStatus(
          whzStatus: null,
          muacStatus: 'SAM',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'SAM',
      );
    });
    test('all Normal is Normal', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'NORMAL',
          muacStatus: 'NORMAL',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'NORMAL',
      );
    });
    test('all null is Unknown', () {
      expect(
        combineNutritionStatus(
          whzStatus: null,
          muacStatus: null,
          muacMethod: null,
          isDirectMeasurement: false,
        ),
        'UNKNOWN',
      );
    });
    test('WHZ overweight surfaces when nothing more severe', () {
      expect(
        combineNutritionStatus(
          whzStatus: 'OVERWEIGHT',
          muacStatus: 'NORMAL',
          muacMethod: 'manual',
          isDirectMeasurement: true,
        ),
        'OVERWEIGHT',
      );
    });
  });
}
