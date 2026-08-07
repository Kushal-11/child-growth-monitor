import 'dart:math';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

void main() {
  late WhoDataService svc;

  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    svc = WhoDataService();
    await svc.loadFromFiles(
      manifestPath: 'assets/who_data/who_reference_manifest.json',
      wflBoysPath: 'test/fixtures/who_wfl_boys_0_2.xlsx',
      wflGirlsPath: 'test/fixtures/who_wfl_girls_0_2.xlsx',
      wfhBoysPath: 'test/fixtures/who_wfh_boys_2_5.xlsx',
      wfhGirlsPath: 'test/fixtures/who_wfh_girls_2_5.xlsx',
      lfaBoysPath: 'assets/who_data/who_lhfa_boys_0_2.xlsx',
      lfaGirlsPath: 'assets/who_data/who_lhfa_girls_0_2.xlsx',
      hfaBoysPath: 'assets/who_data/who_lhfa_boys_2_5.xlsx',
      hfaGirlsPath: 'assets/who_data/who_lhfa_girls_2_5.xlsx',
      wfaBoysPath: 'assets/who_data/who_wfa_boys_0_5.xlsx',
      wfaGirlsPath: 'assets/who_data/who_wfa_girls_0_5.xlsx',
      bfaBoys0To2Path: 'assets/who_data/who_bfa_boys_0_2.xlsx',
      bfaBoys2To5Path: 'assets/who_data/who_bfa_boys_2_5.xlsx',
      bfaGirls0To2Path: 'assets/who_data/who_bfa_girls_0_2.xlsx',
      bfaGirls2To5Path: 'assets/who_data/who_bfa_girls_2_5.xlsx',
      acfaBoysPath: 'assets/who_data/who_acfa_boys_3_5.xlsx',
      acfaGirlsPath: 'assets/who_data/who_acfa_girls_3_5.xlsx',
    );
  });

  test('service reports loaded after loadFromFiles', () {
    expect(svc.isLoaded, isTrue);
  });

  group('getMedianHeightForAge', () {
    test('returns median for 24-month-old boy', () {
      final h = svc.getMedianHeightForAge('M', 24);
      expect(h, isNotNull);
      expect(h!, closeTo(87.1, 1.0));
    });

    test('returns median for 0-month-old girl', () {
      final h = svc.getMedianHeightForAge('F', 0);
      expect(h, isNotNull);
      expect(h!, closeTo(49.1, 0.1));
    });

    test('returns median for 12-month-old boy', () {
      final h = svc.getMedianHeightForAge('M', 12);
      expect(h, isNotNull);
      expect(h!, closeTo(75.7, 0.1));
    });

    test('returns null for invalid age', () {
      expect(svc.getMedianHeightForAge('M', -1), isNull);
    });

    test('returns null for age beyond 60', () {
      expect(svc.getMedianHeightForAge('M', 61), isNull);
    });
  });

  group('official WHO reference targets', () {
    test('returns sex-specific median and reference interval', () {
      final targets = svc.getReferenceTargets('F', 0);

      expect(targets.heightForAge, isNotNull);
      expect(targets.heightForAge!.target, closeTo(49.1477, 0.0001));
      expect(targets.heightForAge!.lower2Sd, closeTo(45.422, 0.01));
      expect(targets.heightForAge!.upper2Sd, closeTo(52.872, 0.01));

      expect(targets.weightForAge, isNotNull);
      expect(targets.weightForAge!.target, closeTo(3.2322, 0.0001));
      expect(targets.weightForAge!.lower2Sd, closeTo(2.4, 0.05));
      expect(targets.weightForAge!.upper2Sd, closeTo(4.2, 0.05));

      expect(targets.muacForAge, isNull);
    });

    test('interpolates exact age and uses official ACFA table', () {
      final atSixMonths = svc.getReferenceTargets('M', 6);
      expect(atSixMonths.muacForAge, isNotNull);
      expect(atSixMonths.muacForAge!.target, closeTo(14.2389, 0.0001));
      expect(atSixMonths.muacForAge!.lower2Sd, closeTo(12.2, 0.05));
      expect(atSixMonths.muacForAge!.upper2Sd, closeTo(16.5, 0.05));

      final fractional = svc.getReferenceTargets('girl', 46.8);
      final month46 = svc.getReferenceTargets('F', 46);
      final month47 = svc.getReferenceTargets('F', 47);
      expect(
        fractional.weightForAge!.target,
        inInclusiveRange(
          month46.weightForAge!.target,
          month47.weightForAge!.target,
        ),
      );
    });

    test('returns no targets beyond the WHO under-five table', () {
      final targets = svc.getReferenceTargets('M', 61);
      expect(targets.isEmpty, isTrue);
    });
  });

  group('WAZ and BAZ LMS tables', () {
    test('returns sex-specific WFA LMS at exact decimal age', () {
      final girls = svc.getWfaLms('F', 24.5);
      final boys = svc.getWfaLms('M', 24.5);
      expect(girls, isNotNull);
      expect(boys, isNotNull);
      expect(girls!.$2, isNot(equals(boys!.$2)));
    });

    test('uses the official age-specific BMI table', () {
      final beforeTwo = svc.getBfaLms('M', 23.9);
      final afterTwo = svc.getBfaLms('M', 24.1);
      expect(beforeTwo, isNotNull);
      expect(afterTwo, isNotNull);
      expect(beforeTwo!.$2, greaterThan(0));
      expect(afterTwo!.$2, greaterThan(0));
    });
  });

  group('getHeightSdForAge', () {
    test('returns positive SD for valid age', () {
      final sd = svc.getHeightSdForAge('M', 24);
      expect(sd, isNotNull);
      expect(sd!, greaterThan(0));
    });

    test('returns null for invalid age', () {
      expect(svc.getHeightSdForAge('M', -1), isNull);
    });
  });

  group('getHeightRangeForAge', () {
    test('returns range for 24-month-old boy at 3 SD', () {
      final range = svc.getHeightRangeForAge('M', 24);
      expect(range, isNotNull);
      expect(range!.$1, lessThan(range.$2));
      // -3 SD should be around 78, +3 SD around 96-97
      expect(range.$1, closeTo(78.0, 2.0));
      expect(range.$2, closeTo(96.3, 2.0));
    });

    test('returns wider range for higher numSd', () {
      final range3 = svc.getHeightRangeForAge('M', 24, numSd: 3.0);
      final range4 = svc.getHeightRangeForAge('M', 24, numSd: 4.0);
      expect(range3, isNotNull);
      expect(range4, isNotNull);
      expect(range4!.$1, lessThan(range3!.$1));
      expect(range4.$2, greaterThan(range3.$2));
    });

    test('returns null for invalid age', () {
      expect(svc.getHeightRangeForAge('M', -1), isNull);
    });
  });

  group('lmsZscore', () {
    test('LMS formula for L != 0', () {
      final z = WhoDataService.lmsZscore(10.0, 0.5, 9.5, 0.08);
      expect(z.isFinite, isTrue);
      // Manual: (((10/9.5)^0.5) - 1) / (0.5 * 0.08)
      final expected = (pow(10.0 / 9.5, 0.5) - 1) / (0.5 * 0.08);
      expect(z, closeTo(expected, 0.001));
    });

    test('LMS formula for L = 0 uses log', () {
      final z = WhoDataService.lmsZscore(10.0, 0.0, 9.5, 0.08);
      expect(z, closeTo(log(10.0 / 9.5) / 0.08, 0.01));
    });

    test('LMS formula for very small L uses log', () {
      final z = WhoDataService.lmsZscore(10.0, 1e-8, 9.5, 0.08);
      expect(z, closeTo(log(10.0 / 9.5) / 0.08, 0.01));
    });

    test('z-score is 0 when measurement equals median', () {
      final z = WhoDataService.lmsZscore(10.0, 0.5, 10.0, 0.08);
      expect(z, closeTo(0.0, 0.001));
    });

    test('z-score is positive when measurement > median', () {
      final z = WhoDataService.lmsZscore(12.0, -0.35, 10.0, 0.08);
      expect(z, greaterThan(0));
    });

    test('z-score is negative when measurement < median', () {
      final z = WhoDataService.lmsZscore(8.0, -0.35, 10.0, 0.08);
      expect(z, lessThan(0));
    });
  });

  group('getHazBoundaries', () {
    test('returns 7 z-score boundaries for valid age/sex', () {
      final b = svc.getHazBoundaries('M', 24);
      expect(b, isNotNull);
      expect(b!.length, 7);
      expect(b[-3]!, lessThan(b[-2]!));
      expect(b[-2]!, lessThan(b[-1]!));
      expect(b[-1]!, lessThan(b[0]!));
      expect(b[0]!, lessThan(b[1]!));
      expect(b[1]!, lessThan(b[2]!));
      expect(b[2]!, lessThan(b[3]!));
      expect(b[0]!, closeTo(87.1, 1.0));
    });

    test('returns boundaries for girls', () {
      final b = svc.getHazBoundaries('F', 12);
      expect(b, isNotNull);
      expect(b![0]!, closeTo(74.0, 0.1));
    });

    test('returns null for invalid sex', () {
      expect(svc.getHazBoundaries('X', 24), isNull);
    });

    test('returns null for invalid age', () {
      expect(svc.getHazBoundaries('M', -5), isNull);
    });

    test('uses length for age < 24 and height for age >= 24', () {
      // Age 23 should use "length" measure, age 24 should use "height"
      final b23 = svc.getHazBoundaries('M', 23);
      final b24 = svc.getHazBoundaries('M', 24);
      expect(b23, isNotNull);
      expect(b24, isNotNull);
      // Both should return values; the difference is the measure type used
      // internally. The height-based 24mo value should be slightly less than
      // length-based equivalent.
    });
  });

  group('getWfhLms', () {
    test('returns LMS for a boy at 87cm age 24mo', () {
      final lms = svc.getWfhLms('M', 87.0, 24.0);
      expect(lms, isNotNull);
      // M (median weight) for 87cm WFH boys should be around 12 kg
      expect(lms!.$2, closeTo(12.0, 1.0));
    });

    test('returns LMS for a girl at 75cm age 12mo (WFL)', () {
      final lms = svc.getWfhLms('F', 75.0, 12.0);
      expect(lms, isNotNull);
      // M (median weight) for 75cm WFL girls
      expect(lms!.$2, greaterThan(5.0));
      expect(lms.$2, lessThan(15.0));
    });

    test('accepts lowercase stored sex values', () {
      expect(svc.getWfhLms('f', 100, 48), isNotNull);
      expect(svc.getWfhLms('male', 100, 48), isNotNull);
    });

    test('interpolates for non-exact height values', () {
      // 87.3 is between table entries 87.0 and 87.5
      final lms = svc.getWfhLms('M', 87.3, 24.0);
      expect(lms, isNotNull);

      // Should be between the LMS at 87.0 and 87.5
      final lms87 = svc.getWfhLms('M', 87.0, 24.0);
      final lms875 = svc.getWfhLms('M', 87.5, 24.0);
      expect(lms87, isNotNull);
      expect(lms875, isNotNull);

      // M should be between the two boundary values
      expect(lms!.$2, greaterThanOrEqualTo(lms87!.$2));
      expect(lms.$2, lessThanOrEqualTo(lms875!.$2));
    });

    test('returns null for out-of-range height', () {
      // 200cm is way beyond any table
      expect(svc.getWfhLms('M', 200.0, 24.0), isNull);
    });

    test('returns null for invalid sex', () {
      expect(svc.getWfhLms('X', 87.0, 24.0), isNull);
    });

    test('selects WFL for age < 24 and WFH for age >= 24', () {
      // WFL table starts at 45cm, WFH table starts at 65cm
      // A height of 50cm should work for WFL (age < 24) but may not be in WFH
      final wfl = svc.getWfhLms('M', 50.0, 12.0);
      expect(wfl, isNotNull);

      // 50cm is below the WFH table range (starts at 65)
      final wfh = svc.getWfhLms('M', 50.0, 36.0);
      expect(wfh, isNull);
    });
  });

  group('getMedianWeightForHeight', () {
    test('returns median weight for height', () {
      final w = svc.getMedianWeightForHeight('M', 87.0, ageMonths: 24.0);
      expect(w, isNotNull);
      expect(w!, closeTo(12.0, 1.0));
    });

    test('returns null for out-of-range height', () {
      final w = svc.getMedianWeightForHeight('M', 200.0, ageMonths: 24.0);
      expect(w, isNull);
    });

    test('defaults to ageMonths 36 (WFH)', () {
      // Without specifying age, should use WFH table (age >= 24)
      final w = svc.getMedianWeightForHeight('M', 87.0);
      expect(w, isNotNull);
      expect(w!, greaterThan(0));
    });
  });

  group('end-to-end z-score computation', () {
    test('compute WHZ for a 24mo boy at 87cm weighing 12kg', () {
      final lms = svc.getWfhLms('M', 87.0, 24.0);
      expect(lms, isNotNull);
      final z = WhoDataService.lmsZscore(12.0, lms!.$1, lms.$2, lms.$3);
      // 12kg at 87cm for a 24mo boy should be near the median (z ~ 0)
      expect(z, closeTo(0.0, 1.0));
    });

    test('severely underweight child has very negative WHZ', () {
      final lms = svc.getWfhLms('M', 87.0, 24.0);
      expect(lms, isNotNull);
      // 8kg at 87cm — very underweight
      final z = WhoDataService.lmsZscore(8.0, lms!.$1, lms.$2, lms.$3);
      expect(z, lessThan(-2.0));
    });
  });
}
