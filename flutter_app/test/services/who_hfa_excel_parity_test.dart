import 'dart:io';
import 'dart:math' as math;

import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late WhoDataService who;

  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    who = WhoDataService();
    await who.loadFromFiles(
      manifestPath: 'assets/who_data/who_reference_manifest.json',
      wflBoysPath: 'test/fixtures/who_wfl_boys_0_2.xlsx',
      wflGirlsPath: 'test/fixtures/who_wfl_girls_0_2.xlsx',
      wfhBoysPath: 'test/fixtures/who_wfh_boys_2_5.xlsx',
      wfhGirlsPath: 'test/fixtures/who_wfh_girls_2_5.xlsx',
      lfaBoysPath: 'assets/who_data/who_lhfa_boys_0_2.xlsx',
      lfaGirlsPath: 'assets/who_data/who_lhfa_girls_0_2.xlsx',
      hfaBoysPath: 'assets/who_data/who_lhfa_boys_2_5.xlsx',
      hfaGirlsPath: 'assets/who_data/who_lhfa_girls_2_5.xlsx',
    );
  });

  const expected = <(String, int), (double, double, double)>{
    ('F', 0): (1.0, 49.1477, 0.03790),
    ('M', 0): (1.0, 49.8842, 0.03795),
    ('F', 24): (1.0, 85.7153, 0.03764),
    ('M', 24): (1.0, 87.1161, 0.03507),
    ('F', 60): (1.0, 109.4233, 0.04347),
    ('M', 60): (1.0, 109.9638, 0.04214),
  };

  test('Dart HFA LMS matches official workbooks at boundary ages', () {
    for (final entry in expected.entries) {
      final actual = who.getHazLms(entry.key.$1, entry.key.$2);
      expect(actual, isNotNull);
      expect(actual!.$1, closeTo(entry.value.$1, 1e-6));
      expect(actual.$2, closeTo(entry.value.$2, 1e-6));
      expect(actual.$3, closeTo(entry.value.$3, 1e-6));
    }
  });

  test('HAZ uses the LMS formula instead of CSV boundary interpolation', () {
    final nutrition = NutritionService(who);
    for (final entry in expected.entries) {
      final (lValue, median, sValue) = entry.value;
      final heightAtMinusTwo =
          median * math.pow(1 + lValue * sValue * -2.0, 1 / lValue).toDouble();
      final actual = nutrition.computeHaz(
        entry.key.$1,
        entry.key.$2,
        heightAtMinusTwo,
      );
      expect(actual, closeTo(-2.0, 1e-9));
    }
  });

  test('checksum mismatch fails closed before workbook parsing', () async {
    final temporaryDirectory =
        await Directory.systemTemp.createTemp('who-hfa-checksum-');
    addTearDown(() => temporaryDirectory.delete(recursive: true));
    final official =
        await File('assets/who_data/who_lhfa_boys_0_2.xlsx').readAsBytes();
    official[0] ^= 0xff;
    final corrupted = File('${temporaryDirectory.path}/corrupted.xlsx');
    await corrupted.writeAsBytes(official);

    final service = WhoDataService();
    await expectLater(
      service.loadFromFiles(
        manifestPath: 'assets/who_data/who_reference_manifest.json',
        wflBoysPath: 'test/fixtures/who_wfl_boys_0_2.xlsx',
        wflGirlsPath: 'test/fixtures/who_wfl_girls_0_2.xlsx',
        wfhBoysPath: 'test/fixtures/who_wfh_boys_2_5.xlsx',
        wfhGirlsPath: 'test/fixtures/who_wfh_girls_2_5.xlsx',
        lfaBoysPath: corrupted.path,
        lfaGirlsPath: 'assets/who_data/who_lhfa_girls_0_2.xlsx',
        hfaBoysPath: 'assets/who_data/who_lhfa_boys_2_5.xlsx',
        hfaGirlsPath: 'assets/who_data/who_lhfa_girls_2_5.xlsx',
      ),
      throwsA(
        isA<StateError>().having(
          (error) => error.message,
          'message',
          contains('checksum mismatch'),
        ),
      ),
    );
  });
}
