import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/muac_service.dart';

void main() {
  test('manual MUAC takes priority', () {
    final r = MuacService.estimate(ageMonths: 24, sex: 'M', whz: -1.0, manualMuacCm: 13.5);
    expect(r.muacCm, 13.5);
    expect(r.muacMethod, 'manual');
    expect(r.muacStatus, 'NORMAL');
  });

  test('estimates from WHZ for boy age 24', () {
    final r = MuacService.estimate(ageMonths: 24, sex: 'M', whz: 0.0);
    expect(r.muacCm!, closeTo(15.7, 0.1));
    expect(r.muacStatus, 'NORMAL');
  });

  test('age out of range returns null status', () {
    final r = MuacService.estimate(ageMonths: 3, sex: 'M', whz: 0.0);
    expect(r.ageInRange, false);
    expect(r.muacStatus, isNull);
  });

  test('null whz returns null muac', () {
    final r = MuacService.estimate(ageMonths: 24, sex: 'M', whz: null);
    expect(r.muacCm, isNull);
  });

  test('median interpolates between table entries', () {
    final m = MuacService.medianForAge(15, 'M');
    expect(m, closeTo(15.35, 0.01));
  });
}
