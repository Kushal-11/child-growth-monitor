import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/muac_service.dart';

void main() {
  test('manual MUAC takes priority', () {
    final r = MuacService.estimate(
      ageMonths: 24,
      sex: 'M',
      whz: -1.0,
      manualMuacCm: 13.5,
    );
    expect(r.muacCm, 13.5);
    expect(r.muacMethod, 'manual');
    expect(r.muacStatus, 'NORMAL');
    expect(r.isDirectMeasurement, isTrue);
    expect(r.requiresConfirmation, isFalse);
  });

  test('estimates from WHZ for boy age 24', () {
    final r = MuacService.estimate(ageMonths: 24, sex: 'M', whz: 0.0);
    expect(r.muacCm!, closeTo(15.7, 0.1));
    expect(r.muacStatus, isNull);
    expect(r.isDirectMeasurement, isFalse);
    expect(r.requiresConfirmation, isTrue);
    expect(r.confidence, 0.4);
  });

  test('uses pose landmarks before WHZ when body proportions are available',
      () {
    final r = MuacService.estimate(
      ageMonths: 24,
      sex: 'M',
      whz: -1.5,
      heightCm: 87,
      upperArmLengthCm: 87 * 0.160,
      shoulderWidthCm: 87 * 0.212,
      landmarkVisibility: 0.95,
      muacMedianCm: 15.75,
    );

    expect(r.muacCm, closeTo(15.8, 0.1));
    expect(r.muacMethod, 'landmark_estimated');
    expect(r.muacStatus, isNull);
    expect(r.confidence, 0.95);
    expect(r.uncertaintyLowerCm, closeTo(15.2, 0.1));
    expect(r.uncertaintyUpperCm, closeTo(16.4, 0.1));
    expect(r.requiresConfirmation, isTrue);
  });

  test('age out of range returns null status', () {
    final r = MuacService.estimate(ageMonths: 3, sex: 'M', whz: 0.0);
    expect(r.ageInRange, false);
    expect(r.muacStatus, isNull);
  });

  test('null whz returns null muac', () {
    final r = MuacService.estimate(ageMonths: 24, sex: 'M', whz: null);
    expect(r.muacCm, isNull);
    expect(r.requiresConfirmation, isTrue);
  });

  test('median interpolates between table entries', () {
    final m = MuacService.medianForAge(15, 'M');
    expect(m, closeTo(15.35, 0.01));
  });
}
