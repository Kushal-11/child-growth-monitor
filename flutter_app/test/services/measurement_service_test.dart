import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/models/body_measurements.dart';
import 'package:child_growth_monitor_app/services/measurement_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

import '../fixtures/who_test_data.dart';

void main() {
  late WhoDataService who;
  late MeasurementService measurement;

  setUpAll(() async {
    who = WhoDataService();
    await loadWhoForTests(who);
    measurement = MeasurementService(who);
  });

  BodySegments segs({
    double totalHeightPx = 800,
    double shoulderWidthPx = 160,
    double hipWidthPx = 140,
    double torsoLengthPx = 240,
    double upperArmLengthPx = 120,
  }) =>
      BodySegments(
        headHeightPx: 100,
        torsoLengthPx: torsoLengthPx,
        legLengthPx: 380,
        shoulderWidthPx: shoulderWidthPx,
        hipWidthPx: hipWidthPx,
        upperArmLengthPx: upperArmLengthPx,
        totalHeightPx: totalHeightPx,
        headTopY: 0,
        chinY: 100,
        shoulderMidpointY: 200,
        hipMidpointY: 440,
        heelY: 800,
        headConfidence: 1.0,
        torsoConfidence: 1.0,
        legConfidence: 1.0,
        hipConfidence: 1.0,
        armConfidence: 1.0,
      );

  test('manual height takes priority over WHO median', () {
    final m = measurement.compute(
      segments: segs(),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 90.0,
      poseConfidence: 0.9,
    );
    expect(m.effectiveHeightCm, 90.0);
    expect(m.estimationMethod, 'manual');
  });

  test('falls back to WHO median when manual height absent', () {
    final m = measurement.compute(
      segments: segs(),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: null,
      poseConfidence: 0.9,
    );
    expect(m.estimationMethod, 'who_statistical');
    expect(m.effectiveHeightCm, closeTo(87.1, 1.0));
  });

  test('shoulder width converts to cm using height as scale', () {
    // totalHeightPx 800, height 80cm -> scale = 0.1 cm/px
    // shoulder 160 px -> 16.0 cm
    final m = measurement.compute(
      segments: segs(totalHeightPx: 800, shoulderWidthPx: 160),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.shoulderWidthCm, closeTo(16.0, 0.01));
  });

  test('body build classified as slender below threshold', () {
    // expected ratio at 24mo = 0.210; threshold 0.02 -> slender if < 0.190
    // height 80, slender shoulder = 80 * 0.18 = 14.4cm -> shoulder 144 px
    final m = measurement.compute(
      segments: segs(totalHeightPx: 800, shoulderWidthPx: 144),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.bodyBuild, 'slender');
    expect(m.bodyBuildScore, -1);
  });

  test('body build classified as stocky above threshold', () {
    // stocky if ratio > 0.230. shoulder 0.24 * 80 = 19.2cm -> 192 px
    final m = measurement.compute(
      segments: segs(totalHeightPx: 800, shoulderWidthPx: 192),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.bodyBuild, 'stocky');
    expect(m.bodyBuildScore, 1);
  });

  test('side-view depths populated when SideViewSegments provided', () {
    final m = measurement.compute(
      segments: segs(),
      sideSegments: SideViewSegments(
        chestDepthPx: 60,
        abdDepthPx: 70,
        totalHeightPx: 800,
        chestConfidence: 1.0,
        abdConfidence: 1.0,
      ),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.sideViewUsed, true);
    expect(m.chestDepthCm, closeTo(6.0, 0.01));
    expect(m.abdDepthCm, closeTo(7.0, 0.01));
  });

  test('chest/abd depth null when no side view', () {
    final m = measurement.compute(
      segments: segs(),
      ageMonths: 24,
      sex: 'M',
      manualHeightCm: 80.0,
      poseConfidence: 0.9,
    );
    expect(m.sideViewUsed, false);
    expect(m.chestDepthCm, isNull);
    expect(m.abdDepthCm, isNull);
  });
}
