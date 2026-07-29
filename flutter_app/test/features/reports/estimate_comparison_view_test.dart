import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/reports/providers/visit_report_provider.dart';
import 'package:child_growth_monitor_app/features/reports/widgets/estimate_comparison_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

CameraScreeningResult estimate() => CameraScreeningResult(
      resultUuid: '30000000-0000-0000-0000-000000000001',
      version: 2,
      supersedesResultUuid: '30000000-0000-0000-0000-000000000000',
      estimatedHeightCm: 88,
      estimatedWeightKg: 11,
      estimatedHaz: -2,
      estimatedStuntingStatus: 'Moderate Stunting',
      method: cameraScreeningMethodV1,
      modelVersion: 'camera-model-v2',
      manifestChecksum: 'a' * 64,
      trainingDataLabel: 'research_only',
      captureQualitySummary: const {},
      createdAt: DateTime.utc(2026, 7, 29),
    );

const measured = MeasuredReportSnapshot(
  heightCm: 83.5,
  hazZscore: -2.1,
  hazStatus: 'Moderate Stunting',
  oedema: 'not_checked',
  whoAcuteStatus: 'UNKNOWN',
  whoAcuteTriggeredBy: [],
  poshanStatus: 'Indeterminate',
  poshanTriggeredBy: [],
  poshanComplete: false,
  classificationMethod: 'poshan_setu_v1',
);

void main() {
  testWidgets('authorized users can compare immutable estimate and measurement',
      (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: EstimateComparisonView(
            estimate: estimate(),
            measured: measured,
            authorized: true,
          ),
        ),
      ),
    );

    expect(find.text('Compare with estimate'), findsOneWidget);
    expect(find.textContaining('camera-model-v2'), findsOneWidget);
    expect(find.textContaining('result version 2'), findsOneWidget);
    expect(find.text('Estimated height: 88.0 cm'), findsOneWidget);
    expect(find.text('Measured height: 83.5 cm'), findsOneWidget);
    expect(find.text('Signed difference: -4.5 cm'), findsOneWidget);
    expect(find.text('Absolute difference: 4.5 cm'), findsOneWidget);
    expect(find.text('Stunting classification agreement: Yes'), findsOneWidget);
    expect(find.textContaining('Estimated weight'), findsNothing);
  });

  testWidgets('comparison is hidden without authorization', (tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: EstimateComparisonView(
            estimate: estimate(),
            measured: measured,
            authorized: false,
          ),
        ),
      ),
    );

    expect(find.text('Compare with estimate'), findsNothing);
    expect(find.textContaining('camera-model-v2'), findsNothing);
  });
}
