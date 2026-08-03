import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/reports/widgets/estimated_report_view.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

CameraScreeningResult result({
  double? height = 88,
  double? weight = 11,
  String? category = 'MAM',
}) {
  return CameraScreeningResult(
    resultUuid: '30000000-0000-0000-0000-000000000001',
    version: 1,
    estimatedHeightCm: height,
    estimatedWeightKg: weight,
    heightSource: height == null ? null : 'who_height_for_age_median_v1',
    weightSource: weight == null ? null : experimentalMlWeightSourceV1,
    estimatedHaz: height == null ? null : -0.5,
    estimatedWhz: weight == null ? null : -2.4,
    estimatedStuntingStatus: height == null ? null : 'Normal',
    estimatedWastingStatus: weight == null ? null : 'MAM',
    experimentalOverallCategory: category,
    componentProbabilities: category == null
        ? const {}
        : const {
            'SAM': 0.1,
            'MAM': 0.6,
            'Normal': 0.2,
            'Risk_Overweight': 0.05,
            'Overweight': 0.05,
          },
    captureQualitySummary: const {
      'overall': 0.85,
      'used_views': ['front', 'side'],
    },
    method: cameraScreeningMethodV1,
    modelVersion: 'bundled-synthetic-baseline-v1',
    manifestChecksum: 'a' * 64,
    trainingDataLabel: 'synthetic_who_research_only',
    createdAt: DateTime.utc(2026, 7, 29),
  );
}

Future<void> pumpReport(
  WidgetTester tester,
  CameraScreeningResult cameraResult,
) {
  return tester.pumpWidget(
    MaterialApp(
      home: Scaffold(
        body: EstimatedReportView(
          result: cameraResult,
          visitDate: DateTime(2026, 7, 29),
          onAddMeasuredDetails: () {},
        ),
      ),
    ),
  );
}

void main() {
  testWidgets('renders the estimated report evidence and notice',
      (tester) async {
    await pumpReport(tester, result());

    expect(find.text('Estimated Growth Screening Report'), findsOneWidget);
    expect(
      find.text(
        'The current camera model is research-only. Calibrated height, weight, '
        'MUAC, and oedema details are required before WHO classifications can '
        'be reported.',
      ),
      findsOneWidget,
    );
    expect(find.textContaining('camera_screening_v1'), findsOneWidget);
    expect(
      find.textContaining('bundled-synthetic-baseline-v1'),
      findsOneWidget,
    );
    expect(find.textContaining('60%'), findsOneWidget);
    expect(find.textContaining('85%'), findsOneWidget);
    expect(find.textContaining('Front, Side'), findsOneWidget);
    expect(find.text('Add Measured Details'), findsOneWidget);
    expect(find.text('Indeterminate'), findsNothing);
  });

  testWidgets('missing components are honest and fabricate no Normal result',
      (tester) async {
    await pumpReport(
      tester,
      result(height: null, weight: null, category: null),
    );

    expect(
      find.text('A calibrated height or length measurement is required.'),
      findsOneWidget,
    );
    expect(
      find.text('A calibrated weight measurement is required.'),
      findsOneWidget,
    );
    expect(find.text('Normal'), findsNothing);
    expect(find.text('Indeterminate'), findsNothing);
  });

  testWidgets('legacy WHO population medians are not displayed as estimates',
      (tester) async {
    await pumpReport(tester, result());

    expect(find.text('Estimated height'), findsNothing);
    expect(find.text('Estimated stunting status'), findsNothing);
    expect(find.text('Estimated wasting status'), findsNothing);
    expect(
      find.text('A calibrated height or length measurement is required.'),
      findsOneWidget,
    );
  });

  testWidgets('Normal appears only when supplied by the camera result',
      (tester) async {
    final normal = CameraScreeningResult(
      resultUuid: '30000000-0000-0000-0000-000000000002',
      version: 1,
      experimentalOverallCategory: 'Normal',
      componentProbabilities: const {
        'SAM': 0.05,
        'MAM': 0.05,
        'Normal': 0.8,
        'Risk_Overweight': 0.05,
        'Overweight': 0.05,
      },
      captureQualitySummary: const {
        'overall': 0.85,
        'used_views': ['front', 'side'],
      },
      method: cameraScreeningMethodV1,
      modelVersion: 'bundled-synthetic-baseline-v1',
      manifestChecksum: 'a' * 64,
      trainingDataLabel: 'synthetic_who_research_only',
      createdAt: DateTime.utc(2026, 7, 29),
    );

    await pumpReport(tester, normal);

    expect(find.text('Normal'), findsOneWidget);
  });
}
