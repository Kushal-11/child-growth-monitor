import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/reports/providers/visit_report_provider.dart';
import 'package:child_growth_monitor_app/features/reports/screens/visit_report_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

CameraScreeningResult cameraResult() => CameraScreeningResult(
      resultUuid: '30000000-0000-0000-0000-000000000001',
      version: 1,
      estimatedHeightCm: 88,
      estimatedWeightKg: 11,
      heightSource: 'who_height_for_age_median_v1',
      weightSource: 'ml_weight_estimator_v1',
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

class FakeVisitReportRepository implements VisitReportRepository {
  FakeVisitReportRepository(this.snapshot);

  VisitReportSnapshot snapshot;
  final loads = <String>[];

  @override
  Future<VisitReportSnapshot> load({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    loads.add('$visitUuid:$ownerUserId');
    return snapshot;
  }
}

class FakeCameraScreeningProcessor implements CameraScreeningProcessor {
  FakeCameraScreeningProcessor(this.onProcess);

  final Future<void> Function() onProcess;
  int calls = 0;

  @override
  Future<void> process({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    calls += 1;
    await onProcess();
  }
}

VisitReportSnapshot snapshot({
  CaptureState state = CaptureState.estimatedReport,
  CameraScreeningResult? result,
  MeasuredReportSnapshot? measuredReport,
}) =>
    VisitReportSnapshot(
      visitUuid: '10000000-0000-0000-0000-000000000001',
      visitDate: DateTime(2026, 7, 29),
      captureState: state,
      latestCameraResult: result,
      acceptedAssetCount: 2,
      measuredReport: measuredReport,
    );

void main() {
  testWidgets('reads persisted report and routes measured details by visit',
      (tester) async {
    final repository = FakeVisitReportRepository(
      snapshot(result: cameraResult()),
    );
    final processor = FakeCameraScreeningProcessor(() async {});
    final router = GoRouter(
      initialLocation: '/visits/10000000-0000-0000-0000-000000000001/report',
      routes: [
        GoRoute(
          path: '/visits/:visitUuid/report',
          builder: (_, state) => VisitReportScreen(
            visitUuid: state.pathParameters['visitUuid']!,
            ownerUserId: 7,
          ),
        ),
        GoRoute(
          path: '/visits/:visitUuid/measured-details',
          builder: (_, state) => Scaffold(
            body: Text(
              '${state.pathParameters['visitUuid']}|'
              '${state.uri.queryParameters['visitDate']}',
            ),
          ),
        ),
      ],
    );

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          visitReportRepositoryProvider.overrideWithValue(repository),
          cameraScreeningProcessorProvider.overrideWithValue(processor),
        ],
        child: MaterialApp.router(routerConfig: router),
      ),
    );
    await tester.pumpAndSettle();

    expect(repository.loads, [
      '10000000-0000-0000-0000-000000000001:7',
    ]);
    expect(find.text('Estimated Growth Screening Report'), findsOneWidget);
    await tester.ensureVisible(find.text('Add Measured Details'));
    await tester.tap(find.text('Add Measured Details'));
    await tester.pumpAndSettle();
    expect(
      find.text(
        '10000000-0000-0000-0000-000000000001|2026-07-29',
      ),
      findsOneWidget,
    );
  });

  testWidgets('processing failure retries without deleting accepted media',
      (tester) async {
    final repository = FakeVisitReportRepository(
      snapshot(state: CaptureState.processingFailed),
    );
    late final FakeCameraScreeningProcessor processor;
    processor = FakeCameraScreeningProcessor(() async {
      repository.snapshot = snapshot(result: cameraResult());
    });

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          visitReportRepositoryProvider.overrideWithValue(repository),
          cameraScreeningProcessorProvider.overrideWithValue(processor),
        ],
        child: const MaterialApp(
          home: VisitReportScreen(
            visitUuid: '10000000-0000-0000-0000-000000000001',
            ownerUserId: 7,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Estimate failed — retry'), findsOneWidget);
    expect(
        find.textContaining('2 accepted photos remain saved'), findsOneWidget);
    await tester.tap(find.text('Retry estimate'));
    await tester.pumpAndSettle();

    expect(processor.calls, 1);
    expect(find.text('Estimated Growth Screening Report'), findsOneWidget);
  });

  testWidgets('draft report route generates from persisted accepted photos',
      (tester) async {
    final repository = FakeVisitReportRepository(
      snapshot(state: CaptureState.draftCapture),
    );
    late final FakeCameraScreeningProcessor processor;
    processor = FakeCameraScreeningProcessor(() async {
      repository.snapshot = snapshot(result: cameraResult());
    });

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          visitReportRepositoryProvider.overrideWithValue(repository),
          cameraScreeningProcessorProvider.overrideWithValue(processor),
        ],
        child: const MaterialApp(
          home: VisitReportScreen(
            visitUuid: '10000000-0000-0000-0000-000000000001',
            ownerUserId: 7,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(processor.calls, 1);
    expect(find.text('Estimated Growth Screening Report'), findsOneWidget);
  });

  testWidgets('measured report is primary and estimate remains comparable',
      (tester) async {
    final repository = FakeVisitReportRepository(
      snapshot(
        state: CaptureState.measuredReport,
        result: cameraResult(),
        measuredReport: const MeasuredReportSnapshot(
          heightCm: 83.5,
          hazZscore: -2.1,
          hazStatus: 'Moderate Stunting',
          oedema: 'not_checked',
          whoAcuteStatus: 'UNKNOWN',
          poshanStatus: 'Indeterminate',
          poshanComplete: false,
          classificationMethod: 'poshan_setu_v1',
        ),
      ),
    );

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          visitReportRepositoryProvider.overrideWithValue(repository),
        ],
        child: const MaterialApp(
          home: VisitReportScreen(
            visitUuid: '10000000-0000-0000-0000-000000000001',
            ownerUserId: 7,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Measurement-based Growth Report'), findsOneWidget);
    expect(find.text('Compare with estimate'), findsOneWidget);
    expect(find.text('Estimated Growth Screening Report'), findsNothing);
  });
}
