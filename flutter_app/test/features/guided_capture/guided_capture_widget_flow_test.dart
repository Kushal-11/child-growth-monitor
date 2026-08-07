import 'dart:io';

import 'package:child_growth_monitor_app/database/daos/capture_asset_dao.dart';
import 'package:child_growth_monitor_app/database/daos/guided_visit_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/providers/guided_capture_provider.dart';
import 'package:child_growth_monitor_app/features/guided_capture/repositories/guided_capture_repository.dart';
import 'package:child_growth_monitor_app/features/guided_capture/screens/capture_consent_screen.dart';
import 'package:child_growth_monitor_app/features/guided_capture/screens/capture_review_screen.dart';
import 'package:child_growth_monitor_app/features/guided_capture/screens/guided_capture_flow_screen.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/guided_camera_controller.dart';
import 'package:child_growth_monitor_app/features/measured_details/domain/measured_details.dart';
import 'package:child_growth_monitor_app/features/measured_details/providers/measured_details_provider.dart';
import 'package:child_growth_monitor_app/features/measured_details/screens/add_measured_details_screen.dart';
import 'package:child_growth_monitor_app/features/measured_details/services/measured_report_service.dart';
import 'package:child_growth_monitor_app/features/reports/providers/visit_report_provider.dart';
import 'package:child_growth_monitor_app/features/reports/screens/visit_report_screen.dart';
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

const _child = GuidedCaptureChild(
  id: 11,
  ownerUserId: 7,
  name: 'Child 011',
  dateOfBirth: '2024-01-29',
  sex: 'F',
);

GuidedRetainedFrame _frame(CaptureAssetRole role) => GuidedRetainedFrame(
      localPath: '/offline/${role.wireValue}.jpg',
      role: role,
      capturedAt: DateTime.utc(2026, 7, 29, 9),
      selectedRank: 1,
      poseScore: 0.9,
      coverageScore: 0.9,
      orientationScore: 0.9,
      sharpnessScore: 0.9,
      lightingScore: 0.9,
      overallScore: 0.9,
      qualityThresholdVersion: 'guided_capture_quality_v1',
      imageWidth: 1080,
      imageHeight: 1920,
      exifOrientation: 1,
      displayOrientation: 0,
      cameraIdentifier: 'fake-camera',
      lensDirection: 'back',
      deviceMetadataJson: '{"platform":"test"}',
    );

CameraScreeningResult _estimate() => CameraScreeningResult(
      resultUuid: '30000000-0000-0000-0000-000000000001',
      version: 1,
      estimatedHeightCm: 88,
      estimatedWeightKg: 11,
      heightSource: 'who_height_for_age_median_v1',
      weightSource: 'ml_weight_estimator_v1',
      estimatedHaz: -1.2,
      estimatedWhz: -0.8,
      estimatedStuntingStatus: 'Normal',
      estimatedWastingStatus: 'NORMAL',
      captureQualitySummary: const {
        'overall': 0.9,
        'used_views': ['front', 'side'],
      },
      method: cameraScreeningMethodV1,
      modelVersion: 'widget-model-v1',
      manifestChecksum: 'a' * 64,
      trainingDataLabel: 'research_only',
      createdAt: DateTime.utc(2026, 7, 29, 10),
    );

class _WorkflowCaptureRepository implements GuidedCaptureRepository {
  GuidedCaptureSnapshot? snapshot;
  final events = <String>[];

  @override
  Future<GuidedCaptureChild?> getOwnerChild({
    required int childId,
    required int ownerUserId,
  }) async =>
      childId == _child.id && ownerUserId == _child.ownerUserId ? _child : null;

  @override
  Future<GuidedCaptureSnapshot> createDraft({
    required GuidedCaptureChild child,
    required String visitUuid,
    required DateTime visitDate,
    required String deviceMetadataJson,
    required String consentVersion,
    required DateTime consentTimestamp,
    required String consentOperatorIdentifier,
  }) async {
    events.add('draft');
    return snapshot = GuidedCaptureSnapshot(
      child: child,
      visitUuid: visitUuid,
      captureState: CaptureState.draftCapture,
      acceptedFrames: const {},
    );
  }

  @override
  Future<GuidedCaptureSnapshot?> loadDraft({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    events.add('load');
    return snapshot;
  }

  @override
  Future<void> saveAcceptedFrames({
    required int ownerUserId,
    required String visitUuid,
    required List<GuidedRetainedFrame> frames,
  }) async {
    events.add('save:${frames.first.role.wireValue}');
    final current = snapshot!;
    snapshot = GuidedCaptureSnapshot(
      child: current.child,
      visitUuid: current.visitUuid,
      captureState: current.captureState,
      acceptedFrames: {
        ...current.acceptedFrames,
        frames.first.role: List.unmodifiable(frames),
      },
    );
  }

  @override
  Future<void> markIncomplete({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    events.add('incomplete');
    final current = snapshot!;
    snapshot = GuidedCaptureSnapshot(
      child: current.child,
      visitUuid: current.visitUuid,
      captureState: CaptureState.incompleteCapture,
      acceptedFrames: current.acceptedFrames,
    );
  }
}

class _WorkflowReportRepository implements VisitReportRepository {
  _WorkflowReportRepository()
      : snapshot = VisitReportSnapshot(
          visitUuid: 'pending',
          visitDate: DateTime(2026, 7, 29),
          captureState: CaptureState.draftCapture,
          latestCameraResult: null,
          acceptedAssetCount: 2,
        );

  VisitReportSnapshot snapshot;

  @override
  Future<VisitReportSnapshot> load({
    required int ownerUserId,
    required String visitUuid,
  }) async =>
      snapshot;
}

class _WorkflowProcessor implements CameraScreeningProcessor {
  _WorkflowProcessor(this.reports);

  final _WorkflowReportRepository reports;
  int calls = 0;

  @override
  Future<void> process({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    calls += 1;
    reports.snapshot = VisitReportSnapshot(
      visitUuid: visitUuid,
      visitDate: DateTime(2026, 7, 29),
      captureState: CaptureState.estimatedReport,
      latestCameraResult: _estimate(),
      acceptedAssetCount: 2,
    );
  }
}

class _WorkflowMeasuredGateway implements MeasuredReportGateway {
  _WorkflowMeasuredGateway(this.reports);

  final _WorkflowReportRepository reports;
  MeasuredDetails? saved;

  @override
  Future<MeasuredVisitContext> loadContext({
    required int ownerUserId,
    required String visitUuid,
  }) async =>
      MeasuredVisitContext(
        visitUuid: visitUuid,
        ownerUserId: ownerUserId,
        childId: _child.id,
        visitDate: DateTime(2026, 7, 29),
        ageMonths: 30,
        completedAgeMonths: 30,
        sex: 'F',
      );

  @override
  Future<void> save({
    required int ownerUserId,
    required String visitUuid,
    required int editorUserId,
    required MeasuredDetails details,
  }) async {
    saved = details;
    final immutableEstimate = reports.snapshot.latestCameraResult;
    reports.snapshot = VisitReportSnapshot(
      visitUuid: visitUuid,
      visitDate: DateTime(2026, 7, 29),
      captureState: CaptureState.measuredReport,
      latestCameraResult: immutableEstimate,
      acceptedAssetCount: 2,
      measuredReport: MeasuredReportSnapshot(
        heightCm: details.heightCm,
        hazZscore: -2.1,
        hazStatus: 'Moderate Stunting',
        oedema: details.oedema.wireValue,
        whoAcuteStatus: 'UNKNOWN',
        poshanStatus: 'Indeterminate',
        poshanComplete: false,
        classificationMethod: 'poshan_setu_v1',
        measuredAt: details.measuredAt,
      ),
    );
  }
}

void _setLargeTestView(WidgetTester tester) {
  tester.view.physicalSize = const Size(800, 1400);
  tester.view.devicePixelRatio = 1;
  addTearDown(tester.view.resetPhysicalSize);
  addTearDown(tester.view.resetDevicePixelRatio);
}

void main() {
  testWidgets(
      'select child through immutable estimate, partial measured report, compare',
      (tester) async {
    _setLargeTestView(tester);
    final captures = _WorkflowCaptureRepository();
    final reports = _WorkflowReportRepository();
    final processor = _WorkflowProcessor(reports);
    final measured = _WorkflowMeasuredGateway(reports);
    late final GoRouter router;
    router = GoRouter(
      initialLocation: '/select-child',
      routes: [
        GoRoute(
          path: '/select-child',
          builder: (context, _) => Scaffold(
            body: Center(
              child: FilledButton(
                onPressed: () => context.go(
                  '/children/11/photo-assessment/consent',
                ),
                child: const Text('Select Child 011'),
              ),
            ),
          ),
        ),
        GoRoute(
          path: '/children/:id',
          builder: (_, __) => const Scaffold(body: Text('Child profile')),
        ),
        GoRoute(
          path: '/children/:id/photo-assessment/consent',
          builder: (_, state) => CaptureConsentScreen(
            childId: int.parse(state.pathParameters['id']!),
            ownerUserId: 7,
            operatorIdentifier: 'worker-7',
          ),
        ),
        GoRoute(
          path: '/visits/:visitUuid/capture',
          builder: (_, state) => GuidedCaptureFlowScreen(
            visitUuid: state.pathParameters['visitUuid']!,
            ownerUserId: 7,
            captureLauncher: (_, role, __) async => [_frame(role)],
          ),
        ),
        GoRoute(
          path: '/visits/:visitUuid/capture/review',
          builder: (_, state) => CaptureReviewScreen(
            visitUuid: state.pathParameters['visitUuid']!,
            ownerUserId: 7,
          ),
        ),
        GoRoute(
          path: '/visits/:visitUuid/report',
          builder: (_, state) => VisitReportScreen(
            visitUuid: state.pathParameters['visitUuid']!,
            ownerUserId: 7,
          ),
        ),
        GoRoute(
          path: '/visits/:visitUuid/measured-details',
          builder: (_, state) => AddMeasuredDetailsScreen(
            visitUuid: state.pathParameters['visitUuid']!,
            ownerUserId: 7,
          ),
        ),
      ],
    );
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          guidedCaptureRepositoryProvider.overrideWithValue(captures),
          visitReportRepositoryProvider.overrideWithValue(reports),
          cameraScreeningProcessorProvider.overrideWithValue(processor),
          measuredReportGatewayProvider.overrideWith((_) async => measured),
        ],
        child: MaterialApp.router(routerConfig: router),
      ),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.text('Select Child 011'));
    await tester.pumpAndSettle();
    expect(find.text('Photo assessment consent'), findsOneWidget);
    await tester.tap(find.text('I have caregiver consent'));
    await tester.pumpAndSettle();
    expect(captures.events.first, 'draft');

    await tester.tap(find.text('Capture front view'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Capture side view'));
    await tester.pumpAndSettle();
    expect(captures.events, containsAllInOrder(['save:front', 'save:side']));

    for (var optionalRole = 0; optionalRole < 3; optionalRole++) {
      await tester.tap(find.text('Skip optional view'));
      await tester.pumpAndSettle();
    }
    expect(find.text('Capture review'), findsOneWidget);
    await tester.tap(find.text('Generate estimated report'));
    await tester.pumpAndSettle();

    expect(processor.calls, 1);
    expect(find.text('Estimated Growth Screening Report'), findsOneWidget);
    expect(find.textContaining('research-only'), findsOneWidget);
    await tester.ensureVisible(find.text('Add Measured Details'));
    await tester.tap(find.text('Add Measured Details'));
    await tester.pumpAndSettle();

    await tester.enterText(find.byKey(const Key('measured_height')), '83.58');
    await tester.ensureVisible(find.text('Save measured details'));
    await tester.tap(find.text('Save measured details'));
    await tester.pumpAndSettle();

    expect(measured.saved?.heightCm, 83.58);
    expect(measured.saved?.weightKg, equals(null));
    expect(find.text('Measurement-based Growth Report'), findsOneWidget);
    expect(find.text('Compare with estimate'), findsOneWidget);
    expect(
      find.text('Camera model widget-model-v1; result version 1'),
      findsOneWidget,
    );
    expect(find.text('Estimated height: 88.0 cm'), findsOneWidget);
    expect(find.text('Measured height: 83.6 cm'), findsOneWidget);
    expect(find.text('Signed difference: -4.4 cm'), findsOneWidget);
    expect(
      find.text(
        'No matching estimated and measured components are available to compare.',
      ),
      findsNothing,
    );
    expect(
      reports.snapshot.latestCameraResult?.resultUuid,
      '30000000-0000-0000-0000-000000000001',
    );
  });

  testWidgets('interruption resumes persisted assets after app restart',
      (tester) async {
    _setLargeTestView(tester);
    const visitUuid = '10000000-0000-0000-0000-000000000001';
    final repository = _WorkflowCaptureRepository()
      ..snapshot = GuidedCaptureSnapshot(
        child: _child,
        visitUuid: visitUuid,
        captureState: CaptureState.draftCapture,
        acceptedFrames: {
          CaptureAssetRole.front: [_frame(CaptureAssetRole.front)],
        },
      );

    Widget app() => ProviderScope(
          overrides: [
            guidedCaptureRepositoryProvider.overrideWithValue(repository),
          ],
          child: MaterialApp(
            home: GuidedCaptureFlowScreen(
              visitUuid: visitUuid,
              ownerUserId: 7,
              captureLauncher: (_, role, __) async => [_frame(role)],
            ),
          ),
        );

    await tester.pumpWidget(app());
    await tester.pumpAndSettle();
    expect(find.text('Side full-body view'), findsOneWidget);
    await tester.tap(find.text('Capture side view'));
    await tester.pumpAndSettle();

    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pumpAndSettle();
    await tester.pumpWidget(app());
    await tester.pumpAndSettle();

    expect(repository.events.where((event) => event == 'load').length, 2);
    expect(find.text('Open capture review'), findsOneWidget);
  });

  testWidgets('three failed required captures save an incomplete visit',
      (tester) async {
    _setLargeTestView(tester);
    const visitUuid = '10000000-0000-0000-0000-000000000001';
    final repository = _WorkflowCaptureRepository()
      ..snapshot = const GuidedCaptureSnapshot(
        child: _child,
        visitUuid: visitUuid,
        captureState: CaptureState.draftCapture,
        acceptedFrames: {},
      );
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          guidedCaptureRepositoryProvider.overrideWithValue(repository),
        ],
        child: MaterialApp(
          home: GuidedCaptureFlowScreen(
            visitUuid: visitUuid,
            ownerUserId: 7,
            captureLauncher: (_, __, ___) async => null,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    for (var attempt = 0; attempt < maxRequiredRoleFailures; attempt++) {
      await tester.tap(find.text('Capture front view'));
      await tester.pumpAndSettle();
    }

    expect(repository.events, contains('incomplete'));
    expect(find.textContaining('Incomplete capture saved'), findsOneWidget);
  });

  testWidgets('failed inference retries with retained accepted assets',
      (tester) async {
    _setLargeTestView(tester);
    final reports = _WorkflowReportRepository()
      ..snapshot = VisitReportSnapshot(
        visitUuid: '10000000-0000-0000-0000-000000000001',
        visitDate: DateTime(2026, 7, 29),
        captureState: CaptureState.processingFailed,
        latestCameraResult: null,
        acceptedAssetCount: 2,
      );
    final processor = _WorkflowProcessor(reports);
    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          visitReportRepositoryProvider.overrideWithValue(reports),
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

    expect(
        find.textContaining('2 accepted photos remain saved'), findsOneWidget);
    await tester.tap(find.text('Retry estimate'));
    await tester.pumpAndSettle();

    expect(processor.calls, 1);
    expect(find.text('Estimated Growth Screening Report'), findsOneWidget);
  });

  test('offline typed outbox and accepted photo survive database restart',
      () async {
    final directory =
        await Directory.systemTemp.createTemp('guided-widget-restart-');
    addTearDown(() async => directory.delete(recursive: true));
    final databaseFile = File('${directory.path}/offline.sqlite');
    const visitUuid = '10000000-0000-0000-0000-000000000001';

    var db = AppDatabase.forTesting(NativeDatabase(databaseFile));
    final childId = await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: _child.name,
            dateOfBirth: _child.dateOfBirth,
            sex: _child.sex,
            ownerUserId: const Value(7),
          ),
        );
    var repository = DriftGuidedCaptureRepository(
      database: db,
      visitDao: GuidedVisitDao(db),
      captureAssetDao: CaptureAssetDao(db),
    );
    final child =
        await repository.getOwnerChild(childId: childId, ownerUserId: 7);
    await repository.createDraft(
      child: child!,
      visitUuid: visitUuid,
      visitDate: DateTime(2026, 7, 29),
      deviceMetadataJson: '{"offline":true}',
      consentVersion: guidedCaptureConsentVersion,
      consentTimestamp: DateTime.utc(2026, 7, 29, 9),
      consentOperatorIdentifier: 'worker-7',
    );
    await repository.saveAcceptedFrames(
      ownerUserId: 7,
      visitUuid: visitUuid,
      frames: [_frame(CaptureAssetRole.front)],
    );
    await db.close();

    db = AppDatabase.forTesting(NativeDatabase(databaseFile));
    addTearDown(db.close);
    repository = DriftGuidedCaptureRepository(
      database: db,
      visitDao: GuidedVisitDao(db),
      captureAssetDao: CaptureAssetDao(db),
    );
    final resumed = await repository.loadDraft(
      ownerUserId: 7,
      visitUuid: visitUuid,
    );
    final outbox = await db.select(db.syncOutbox).get();

    expect(
      resumed?.acceptedFrames[CaptureAssetRole.front],
      hasLength(1),
    );
    expect(outbox.map((entry) => entry.entityType), [
      'visit',
      'capture_asset',
    ]);
    expect(outbox.every((entry) => entry.status == 'pending'), isTrue);
  });
}
