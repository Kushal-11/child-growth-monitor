import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/providers/guided_capture_provider.dart';
import 'package:child_growth_monitor_app/features/guided_capture/repositories/guided_capture_repository.dart';
import 'package:child_growth_monitor_app/features/guided_capture/screens/capture_consent_screen.dart';
import 'package:child_growth_monitor_app/features/guided_capture/screens/capture_review_screen.dart';
import 'package:child_growth_monitor_app/features/guided_capture/screens/guided_capture_flow_screen.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/guided_camera_controller.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';

const testChild = GuidedCaptureChild(
  id: 11,
  ownerUserId: 7,
  name: 'Child 011',
  dateOfBirth: '2023-01-15',
  sex: 'F',
);

class WidgetRepository implements GuidedCaptureRepository {
  final events = <String>[];
  GuidedCaptureSnapshot? snapshot;

  @override
  Future<GuidedCaptureChild?> getOwnerChild({
    required int childId,
    required int ownerUserId,
  }) async =>
      childId == testChild.id && ownerUserId == testChild.ownerUserId
          ? testChild
          : null;

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
    events.add('draft-created');
    return GuidedCaptureSnapshot(
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
    events.add('load:$visitUuid:$ownerUserId');
    return snapshot;
  }

  @override
  Future<void> saveAcceptedFrames({
    required int ownerUserId,
    required String visitUuid,
    required List<GuidedRetainedFrame> frames,
  }) async {
    events.add('saved:${frames.first.role.wireValue}');
  }

  @override
  Future<void> markIncomplete({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    events.add('incomplete');
  }
}

GuidedRetainedFrame frame(CaptureAssetRole role) => GuidedRetainedFrame(
      localPath: '/visit/${role.wireValue}.jpg',
      role: role,
      capturedAt: DateTime.utc(2026, 7, 29),
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
      cameraIdentifier: 'fake',
      lensDirection: 'back',
      deviceMetadataJson: '{}',
    );

void main() {
  testWidgets('consent creates draft before front-side-review flow',
      (tester) async {
    final repository = WidgetRepository();
    late final GoRouter router;
    Future<List<GuidedRetainedFrame>?> launcher(
      BuildContext context,
      CaptureAssetRole role,
      String visitUuid,
    ) async {
      return [frame(role)];
    }

    router = GoRouter(
      initialLocation: '/children/11/photo-assessment/consent',
      routes: [
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
            captureLauncher: launcher,
          ),
        ),
        GoRoute(
          path: '/visits/:visitUuid/capture/review',
          builder: (_, state) => CaptureReviewScreen(
            visitUuid: state.pathParameters['visitUuid']!,
          ),
        ),
      ],
    );

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          guidedCaptureRepositoryProvider.overrideWithValue(repository),
        ],
        child: MaterialApp.router(routerConfig: router),
      ),
    );
    await tester.pumpAndSettle();

    expect(
      find.textContaining('estimated growth screening and model evaluation'),
      findsOneWidget,
    );
    await tester.tap(find.text('I have caregiver consent'));
    await tester.pumpAndSettle();

    expect(repository.events.first, 'draft-created');
    expect(find.text('Front full-body view'), findsOneWidget);

    await tester.tap(find.text('Capture front view'));
    await tester.pumpAndSettle();
    expect(repository.events, contains('saved:front'));
    expect(find.text('Side full-body view'), findsOneWidget);

    await tester.tap(find.text('Capture side view'));
    await tester.pumpAndSettle();
    expect(repository.events, contains('saved:side'));
    expect(find.text('Review required photos'), findsOneWidget);

    await tester.tap(find.text('Review required photos'));
    await tester.pumpAndSettle();

    expect(find.text('Capture review'), findsOneWidget);
    expect(find.text('Front view'), findsOneWidget);
    expect(find.text('Side view'), findsOneWidget);
  });

  testWidgets('direct review route restores persisted accepted photos',
      (tester) async {
    final repository = WidgetRepository()
      ..snapshot = GuidedCaptureSnapshot(
        child: testChild,
        visitUuid: 'visit-resume-1234',
        captureState: CaptureState.draftCapture,
        acceptedFrames: {
          CaptureAssetRole.front: [frame(CaptureAssetRole.front)],
          CaptureAssetRole.side: [frame(CaptureAssetRole.side)],
        },
      );

    await tester.pumpWidget(
      ProviderScope(
        overrides: [
          guidedCaptureRepositoryProvider.overrideWithValue(repository),
        ],
        child: const MaterialApp(
          home: CaptureReviewScreen(
            visitUuid: 'visit-resume-1234',
            ownerUserId: 7,
          ),
        ),
      ),
    );
    await tester.pumpAndSettle();

    expect(repository.events, contains('load:visit-resume-1234:7'));
    expect(find.text('Front view'), findsOneWidget);
    expect(find.text('Side view'), findsOneWidget);
    final reportButton = tester.widget<FilledButton>(
      find.widgetWithText(FilledButton, 'Generate estimated report'),
    );
    expect(reportButton.onPressed, isNotNull);
  });
}
