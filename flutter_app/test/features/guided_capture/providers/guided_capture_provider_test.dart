import 'dart:async';

import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/providers/guided_capture_provider.dart';
import 'package:child_growth_monitor_app/features/guided_capture/repositories/guided_capture_repository.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/guided_camera_controller.dart';
import 'package:flutter_test/flutter_test.dart';

const child = GuidedCaptureChild(
  id: 11,
  ownerUserId: 7,
  name: 'Child 011',
  dateOfBirth: '2023-01-15',
  sex: 'F',
);

GuidedRetainedFrame retained(CaptureAssetRole role, {int rank = 1}) =>
    GuidedRetainedFrame(
      localPath: '/visits/visit-1/${role.wireValue}-$rank.jpg',
      role: role,
      capturedAt: DateTime.utc(2026, 7, 29),
      selectedRank: rank,
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
      cameraIdentifier: 'rear-0',
      lensDirection: 'back',
      deviceMetadataJson: '{"platform":"test"}',
    );

class FakeGuidedCaptureRepository implements GuidedCaptureRepository {
  GuidedCaptureChild? ownerChild = child;
  GuidedCaptureSnapshot? snapshot;
  Completer<void>? createGate;
  final events = <String>[];
  final savedRoles = <CaptureAssetRole>[];

  @override
  Future<GuidedCaptureChild?> getOwnerChild({
    required int childId,
    required int ownerUserId,
  }) async {
    events.add('get-child:$childId:$ownerUserId');
    final candidate = ownerChild;
    if (candidate?.id != childId || candidate?.ownerUserId != ownerUserId) {
      return null;
    }
    return candidate;
  }

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
    events.add('create-draft:$visitUuid:$consentVersion');
    await createGate?.future;
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
    events.add('load:$visitUuid:$ownerUserId');
    return snapshot;
  }

  @override
  Future<void> saveAcceptedFrames({
    required int ownerUserId,
    required String visitUuid,
    required List<GuidedRetainedFrame> frames,
  }) async {
    events.add('save:${frames.first.role.wireValue}');
    savedRoles.add(frames.first.role);
  }

  @override
  Future<void> markIncomplete({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    events.add('incomplete:$visitUuid');
  }
}

GuidedCaptureNotifier notifier(FakeGuidedCaptureRepository repository) =>
    GuidedCaptureNotifier(
      repository: repository,
      newUuid: () => '10000000-0000-0000-0000-000000000001',
      now: () => DateTime.utc(2026, 7, 29, 10),
    );

Future<void> initializeAndConsent(
  GuidedCaptureNotifier notifier, {
  String deviceMetadataJson = '{"platform":"test"}',
}) async {
  await notifier.initializeNew(childId: child.id, ownerUserId: 7);
  await notifier.acceptConsent(
    operatorIdentifier: 'worker-7',
    deviceMetadataJson: deviceMetadataJson,
  );
}

void main() {
  test('requires a valid owner-scoped child before showing consent', () async {
    final repository = FakeGuidedCaptureRepository()..ownerChild = null;
    final workflow = notifier(repository);

    await expectLater(
      workflow.initializeNew(childId: child.id, ownerUserId: 8),
      throwsStateError,
    );
    expect(workflow.state.phase, GuidedCapturePhase.error);
  });

  test('records consent and creates draft before capture state is exposed',
      () async {
    final repository = FakeGuidedCaptureRepository()
      ..createGate = Completer<void>();
    final workflow = notifier(repository);
    await workflow.initializeNew(childId: child.id, ownerUserId: 7);

    final accepting = workflow.acceptConsent(
      operatorIdentifier: 'worker-7',
      deviceMetadataJson: '{"platform":"test"}',
    );
    await Future<void>.delayed(Duration.zero);

    expect(workflow.state.phase, GuidedCapturePhase.consent);
    expect(workflow.state.saving, isTrue);
    expect(repository.events.last, startsWith('create-draft:'));

    repository.createGate!.complete();
    await accepting;

    expect(workflow.state.consentRecorded, isTrue);
    expect(workflow.state.phase, GuidedCapturePhase.capture);
    expect(workflow.state.currentRole, CaptureAssetRole.front);
    expect(workflow.state.visitUuid, isNotNull);
  });

  test('front and side are required while back and arm roles are skippable',
      () async {
    final repository = FakeGuidedCaptureRepository();
    final workflow = notifier(repository);
    await initializeAndConsent(workflow);

    await expectLater(workflow.skipCurrentRole(), throwsStateError);
    await workflow.acceptFrames([retained(CaptureAssetRole.front)]);
    expect(workflow.state.currentRole, CaptureAssetRole.side);
    await expectLater(workflow.skipCurrentRole(), throwsStateError);

    await workflow.acceptFrames([retained(CaptureAssetRole.side)]);
    expect(workflow.state.currentRole, CaptureAssetRole.back);
    expect(workflow.state.canReviewRequired, isTrue);

    await workflow.skipCurrentRole();
    expect(workflow.state.currentRole, CaptureAssetRole.armFront);
    await workflow.skipCurrentRole();
    expect(workflow.state.currentRole, CaptureAssetRole.armSide);
    await workflow.skipCurrentRole();
    expect(workflow.state.phase, GuidedCapturePhase.review);
    expect(repository.savedRoles, [
      CaptureAssetRole.front,
      CaptureAssetRole.side,
    ]);
  });

  test('accepted frames are persisted before advancing to the next role',
      () async {
    final repository = FakeGuidedCaptureRepository();
    final workflow = notifier(repository);
    await initializeAndConsent(workflow);

    await workflow.acceptFrames([retained(CaptureAssetRole.front)]);

    expect(repository.events.last, 'save:front');
    expect(workflow.state.currentRole, CaptureAssetRole.side);
    expect(
      workflow.state.acceptedFrames[CaptureAssetRole.front],
      hasLength(1),
    );
  });

  test('interrupted draft resumes at the first missing required role',
      () async {
    final repository = FakeGuidedCaptureRepository()
      ..snapshot = GuidedCaptureSnapshot(
        child: child,
        visitUuid: 'visit-resume',
        captureState: CaptureState.draftCapture,
        acceptedFrames: {
          CaptureAssetRole.front: [retained(CaptureAssetRole.front)],
        },
      );
    final workflow = notifier(repository);

    await workflow.resume(visitUuid: 'visit-resume', ownerUserId: 7);

    expect(workflow.state.phase, GuidedCapturePhase.capture);
    expect(workflow.state.currentRole, CaptureAssetRole.side);
  });

  test('repeated required-role failure saves an incomplete visit', () async {
    final repository = FakeGuidedCaptureRepository();
    final workflow = notifier(repository);
    await initializeAndConsent(workflow);

    for (var attempt = 0; attempt < maxRequiredRoleFailures; attempt++) {
      await workflow.recordRoleFailure(CaptureAssetRole.front);
    }

    expect(workflow.state.phase, GuidedCapturePhase.incomplete);
    expect(workflow.state.captureState, CaptureState.incompleteCapture);
    expect(repository.events.last, startsWith('incomplete:'));
  });

  test('capture state contains no height, weight, or MUAC fields', () async {
    final repository = FakeGuidedCaptureRepository();
    final workflow = notifier(repository);
    await initializeAndConsent(workflow);

    final serializedKeys = workflow.state.toDebugJson().keys.join(' ');
    expect(serializedKeys, isNot(contains('height')));
    expect(serializedKeys, isNot(contains('weight')));
    expect(serializedKeys, isNot(contains('muac')));
  });
}
