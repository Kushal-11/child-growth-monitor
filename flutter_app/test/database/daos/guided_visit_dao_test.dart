import 'package:child_growth_monitor_app/database/daos/camera_result_dao.dart';
import 'package:child_growth_monitor_app/database/daos/capture_asset_dao.dart';
import 'package:child_growth_monitor_app/database/daos/guided_visit_dao.dart';
import 'package:child_growth_monitor_app/database/daos/measured_detail_revision_dao.dart';
import 'package:child_growth_monitor_app/database/daos/sync_outbox_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  late AppDatabase db;
  late GuidedVisitDao visitDao;
  late CaptureAssetDao assetDao;
  late CameraResultDao resultDao;
  late MeasuredDetailRevisionDao revisionDao;
  late int childId;

  const ownerUserId = 7;
  const visitUuid = '10000000-0000-0000-0000-000000000001';

  setUp(() async {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    visitDao = GuidedVisitDao(db);
    assetDao = CaptureAssetDao(db);
    resultDao = CameraResultDao(db);
    revisionDao = MeasuredDetailRevisionDao(db);
    childId = await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: 'Child 001',
            dateOfBirth: '2024-01-01',
            sex: 'F',
            ownerUserId: const Value(ownerUserId),
          ),
        );
  });

  tearDown(() => db.close());

  Future<Visit> createDraft() => visitDao.createDraft(
        childId: childId,
        ownerUserId: ownerUserId,
        localUuid: visitUuid,
        visitDate: DateTime(2026, 7, 29),
        ageMonths: 30,
        deviceMetadataJson: '{"platform":"android"}',
        consentVersion: 'guided_capture_consent_v1',
        consentTimestamp: DateTime(2026, 7, 29),
        consentOperatorIdentifier: '$ownerUserId',
      );

  test('createDraft persists visit and typed outbox atomically', () async {
    final visit = await createDraft();
    final outbox = await db.select(db.syncOutbox).getSingle();

    expect(visit.captureState, 'draft_capture');
    expect(visit.ownerUserId, ownerUserId);
    expect(outbox.entityType, 'visit');
    expect(outbox.entityUuid, visitUuid);
    expect(outbox.visitUuid, visitUuid);
    expect(outbox.payloadChecksum, hasLength(64));
  });

  test('createDraft rolls back visit when outbox insert fails', () async {
    await db.into(db.syncOutbox).insert(
          SyncOutboxCompanion.insert(
            ownerUserId: ownerUserId,
            visitUuid: visitUuid,
            entityType: 'visit',
            entityUuid: visitUuid,
            payloadJson: '{}',
            payloadChecksum: 'a' * 64,
          ),
        );

    await expectLater(createDraft(), throwsA(isA<SqliteException>()));
    expect(await db.select(db.visits).get(), isEmpty);
  });

  test('markIncompleteCapture is owner scoped and refreshes visit outbox',
      () async {
    await createDraft();
    final outbox = await db.select(db.syncOutbox).getSingle();
    await SyncOutboxDao(db).acknowledge(
      ownerUserId,
      outbox.id,
      '{"server_visit_id":42}',
    );

    await expectLater(
      visitDao.markIncompleteCapture(
        ownerUserId: ownerUserId + 1,
        visitUuid: visitUuid,
      ),
      throwsStateError,
    );

    final visit = await visitDao.markIncompleteCapture(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    final refreshedOutbox = await db.select(db.syncOutbox).getSingle();

    expect(visit.captureState, 'incomplete_capture');
    expect(refreshedOutbox.status, 'pending');
    expect(refreshedOutbox.payloadJson, contains('incomplete_capture'));
    expect(refreshedOutbox.payloadChecksum, hasLength(64));
  });

  test('saveAcceptedAssets rolls back every asset on a duplicate', () async {
    await createDraft();
    const duplicateUuid = '20000000-0000-0000-0000-000000000001';
    final assets = [
      AcceptedCaptureAsset(
        assetUuid: duplicateUuid,
        role: 'front',
        localPath: '/retained/front.jpg',
        capturedAt: DateTime(2026, 7, 29, 10),
        overallScore: 0.9,
        payloadJson: '{"role":"front"}',
      ),
      AcceptedCaptureAsset(
        assetUuid: duplicateUuid,
        role: 'side',
        localPath: '/retained/side.jpg',
        capturedAt: DateTime(2026, 7, 29, 10, 1),
        overallScore: 0.9,
        payloadJson: '{"role":"side"}',
      ),
    ];

    await expectLater(
      assetDao.saveAcceptedAssets(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        assets: assets,
      ),
      throwsA(isA<SqliteException>()),
    );
    expect(await db.select(db.captureAssets).get(), isEmpty);
  });

  test('appendCameraResult rolls back result when outbox insert fails',
      () async {
    final visit = await createDraft();
    await (db.update(db.visits)..where((row) => row.id.equals(visit.id))).write(
      const VisitsCompanion(captureState: Value('processing')),
    );
    const resultUuid = '30000000-0000-0000-0000-000000000001';
    await db.into(db.syncOutbox).insert(
          SyncOutboxCompanion.insert(
            ownerUserId: ownerUserId,
            visitUuid: visitUuid,
            entityType: 'camera_result',
            entityUuid: resultUuid,
            payloadJson: '{}',
            payloadChecksum: 'b' * 64,
          ),
        );
    final result = CameraResultsCompanion.insert(
      resultUuid: resultUuid,
      visitId: visit.id,
      version: 1,
      method: 'camera_screening_v1',
      modelVersion: 'model-v1',
      manifestChecksum: 'c' * 64,
      trainingDataLabel: 'research_only',
      estimatedHeightCm: const Value(88),
    );

    await expectLater(
      resultDao.appendCameraResult(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        result: result,
        payloadJson: '{"estimated_height_cm":88}',
      ),
      throwsA(isA<SqliteException>()),
    );
    expect(await db.select(db.cameraResults).get(), isEmpty);
    final storedVisit = await (db.select(db.visits)
          ..where((row) => row.id.equals(visit.id)))
        .getSingle();
    expect(storedVisit.captureState, 'processing');
  });

  test('appendCameraResult creates a version and rejects in-place replacement',
      () async {
    final visit = await createDraft();
    await (db.update(db.visits)..where((row) => row.id.equals(visit.id))).write(
      const VisitsCompanion(captureState: Value('processing')),
    );
    const resultUuid = '30000000-0000-0000-0000-000000000009';
    final companion = CameraResultsCompanion.insert(
      resultUuid: resultUuid,
      visitId: visit.id,
      version: 1,
      method: 'camera_screening_v1',
      modelVersion: 'model-v1',
      manifestChecksum: 'c' * 64,
      trainingDataLabel: 'research_only',
      estimatedHeightCm: const Value(88),
    );

    final result = await resultDao.appendCameraResult(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      result: companion,
      payloadJson: '{"estimated_height_cm":88}',
    );
    expect(result.nonClinical, isTrue);
    final storedVisit = await (db.select(db.visits)
          ..where((row) => row.id.equals(visit.id)))
        .getSingle();
    expect(storedVisit.captureState, 'estimated_report');

    await expectLater(
      resultDao.appendCameraResult(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        result: companion.copyWith(estimatedHeightCm: const Value(99)),
        payloadJson: '{"estimated_height_cm":99}',
      ),
      throwsA(
        isA<StateError>().having(
          (error) => error.message,
          'message',
          contains('immutable'),
        ),
      ),
    );
    expect(
      (await db.select(db.cameraResults).getSingle()).estimatedHeightCm,
      88,
    );
  });

  test('saveMeasuredReport rolls back revision and current report together',
      () async {
    final visit = await createDraft();
    await (db.update(db.visits)..where((row) => row.id.equals(visit.id))).write(
      const VisitsCompanion(captureState: Value('estimated_report')),
    );
    await db.into(db.measurements).insert(
          MeasurementsCompanion.insert(
            visitId: visit.id,
            manualHeightCm: const Value(88),
          ),
        );
    const revisionUuid = '40000000-0000-0000-0000-000000000001';
    await db.into(db.syncOutbox).insert(
          SyncOutboxCompanion.insert(
            ownerUserId: ownerUserId,
            visitUuid: visitUuid,
            entityType: 'measured_revision',
            entityUuid: revisionUuid,
            payloadJson: '{}',
            payloadChecksum: 'd' * 64,
          ),
        );

    await expectLater(
      revisionDao.saveMeasuredReport(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        revisionUuid: revisionUuid,
        beforeJson: '{"height_cm":88}',
        afterJson: '{"height_cm":70}',
        measurement: const MeasurementsCompanion(
          manualHeightCm: Value(70),
          measurementMode: Value('standing_height'),
          oedema: Value('no'),
        ),
        payloadJson: '{"height_cm":70}',
      ),
      throwsA(isA<SqliteException>()),
    );

    expect(await db.select(db.measuredDetailRevisions).get(), isEmpty);
    final measurement = await db.select(db.measurements).getSingle();
    expect(measurement.manualHeightCm, 88);
    final storedVisit = await (db.select(db.visits)
          ..where((row) => row.id.equals(visit.id)))
        .getSingle();
    expect(storedVisit.captureState, 'estimated_report');
  });
}
