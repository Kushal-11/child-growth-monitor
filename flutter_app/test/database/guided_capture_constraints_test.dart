import 'package:child_growth_monitor_app/database/database.dart';
import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sqlite3/sqlite3.dart' show SqliteException;

void main() {
  late AppDatabase db;
  late int visitId;

  setUp(() async {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    final childId = await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: 'Child 001',
            dateOfBirth: '2024-01-01',
            sex: 'F',
            ownerUserId: const Value(7),
          ),
        );
    visitId = await db.into(db.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: 'guided-visit-00000000000000000000001',
            ageMonths: 30,
            ownerUserId: const Value(7),
            captureState: const Value('draft_capture'),
          ),
        );
  });

  tearDown(() => db.close());

  test('asset UUIDs are unique and deleting metadata preserves visit',
      () async {
    final companion = CaptureAssetsCompanion.insert(
      assetUuid: 'asset-00000000-0000-0000-0000-000000000001',
      visitId: visitId,
      role: 'front',
      capturedAt: DateTime(2026, 7, 29),
    );
    final assetId = await db.into(db.captureAssets).insert(companion);

    await expectLater(
      db.into(db.captureAssets).insert(companion),
      throwsA(isA<SqliteException>()),
    );
    await (db.delete(db.captureAssets)
          ..where((asset) => asset.id.equals(assetId)))
        .go();
    expect(
      await (db.select(db.visits)..where((visit) => visit.id.equals(visitId)))
          .getSingleOrNull(),
      isNotNull,
    );
  });

  test('camera results require non-clinical true and unique versions',
      () async {
    final result = CameraResultsCompanion.insert(
      resultUuid: 'result-0000000-0000-0000-0000-000000000001',
      visitId: visitId,
      version: 1,
      method: 'camera_screening_v1',
      modelVersion: 'model-v1',
      manifestChecksum: 'a' * 64,
      trainingDataLabel: 'research_only',
    );
    await db.into(db.cameraResults).insert(result);

    await expectLater(
      db.into(db.cameraResults).insert(
            result.copyWith(
              resultUuid: const Value(
                'result-0000000-0000-0000-0000-000000000002',
              ),
            ),
          ),
      throwsA(isA<SqliteException>()),
    );
    await expectLater(
      db.into(db.cameraResults).insert(
            result.copyWith(
              resultUuid: const Value(
                'result-0000000-0000-0000-0000-000000000003',
              ),
              version: const Value(2),
              nonClinical: const Value(false),
            ),
          ),
      throwsA(isA<SqliteException>()),
    );
  });

  test('revision UUID and visit revision number are unique', () async {
    final revision = MeasuredDetailRevisionsCompanion.insert(
      revisionUuid: 'revision-000000-0000-0000-0000-000000000001',
      visitId: visitId,
      revisionNumber: 1,
      beforeJson: '{}',
      afterJson: '{"height_cm":88}',
    );
    await db.into(db.measuredDetailRevisions).insert(revision);

    await expectLater(
      db.into(db.measuredDetailRevisions).insert(
            revision.copyWith(
              revisionUuid: const Value(
                'revision-000000-0000-0000-0000-000000000002',
              ),
            ),
          ),
      throwsA(isA<SqliteException>()),
    );
  });

  test('deleting a visit cascades guided child records', () async {
    await db.into(db.captureAssets).insert(
          CaptureAssetsCompanion.insert(
            assetUuid: 'asset-00000000-0000-0000-0000-000000000004',
            visitId: visitId,
            role: 'front',
            capturedAt: DateTime(2026, 7, 29),
          ),
        );
    await db.into(db.measuredDetailRevisions).insert(
          MeasuredDetailRevisionsCompanion.insert(
            revisionUuid: 'revision-000000-0000-0000-0000-000000000004',
            visitId: visitId,
            revisionNumber: 1,
            beforeJson: '{}',
            afterJson: '{}',
          ),
        );

    await (db.delete(db.visits)..where((visit) => visit.id.equals(visitId)))
        .go();
    expect(await db.select(db.captureAssets).get(), isEmpty);
    expect(await db.select(db.measuredDetailRevisions).get(), isEmpty);
  });
}
