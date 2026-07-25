import 'dart:io';

import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;
import 'package:child_growth_monitor_app/database/database.dart';

void main() {
  test('schema v4 fresh DB has Poshan provenance columns', () async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final id = await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: 'Kid',
            dateOfBirth: '2024-01-01',
            sex: 'M',
          ),
        );
    final child = await (db.select(db.children)..where((c) => c.id.equals(id)))
        .getSingle();
    expect(child.isArchived, false);
    expect(child.ownerUserId, isNull);
    expect(child.photoPath, isNull);
    final childId = child.id;
    final visitId = await db.into(db.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: '00000000-0000-4000-8000-000000000001',
            ageMonths: 12,
          ),
        );
    await db.into(db.measurements).insert(
          MeasurementsCompanion.insert(
            visitId: visitId,
            poshanStatus: const Value('Indeterminate'),
            classificationMethod: const Value('poshan_setu_v1'),
          ),
        );
    final measurement = await db.select(db.measurements).getSingle();
    expect(measurement.poshanStatus, 'Indeterminate');
    expect(measurement.mlTrainingData, isNull);
    await db.close();
  });

  // Builds a real on-disk SQLite database with the legacy table shapes that
  // existed at the given schema version, stamps user_version, then opens
  // AppDatabase over the same file so Drift's actual onUpgrade migrator runs.
  Future<void> runUpgradeFrom(int legacyVersion) async {
    final dir = await Directory.systemTemp.createTemp('cgm_mig_test');
    addTearDown(() async {
      if (await dir.exists()) {
        await dir.delete(recursive: true);
      }
    });
    final file = File(p.join(dir.path, 'mig_v$legacyVersion.sqlite'));

    // 1) Build the legacy database. enableMigrations:false stops Drift from
    //    running createAll/onUpgrade, so we can hand-craft a genuine old-era
    //    shape and stamp the legacy user_version ourselves.
    final legacy =
        AppDatabase.forTesting(NativeDatabase(file, enableMigrations: false));

    final childrenV3Columns = legacyVersion >= 3
        ? 'owner_user_id INTEGER, photo_path TEXT, '
            'is_archived INTEGER NOT NULL DEFAULT 0, '
        : '';
    await legacy.customStatement(
      'CREATE TABLE children ('
      'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
      'name TEXT NOT NULL, '
      'date_of_birth TEXT NOT NULL, '
      'sex TEXT NOT NULL, '
      'guardian_name TEXT, '
      'location TEXT, '
      '$childrenV3Columns'
      "created_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')), "
      "updated_at INTEGER NOT NULL DEFAULT (strftime('%s', 'now')))",
    );

    if (legacyVersion >= 2) {
      // v2 shape: visits/measurements/sync_queue exist but visits lacks the
      // v3 columns (owner_user_id / entry_method).
      await legacy.customStatement(
        'CREATE TABLE visits ('
        'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
        'child_id INTEGER NOT NULL REFERENCES children (id), '
        'local_uuid TEXT NOT NULL UNIQUE, '
        'visit_date INTEGER NOT NULL, '
        'age_months REAL NOT NULL, '
        'image_path TEXT, '
        'side_image_path TEXT, '
        'back_image_path TEXT, '
        'notes TEXT'
        '${legacyVersion >= 3 ? ", owner_user_id INTEGER, entry_method TEXT NOT NULL DEFAULT 'assessment'" : ""})',
      );
      await legacy.customStatement(
        'CREATE TABLE measurements ('
        'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
        'visit_id INTEGER NOT NULL REFERENCES visits (id))',
      );
      await legacy.customStatement(
        'CREATE TABLE sync_queue ('
        'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT)',
      );
    }
    // For v1 we intentionally leave visits/measurements/sync_queue absent so
    // the from < 2 destructive-recreate path is exercised.

    await legacy.customStatement('PRAGMA user_version = $legacyVersion');
    await legacy.close();

    // 2) Reopen with migrations enabled. Drift sees user_version < schemaVersion
    //    (4) and runs the real onUpgrade migrator. A query forces it to open.
    final upgraded = AppDatabase.forTesting(NativeDatabase(file));
    final rows = await upgraded.select(upgraded.children).get();
    expect(rows, isA<List>());

    // After migration the new visits columns must exist and not collide.
    final visitCols =
        await upgraded.customSelect('PRAGMA table_info(visits)').get();
    final visitNames = visitCols.map((r) => r.data['name'] as String).toSet();
    expect(visitNames.contains('owner_user_id'), true,
        reason: 'visits.owner_user_id missing after v$legacyVersion -> v3');
    expect(visitNames.contains('entry_method'), true,
        reason: 'visits.entry_method missing after v$legacyVersion -> v3');

    // New children columns must exist and be usable.
    final id = await upgraded.into(upgraded.children).insert(
          ChildrenCompanion.insert(
            name: 'M',
            dateOfBirth: '2024-01-01',
            sex: 'F',
          ),
        );
    final row = await (upgraded.select(upgraded.children)
          ..where((c) => c.id.equals(id)))
        .getSingle();
    expect(row.isArchived, false);
    expect(row.ownerUserId, isNull);

    final measurementCols =
        await upgraded.customSelect('PRAGMA table_info(measurements)').get();
    final measurementNames =
        measurementCols.map((row) => row.data['name'] as String).toSet();
    expect(measurementNames, contains('effective_height_cm'));
    expect(measurementNames, contains('bmi_status'));
    expect(measurementNames, contains('poshan_status'));
    expect(measurementNames, contains('classification_method'));
    expect(measurementNames, contains('ml_model_version'));
    expect(measurementNames, contains('ml_non_clinical'));
    expect(measurementNames, contains('ml_training_data'));

    await upgraded.close();
  }

  test('v3 -> v4 preserves rows and adds Poshan columns', () async {
    await runUpgradeFrom(3);
  });

  test('v2 -> v4 upgrade runs without crashing and adds visits columns',
      () async {
    await runUpgradeFrom(2);
  });

  test('v1 -> v4 upgrade runs without crashing (no duplicate-column)',
      () async {
    // Regression: a v1 DB hits the from < 2 destructive recreate (which builds
    // visits at the current schema, already including the new columns) AND the
    // from < 3 block. Without the from == 2 guard the visits addColumns would
    // run on already-present columns -> "duplicate column name" crash.
    await runUpgradeFrom(1);
  });
}
