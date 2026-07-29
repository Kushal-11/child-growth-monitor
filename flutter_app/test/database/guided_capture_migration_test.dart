import 'dart:io';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:path/path.dart' as p;

void main() {
  test('v5 to v7 migration preserves legacy rows and adds guided schema',
      () async {
    final directory =
        await Directory.systemTemp.createTemp('guided-capture-migration-');
    addTearDown(() async => directory.delete(recursive: true));
    final file = File(p.join(directory.path, 'schema-v5.sqlite'));
    final legacy =
        AppDatabase.forTesting(NativeDatabase(file, enableMigrations: false));

    await legacy.customStatement(
      'CREATE TABLE children ('
      'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
      'name TEXT NOT NULL, date_of_birth TEXT NOT NULL, sex TEXT NOT NULL, '
      'guardian_name TEXT, location TEXT, owner_user_id INTEGER, '
      'photo_path TEXT, is_archived INTEGER NOT NULL DEFAULT 0, '
      'created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)',
    );
    await legacy.customStatement(
      'CREATE TABLE visits ('
      'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
      'child_id INTEGER NOT NULL REFERENCES children(id), '
      'local_uuid TEXT NOT NULL UNIQUE, visit_date INTEGER NOT NULL, '
      'age_months REAL NOT NULL, image_path TEXT, side_image_path TEXT, '
      'back_image_path TEXT, notes TEXT, owner_user_id INTEGER, '
      "entry_method TEXT NOT NULL DEFAULT 'assessment')",
    );
    await legacy.customStatement(
      'CREATE TABLE measurements ('
      'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
      'visit_id INTEGER NOT NULL UNIQUE REFERENCES visits(id), '
      'manual_height_cm REAL, manual_weight_kg REAL)',
    );
    await legacy.customStatement(
      'CREATE TABLE sync_queue ('
      'id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT, '
      'visit_id INTEGER NOT NULL REFERENCES visits(id), '
      "status TEXT NOT NULL DEFAULT 'pending', retry_count INTEGER NOT NULL "
      'DEFAULT 0, created_at INTEGER NOT NULL, last_attempt_at INTEGER, '
      'server_visit_id INTEGER, error_message TEXT)',
    );
    await legacy.customStatement(
      "INSERT INTO children VALUES "
      "(1, 'Child 001', '2024-01-01', 'F', NULL, NULL, 7, NULL, 0, 1, 1)",
    );
    await legacy.customStatement(
      "INSERT INTO visits VALUES "
      "(2, 1, 'legacy-visit-00000000000000000000001', 1, 30, NULL, NULL, "
      "NULL, 'keep', 7, 'manual')",
    );
    await legacy.customStatement(
      'INSERT INTO measurements '
      '(id, visit_id, manual_height_cm, manual_weight_kg) '
      'VALUES (3, 2, 88, 12)',
    );
    await legacy.customStatement(
      "INSERT INTO sync_queue VALUES (4, 2, 'pending', 0, 1, NULL, NULL, NULL)",
    );
    await legacy.customStatement('PRAGMA user_version = 5');
    await legacy.close();

    final upgraded = AppDatabase.forTesting(NativeDatabase(file));
    final preserved = await upgraded
        .customSelect(
          'SELECT v.notes, m.manual_height_cm, s.status '
          'FROM visits v JOIN measurements m ON m.visit_id = v.id '
          'JOIN sync_queue s ON s.visit_id = v.id',
        )
        .getSingle();
    expect(preserved.data['notes'], 'keep');
    expect(preserved.data['manual_height_cm'], 88.0);
    expect(preserved.data['status'], 'pending');

    final visitColumns =
        await upgraded.customSelect('PRAGMA table_info(visits)').get();
    final visitNames =
        visitColumns.map((row) => row.data['name'] as String).toSet();
    expect(
      visitNames,
      containsAll([
        'capture_state',
        'capture_started_at',
        'capture_completed_at',
        'device_metadata_json',
        'consent_version',
        'consent_timestamp',
        'consent_operator_identifier',
        'media_deleted_at',
        'server_id',
      ]),
    );

    final measurementColumns =
        await upgraded.customSelect('PRAGMA table_info(measurements)').get();
    final measurementNames =
        measurementColumns.map((row) => row.data['name'] as String).toSet();
    expect(
      measurementNames,
      containsAll([
        'measurement_mode',
        'oedema',
        'measured_at',
        'editor_user_id',
        'measured_notes',
        'who_acute_status',
        'who_acute_triggered_by',
        'who_acute_rationale',
      ]),
    );

    final tables = await upgraded
        .customSelect("SELECT name FROM sqlite_master WHERE type = 'table'")
        .get();
    expect(
      tables.map((row) => row.data['name']),
      containsAll([
        'capture_assets',
        'camera_results',
        'measured_detail_revisions',
        'sync_outbox',
      ]),
    );
    for (final table in [
      'capture_assets',
      'camera_results',
      'measured_detail_revisions',
    ]) {
      final columns =
          await upgraded.customSelect('PRAGMA table_info($table)').get();
      expect(
        columns.map((row) => row.data['name']),
        contains('server_id'),
        reason: '$table should persist the server identity',
      );
    }
    final indexes = await upgraded
        .customSelect("SELECT name FROM sqlite_master WHERE type = 'index'")
        .get();
    expect(
      indexes.map((row) => row.data['name']),
      containsAll([
        'ix_visits_owner_local_uuid',
        'ix_capture_assets_visit_role',
        'ix_camera_results_visit_version',
        'ix_measured_revisions_visit_revision',
      ]),
    );
    await upgraded.close();
  });
}
