import 'dart:io';
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import 'tables/children_table.dart';
import 'tables/visits_table.dart';
import 'tables/measurements_table.dart';
import 'tables/sync_queue_table.dart';
import 'tables/capture_assets_table.dart';
import 'tables/camera_results_table.dart';
import 'tables/measured_detail_revisions_table.dart';
import 'tables/sync_outbox_table.dart';

part 'database.g.dart';

@DriftDatabase(
  tables: [
    Children,
    Visits,
    Measurements,
    SyncQueue,
    CaptureAssets,
    CameraResults,
    MeasuredDetailRevisions,
    SyncOutbox,
  ],
)
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  /// For testing with in-memory database
  AppDatabase.forTesting(super.e);

  @override
  int get schemaVersion => 7;

  @override
  MigrationStrategy get migration => MigrationStrategy(
        onUpgrade: (migrator, from, to) async {
          Future<bool> hasColumn(String table, String column) async {
            final rows =
                await customSelect('PRAGMA table_info("$table")').get();
            return rows.any((row) => row.data['name'] == column);
          }

          if (from < 2) {
            // No production users yet — destructive recreate is acceptable.
            // Drop dependents first to respect foreign-key constraints; the
            // children table is left intact (no schema change needed there).
            await migrator.deleteTable('sync_queue');
            await migrator.deleteTable('measurements');
            await migrator.deleteTable('visits');
            await migrator.createTable(visits);
            await migrator.createTable(measurements);
            await migrator.createTable(syncQueue);
          }
          if (from < 3) {
            // children is never recreated above, so it always needs the new
            // columns.
            await migrator.addColumn(children, children.ownerUserId);
            await migrator.addColumn(children, children.photoPath);
            await migrator.addColumn(children, children.isArchived);
            // visits: when from < 2 it was just recreated with the current
            // schema (which already includes these columns), so only add them
            // for a v2 -> v3 upgrade to avoid a duplicate-column collision.
            if (from == 2) {
              await migrator.addColumn(visits, visits.ownerUserId);
              await migrator.addColumn(visits, visits.entryMethod);
            }
          }
          // From v1 the measurements table was recreated above using the
          // current schema, so only additive-upgrade existing v2/v3 tables.
          if (from >= 2 && from < 4) {
            await migrator.addColumn(
                measurements, measurements.effectiveHeightCm);
            await migrator.addColumn(
                measurements, measurements.effectiveWeightKg);
            await migrator.addColumn(measurements, measurements.heightMethod);
            await migrator.addColumn(measurements, measurements.weightMethod);
            await migrator.addColumn(measurements, measurements.bmi);
            await migrator.addColumn(measurements, measurements.bmiStatus);
            await migrator.addColumn(
                measurements, measurements.heightConfidence);
            await migrator.addColumn(
                measurements, measurements.weightConfidence);
            await migrator.addColumn(
                measurements, measurements.classificationConfidence);
            await migrator.addColumn(measurements, measurements.wastingMethod);
            await migrator.addColumn(measurements, measurements.muacAgeInRange);
            await migrator.addColumn(measurements, measurements.muacConfidence);
            await migrator.addColumn(
                measurements, measurements.muacUncertaintyLowerCm);
            await migrator.addColumn(
                measurements, measurements.muacUncertaintyUpperCm);
            await migrator.addColumn(
                measurements, measurements.muacModelVersion);
            await migrator.addColumn(
                measurements, measurements.muacCalibrationVersion);
            await migrator.addColumn(
                measurements, measurements.muacIsDirectMeasurement);
            await migrator.addColumn(
                measurements, measurements.muacRequiresConfirmation);
            await migrator.addColumn(
                measurements, measurements.muacReferralGuidance);
            await migrator.addColumn(measurements, measurements.combinedStatus);
            await migrator.addColumn(
                measurements, measurements.combinedTriggeredBy);
            await migrator.addColumn(
                measurements, measurements.combinedRationale);
            await migrator.addColumn(measurements, measurements.combinedMethod);
            await migrator.addColumn(
                measurements, measurements.combinedConfidenceScore);
            await migrator.addColumn(
                measurements, measurements.combinedProtocolVersion);
          }
          // v1 recreates the current table above; only existing v2-v4 tables
          // need the additive Poshan Setu evidence columns.
          if (from >= 2 && from < 5) {
            await migrator.addColumn(measurements, measurements.poshanStatus);
            await migrator.addColumn(
                measurements, measurements.poshanTriggeredBy);
            await migrator.addColumn(
                measurements, measurements.classificationMethod);
            await migrator.addColumn(
                measurements, measurements.classificationRationale);
            await migrator.addColumn(measurements, measurements.poshanComplete);
          }
          if (from >= 2 && from < 6) {
            await migrator.addColumn(visits, visits.captureState);
            await migrator.addColumn(visits, visits.captureStartedAt);
            await migrator.addColumn(visits, visits.captureCompletedAt);
            await migrator.addColumn(visits, visits.deviceMetadataJson);
            await migrator.addColumn(visits, visits.consentVersion);
            await migrator.addColumn(visits, visits.consentTimestamp);
            await migrator.addColumn(
              visits,
              visits.consentOperatorIdentifier,
            );
            await migrator.addColumn(visits, visits.mediaDeletedAt);

            await migrator.addColumn(
              measurements,
              measurements.measurementMode,
            );
            await migrator.addColumn(measurements, measurements.oedema);
            await migrator.addColumn(measurements, measurements.measuredAt);
            await migrator.addColumn(measurements, measurements.editorUserId);
            await migrator.addColumn(measurements, measurements.measuredNotes);
            await migrator.addColumn(
              measurements,
              measurements.whoAcuteStatus,
            );
            await migrator.addColumn(
              measurements,
              measurements.whoAcuteTriggeredBy,
            );
            await migrator.addColumn(
              measurements,
              measurements.whoAcuteRationale,
            );

            await customStatement(
              "UPDATE visits SET capture_state = CASE "
              "WHEN entry_method = 'manual' THEN 'measured_report' "
              "WHEN EXISTS (SELECT 1 FROM measurements m "
              "WHERE m.visit_id = visits.id) THEN 'estimated_report' "
              "ELSE 'incomplete_capture' END "
              "WHERE capture_state IS NULL",
            );
          }
          if (from < 6) {
            await migrator.createTable(captureAssets);
            await migrator.createTable(cameraResults);
            await migrator.createTable(measuredDetailRevisions);
            await migrator.createTable(syncOutbox);
            await customStatement(
              'CREATE INDEX IF NOT EXISTS ix_visits_owner_local_uuid '
              'ON visits (owner_user_id, local_uuid)',
            );
            await customStatement(
              'CREATE INDEX IF NOT EXISTS ix_capture_assets_visit_role '
              'ON capture_assets (visit_id, role)',
            );
            await customStatement(
              'CREATE INDEX IF NOT EXISTS ix_camera_results_visit_version '
              'ON camera_results (visit_id, version)',
            );
            await customStatement(
              'CREATE INDEX IF NOT EXISTS '
              'ix_measured_revisions_visit_revision '
              'ON measured_detail_revisions (visit_id, revision_number)',
            );
            await customStatement(
              'CREATE INDEX IF NOT EXISTS '
              'ix_sync_outbox_owner_status_created '
              'ON sync_outbox (owner_user_id, status, created_at)',
            );
            await customStatement(
              'CREATE INDEX IF NOT EXISTS ix_sync_outbox_visit_type '
              'ON sync_outbox (visit_uuid, entity_type)',
            );
          }
          // Visits already exist for v2+ databases, while the typed capture
          // tables are first created at v6. Add only the columns that were not
          // included by an earlier createTable call on each upgrade path.
          if (from >= 2 &&
              from < 7 &&
              !await hasColumn('visits', 'server_id')) {
            await migrator.addColumn(visits, visits.serverId);
          }
          if (from >= 6 && from < 7) {
            if (!await hasColumn('capture_assets', 'server_id')) {
              await migrator.addColumn(captureAssets, captureAssets.serverId);
            }
            if (!await hasColumn('camera_results', 'server_id')) {
              await migrator.addColumn(cameraResults, cameraResults.serverId);
            }
            if (!await hasColumn(
              'measured_detail_revisions',
              'server_id',
            )) {
              await migrator.addColumn(
                measuredDetailRevisions,
                measuredDetailRevisions.serverId,
              );
            }
          }
        },
        beforeOpen: (_) async {
          await customStatement('PRAGMA foreign_keys = ON');
        },
      );
}

LazyDatabase _openConnection() {
  return LazyDatabase(() async {
    final dbFolder = await getApplicationDocumentsDirectory();
    final file = File(p.join(dbFolder.path, 'child_growth_monitor.sqlite'));
    return NativeDatabase.createInBackground(file);
  });
}
