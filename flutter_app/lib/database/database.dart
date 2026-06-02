import 'dart:io';
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;

import 'tables/children_table.dart';
import 'tables/visits_table.dart';
import 'tables/measurements_table.dart';
import 'tables/sync_queue_table.dart';

part 'database.g.dart';

@DriftDatabase(tables: [Children, Visits, Measurements, SyncQueue])
class AppDatabase extends _$AppDatabase {
  AppDatabase() : super(_openConnection());

  /// For testing with in-memory database
  AppDatabase.forTesting(super.e);

  @override
  int get schemaVersion => 3;

  @override
  MigrationStrategy get migration => MigrationStrategy(
        onUpgrade: (migrator, from, to) async {
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
