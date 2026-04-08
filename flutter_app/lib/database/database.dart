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
  int get schemaVersion => 1;
}

LazyDatabase _openConnection() {
  return LazyDatabase(() async {
    final dbFolder = await getApplicationDocumentsDirectory();
    final file = File(p.join(dbFolder.path, 'child_growth_monitor.sqlite'));
    return NativeDatabase.createInBackground(file);
  });
}
