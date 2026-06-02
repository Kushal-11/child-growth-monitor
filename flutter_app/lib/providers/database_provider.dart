import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../database/database.dart';
import '../database/daos/child_dao.dart';
import '../database/daos/manual_visit_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';

final databaseProvider = Provider<AppDatabase>((ref) {
  final db = AppDatabase();
  ref.onDispose(db.close);
  return db;
});

final childDaoProvider =
    Provider<ChildDao>((ref) => ChildDao(ref.watch(databaseProvider)));

final visitDaoProvider =
    Provider<VisitDao>((ref) => VisitDao(ref.watch(databaseProvider)));

final syncQueueDaoProvider =
    Provider<SyncQueueDao>((ref) => SyncQueueDao(ref.watch(databaseProvider)));

final manualVisitDaoProvider =
    Provider<ManualVisitDao>((ref) => ManualVisitDao(ref.watch(databaseProvider)));
