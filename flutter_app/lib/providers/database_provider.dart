import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../database/database.dart';
import '../database/daos/child_dao.dart';
import '../database/daos/manual_visit_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';
import '../database/daos/guided_visit_dao.dart';
import '../database/daos/capture_asset_dao.dart';
import '../database/daos/camera_result_dao.dart';
import '../database/daos/measured_detail_revision_dao.dart';
import '../database/daos/sync_outbox_dao.dart';

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

final manualVisitDaoProvider = Provider<ManualVisitDao>(
    (ref) => ManualVisitDao(ref.watch(databaseProvider)));

final guidedVisitDaoProvider = Provider<GuidedVisitDao>(
    (ref) => GuidedVisitDao(ref.watch(databaseProvider)));

final captureAssetDaoProvider = Provider<CaptureAssetDao>(
    (ref) => CaptureAssetDao(ref.watch(databaseProvider)));

final cameraResultDaoProvider = Provider<CameraResultDao>(
    (ref) => CameraResultDao(ref.watch(databaseProvider)));

final measuredDetailRevisionDaoProvider = Provider<MeasuredDetailRevisionDao>(
  (ref) => MeasuredDetailRevisionDao(ref.watch(databaseProvider)),
);

final syncOutboxDaoProvider = Provider<SyncOutboxDao>(
    (ref) => SyncOutboxDao(ref.watch(databaseProvider)));
