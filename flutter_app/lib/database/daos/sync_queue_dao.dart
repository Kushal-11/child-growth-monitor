import 'package:drift/drift.dart';
import '../database.dart';

class SyncQueueDao {
  final AppDatabase _db;
  SyncQueueDao(this._db);

  Future<int> enqueue(int visitId) => _db
      .into(_db.syncQueue)
      .insert(SyncQueueCompanion.insert(visitId: visitId));

  Stream<List<SyncQueueData>> watchPending(int ownerUserId) {
    final query = _db.select(_db.syncQueue).join([
      innerJoin(
        _db.visits,
        _db.visits.id.equalsExp(_db.syncQueue.visitId),
      ),
    ])
      ..where(
        (_db.syncQueue.status.equals('pending') |
                _db.syncQueue.status.equals('failed')) &
            _db.syncQueue.retryCount.isSmallerThanValue(5) &
            _db.visits.ownerUserId.equals(ownerUserId),
      )
      ..orderBy([OrderingTerm.asc(_db.syncQueue.createdAt)]);
    return query.watch().map(
          (rows) => rows
              .map((row) => row.readTable(_db.syncQueue))
              .toList(growable: false),
        );
  }

  Stream<int> watchPendingCount(int ownerUserId) {
    final count = _db.syncQueue.id.count();
    final query = _db.selectOnly(_db.syncQueue).join([
      innerJoin(
        _db.visits,
        _db.visits.id.equalsExp(_db.syncQueue.visitId),
      ),
    ])
      ..addColumns([count])
      ..where(
        (_db.syncQueue.status.equals('pending') |
                _db.syncQueue.status.equals('failed')) &
            _db.syncQueue.retryCount.isSmallerThanValue(5) &
            _db.visits.ownerUserId.equals(ownerUserId),
      );
    return query.watchSingle().map((row) => row.read(count) ?? 0);
  }

  Future<void> markSyncing(int id) =>
      (_db.update(_db.syncQueue)..where((s) => s.id.equals(id))).write(
          SyncQueueCompanion(
              status: const Value('syncing'),
              lastAttemptAt: Value(DateTime.now())));

  Future<void> markSynced(int id, {int? serverVisitId}) =>
      (_db.update(_db.syncQueue)..where((s) => s.id.equals(id))).write(
          SyncQueueCompanion(
              status: const Value('synced'),
              serverVisitId: Value(serverVisitId),
              lastAttemptAt: Value(DateTime.now())));

  Future<void> markFailed(int id, String error) async {
    final entry = await (_db.select(_db.syncQueue)
          ..where((s) => s.id.equals(id)))
        .getSingle();
    await (_db.update(_db.syncQueue)..where((s) => s.id.equals(id))).write(
        SyncQueueCompanion(
            status: const Value('failed'),
            retryCount: Value(entry.retryCount + 1),
            errorMessage: Value(error),
            lastAttemptAt: Value(DateTime.now())));
  }
}
