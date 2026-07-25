import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';
import 'package:child_growth_monitor_app/database/daos/sync_queue_dao.dart';

void main() {
  late AppDatabase db;
  late SyncQueueDao syncDao;
  late ChildDao childDao;
  late VisitDao visitDao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    syncDao = SyncQueueDao(db);
    childDao = ChildDao(db);
    visitDao = VisitDao(db);
  });
  tearDown(() => db.close());

  Future<int> createVisit() async {
    final child = await childDao.findOrCreate(
      name: 'Test',
      dateOfBirth: '2023-01-01',
      sex: 'M',
      ownerUserId: 1,
    );
    return visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 24.0,
      imagePath: '/test/image.jpg',
      ownerUserId: 1,
      measurement: const MeasurementsCompanion(),
    );
  }

  test('enqueue creates pending entry', () async {
    final visitId = await createVisit();
    await syncDao.enqueue(visitId);
    final pending = await syncDao.watchPending(1).first;
    expect(pending.length, 1);
    expect(pending.first.status, 'pending');
  });

  test('markSynced updates status', () async {
    final visitId = await createVisit();
    await syncDao.enqueue(visitId);
    final entries = await syncDao.watchPending(1).first;
    await syncDao.markSynced(entries.first.id, serverVisitId: 42);
    final updated = await syncDao.watchPending(1).first;
    expect(updated, isEmpty);
  });

  test('markFailed increments retryCount', () async {
    final visitId = await createVisit();
    await syncDao.enqueue(visitId);
    final entries = await syncDao.watchPending(1).first;
    await syncDao.markFailed(entries.first.id, 'Network error');
    final updated = await syncDao.watchPending(1).first;
    expect(updated.first.retryCount, 1);
    expect(updated.first.errorMessage, 'Network error');
  });
}
