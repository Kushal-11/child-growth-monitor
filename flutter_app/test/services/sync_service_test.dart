import 'dart:io';

import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/sync_queue_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';
import 'package:child_growth_monitor_app/services/sync_service.dart';

Future<int> _seedVisit(AppDatabase db) async {
  final childDao = ChildDao(db);
  final visitDao = VisitDao(db);
  final syncDao = SyncQueueDao(db);
  final child = await childDao.findOrCreate(
    name: 'A',
    dateOfBirth: '2024-01-01',
    sex: 'F',
  );
  final tmp = File(
      '${Directory.systemTemp.path}/sync_${DateTime.now().microsecondsSinceEpoch}.jpg')
    ..writeAsBytesSync([1, 2, 3]);
  final visitId = await visitDao.createWithMeasurement(
    childId: child.id,
    ageMonths: 12,
    imagePath: tmp.path,
    measurement: const MeasurementsCompanion(),
  );
  return syncDao.enqueue(visitId);
}

void main() {
  late AppDatabase db;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
  });

  tearDown(() async => db.close());

  test('drains pending queue on success', () async {
    final queueId = await _seedVisit(db);
    final mockClient = MockClient((_) async {
      return http.Response('{"server_visit_id": 7, "status": "synced"}', 200);
    });
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: mockClient,
    );

    await svc.runOnce();

    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'synced');
    expect(entry.serverVisitId, 7);
  });

  test('treats already_synced as success', () async {
    final queueId = await _seedVisit(db);
    final mockClient = MockClient((_) async {
      return http.Response(
          '{"server_visit_id": 9, "status": "already_synced"}', 200);
    });
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: mockClient,
    );
    await svc.runOnce();
    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'synced');
  });

  test('marks failed and increments retry on 500', () async {
    final queueId = await _seedVisit(db);
    final mockClient = MockClient((_) async => http.Response('boom', 500));
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: mockClient,
    );
    await svc.runOnce();
    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'failed');
    expect(entry.retryCount, 1);
    expect(entry.errorMessage, contains('500'));
  });

  test('skips entries past 5 retries', () async {
    await _seedVisit(db);
    await db
        .update(db.syncQueue)
        .write(const SyncQueueCompanion(retryCount: Value(5)));
    var calls = 0;
    final mockClient = MockClient((_) async {
      calls++;
      return http.Response('{}', 200);
    });
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: mockClient,
    );
    await svc.runOnce();
    expect(calls, 0);
  });

  test('recovers entries stuck in syncing state on next runOnce', () async {
    final queueId = await _seedVisit(db);
    // Simulate a previous run that crashed mid-sync — entry left in 'syncing'.
    await (db.update(db.syncQueue)..where((s) => s.id.equals(queueId)))
        .write(const SyncQueueCompanion(status: Value('syncing')));

    final mockClient = MockClient((_) async {
      return http.Response('{"server_visit_id": 11, "status": "synced"}', 200);
    });
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: mockClient,
    );

    await svc.runOnce();

    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'synced');
    expect(entry.serverVisitId, 11);
  });
}
