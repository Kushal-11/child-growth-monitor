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

Future<int> _seedVisit(AppDatabase db, {int ownerUserId = 1}) async {
  final childDao = ChildDao(db);
  final visitDao = VisitDao(db);
  final syncDao = SyncQueueDao(db);
  final child = await childDao.findOrCreate(
    name: 'A',
    dateOfBirth: '2024-01-01',
    sex: 'F',
    ownerUserId: ownerUserId,
  );
  final tmp = File(
      '${Directory.systemTemp.path}/sync_${DateTime.now().microsecondsSinceEpoch}.jpg')
    ..writeAsBytesSync([1, 2, 3]);
  final visitId = await visitDao.createWithMeasurement(
    childId: child.id,
    ageMonths: 12,
    imagePath: tmp.path,
    ownerUserId: ownerUserId,
    measurement: const MeasurementsCompanion(
      effectiveHeightCm: Value(100),
      effectiveWeightKg: Value(13.5),
      heightSource: Value('manual'),
      weightSource: Value('manual'),
      bmi: Value(13.5),
      bmiStatus: Value('Normal'),
      muacCm: Value(12.5),
      muacStatus: Value('Normal'),
      muacMethod: Value('manual'),
      poshanStatus: Value('Normal'),
      poshanTriggeredBy: Value('["bmi","muac"]'),
      classificationMethod: Value('poshan_setu_v1'),
      classificationRationale: Value('both measured'),
      mlModelVersion: Value('v1'),
      mlNonClinical: Value(true),
      mlTrainingData: Value('synthetic'),
    ),
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
      authToken: 'token',
      ownerUserId: 1,
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
      authToken: 'token',
      ownerUserId: 1,
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
      authToken: 'token',
      ownerUserId: 1,
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
      authToken: 'token',
      ownerUserId: 1,
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
      authToken: 'token',
      ownerUserId: 1,
    );

    await svc.runOnce();

    final entry = await (db.select(db.syncQueue)
          ..where((s) => s.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'synced');
    expect(entry.serverVisitId, 11);
  });

  test('multipart payload includes Poshan and artifact provenance', () async {
    await _seedVisit(db);
    late String body;
    final mockClient = MockClient((request) async {
      body = request.body;
      return http.Response('{"server_visit_id": 12, "status": "synced"}', 200);
    });
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: mockClient,
      authToken: 'token',
      ownerUserId: 1,
    );

    await svc.runOnce();

    expect(body, contains('effective_height_cm'));
    expect(body, contains('poshan_status'));
    expect(body, contains('poshan_setu_v1'));
    expect(body, contains('poshan_triggered_by'));
    expect(body, contains('["bmi","muac"]'));
    expect(body, contains('ml_model_version'));
    expect(body, contains('ml_non_clinical'));
    expect(body, contains('ml_training_data'));
  });

  test('does not sync without authenticated owner', () async {
    final queueId = await _seedVisit(db);
    var calls = 0;
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: MockClient((_) async {
        calls++;
        return http.Response('{}', 200);
      }),
    );

    await svc.runOnce();

    expect(calls, 0);
    final entry = await (db.select(db.syncQueue)
          ..where((row) => row.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'pending');
    expect(entry.retryCount, 0);
  });

  test('sync is scoped to current owner', () async {
    final queueId = await _seedVisit(db, ownerUserId: 2);
    var calls = 0;
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: MockClient((_) async {
        calls++;
        return http.Response('{}', 200);
      }),
      authToken: 'token',
      ownerUserId: 1,
    );

    await svc.runOnce();

    expect(calls, 0);
    final entry = await (db.select(db.syncQueue)
          ..where((row) => row.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'pending');
  });

  test('does not upload a visit whose child owner does not match', () async {
    final childId = await db.into(db.children).insert(
          ChildrenCompanion.insert(
            name: 'Mismatched child',
            dateOfBirth: '2024-01-01',
            sex: 'F',
            ownerUserId: const Value(2),
          ),
        );
    final visitId = await db.into(db.visits).insert(
          VisitsCompanion.insert(
            childId: childId,
            localUuid: '00000000-0000-0000-0000-000000000001',
            ageMonths: 12,
            ownerUserId: const Value(1),
          ),
        );
    final queueId = await db.into(db.syncQueue).insert(
          SyncQueueCompanion.insert(visitId: visitId),
        );
    var calls = 0;
    final svc = SyncService(
      db: db,
      visitDao: VisitDao(db),
      childDao: ChildDao(db),
      syncDao: SyncQueueDao(db),
      baseUrl: 'http://example.test',
      httpClient: MockClient((_) async {
        calls++;
        return http.Response('{}', 200);
      }),
      authToken: 'token',
      ownerUserId: 1,
    );

    await svc.runOnce();

    expect(calls, 0);
    final entry = await (db.select(db.syncQueue)
          ..where((row) => row.id.equals(queueId)))
        .getSingle();
    expect(entry.status, 'failed');
    expect(entry.errorMessage, contains('signed-in user'));
  });
}
