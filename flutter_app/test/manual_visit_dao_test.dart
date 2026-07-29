import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/manual_visit_dao.dart';

void main() {
  late AppDatabase db;
  late ChildDao childDao;
  late ManualVisitDao dao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    childDao = ChildDao(db);
    dao = ManualVisitDao(db);
  });
  tearDown(() => db.close());

  test(
      'createManualVisit stores visit (entry_method=manual) + measurement + sync queue',
      () async {
    final childId = await childDao.createChild(
        name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 5);
    final visitId = await dao.createManualVisit(
      childId: childId,
      ownerUserId: 5,
      ageMonths: 18.0,
      visitDate: DateTime(2026, 6, 1),
      heightCm: 80.0,
      weightKg: 10.5,
      muacCm: 13.0,
      hazZscore: -1.2,
      whzZscore: -0.5,
      hazStatus: 'Normal',
      whzStatus: 'Normal',
      muacStatus: 'Normal',
      notes: 'monthly visit',
    );

    final visit = await (db.select(db.visits)
          ..where((v) => v.id.equals(visitId)))
        .getSingle();
    expect(visit.entryMethod, 'manual');
    expect(visit.ownerUserId, 5);
    expect(visit.imagePath, isNull);
    expect(visit.notes, 'monthly visit');

    final m = await (db.select(db.measurements)
          ..where((x) => x.visitId.equals(visitId)))
        .getSingle();
    expect(m.manualHeightCm, 80.0);
    expect(m.manualWeightKg, 10.5);
    expect(m.muacCm, 13.0);
    expect(m.muacMethod, 'manual');

    final queued = await db.select(db.syncQueue).get();
    expect(queued.length, 1);
    expect(queued.first.visitId, visitId);
  });
}
