import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';

void main() {
  late AppDatabase db;
  late ChildDao childDao;
  late VisitDao visitDao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    childDao = ChildDao(db);
    visitDao = VisitDao(db);
  });

  tearDown(() async => db.close());

  test('createWithMeasurement assigns a non-empty localUuid', () async {
    final child = await childDao.findOrCreate(
      name: 'Test',
      dateOfBirth: '2024-01-01',
      sex: 'M',
      ownerUserId: 1,
    );
    final visitId = await visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 16,
      imagePath: '/tmp/front.jpg',
      ownerUserId: 1,
      measurement: const MeasurementsCompanion(),
    );
    final row = await visitDao.getById(visitId, ownerUserId: 1);
    expect(row, isNotNull);
    expect(row!.visit.localUuid.isNotEmpty, isTrue);
    expect(row.visit.localUuid.length, 36);
  });

  test('two visits get distinct localUuids', () async {
    final child = await childDao.findOrCreate(
      name: 'Test',
      dateOfBirth: '2024-01-01',
      sex: 'M',
      ownerUserId: 1,
    );
    final v1 = await visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 16,
      imagePath: '/tmp/a.jpg',
      ownerUserId: 1,
      measurement: const MeasurementsCompanion(),
    );
    final v2 = await visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 16,
      imagePath: '/tmp/b.jpg',
      ownerUserId: 1,
      measurement: const MeasurementsCompanion(),
    );
    final r1 = await visitDao.getById(v1, ownerUserId: 1);
    final r2 = await visitDao.getById(v2, ownerUserId: 1);
    expect(r1!.visit.localUuid, isNot(equals(r2!.visit.localUuid)));
  });

  test('visit creation and reads reject a different owner', () async {
    final child = await childDao.findOrCreate(
      name: 'Owned',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      ownerUserId: 1,
    );
    await expectLater(
      visitDao.createWithMeasurement(
        childId: child.id,
        ageMonths: 16,
        imagePath: '/tmp/front.jpg',
        ownerUserId: 2,
        measurement: const MeasurementsCompanion(),
      ),
      throwsStateError,
    );
    final visitId = await visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: 16,
      imagePath: '/tmp/front.jpg',
      ownerUserId: 1,
      measurement: const MeasurementsCompanion(),
    );
    expect(
      await visitDao.getById(visitId, ownerUserId: 2),
      isNull,
    );
    expect(
      await visitDao.watchByChildId(child.id, ownerUserId: 2).first,
      isEmpty,
    );
  });
}
