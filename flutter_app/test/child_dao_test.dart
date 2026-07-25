import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';

void main() {
  late AppDatabase db;
  late ChildDao dao;

  setUp(() {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    dao = ChildDao(db);
  });
  tearDown(() => db.close());

  test('createChild stores owner + photo', () async {
    final id = await dao.createChild(
      name: 'Kid',
      dateOfBirth: '2024-01-01',
      sex: 'M',
      guardianName: 'Mom',
      location: 'Village',
      ownerUserId: 7,
      photoPath: '/p.jpg',
    );
    final child = await dao.getById(id, ownerUserId: 7);
    expect(child!.ownerUserId, 7);
    expect(child.photoPath, '/p.jpg');
    expect(child.isArchived, false);
  });

  test('updateChild changes fields', () async {
    final id = await dao.createChild(
        name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.updateChild(
      id: id,
      ownerUserId: 1,
      name: 'Renamed',
      guardianName: 'Dad',
      location: 'Town',
      photoPath: '/q.jpg',
    );
    final child = await dao.getById(id, ownerUserId: 1);
    expect(child!.name, 'Renamed');
    expect(child.guardianName, 'Dad');
    expect(child.photoPath, '/q.jpg');
  });

  test('archive sets isArchived and watchForOwner excludes archived', () async {
    final id = await dao.createChild(
        name: 'Kid', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.setArchived(id, true, ownerUserId: 1);
    final child = await dao.getById(id, ownerUserId: 1);
    expect(child!.isArchived, true);
    final visible = await dao.watchForOwner(1).first;
    expect(visible.where((c) => c.id == id), isEmpty);
  });

  test('watchForOwner only returns that owner', () async {
    await dao.createChild(
        name: 'A', dateOfBirth: '2024-01-01', sex: 'M', ownerUserId: 1);
    await dao.createChild(
        name: 'B', dateOfBirth: '2024-01-01', sex: 'F', ownerUserId: 2);
    final forOne = await dao.watchForOwner(1).first;
    expect(forOne.map((c) => c.name), ['A']);
  });

  test('findOrCreate persists ownerUserId on new child', () async {
    final child = await dao.findOrCreate(
      name: 'Owned Child',
      dateOfBirth: '2022-01-01',
      sex: 'F',
      ownerUserId: 9001,
    );
    expect(child.ownerUserId, 9001);

    final fetched = await dao.getById(child.id, ownerUserId: 9001);
    expect(fetched!.ownerUserId, 9001);
  });

  test('same child identity is isolated between owners', () async {
    final first = await dao.findOrCreate(
      name: 'Same Child',
      dateOfBirth: '2023-01-01',
      sex: 'F',
      ownerUserId: 1,
    );
    final second = await dao.findOrCreate(
      name: 'Same Child',
      dateOfBirth: '2023-01-01',
      sex: 'F',
      ownerUserId: 2,
    );

    expect(second.id, isNot(first.id));
    expect(await dao.getById(first.id, ownerUserId: 2), isNull);
    await expectLater(
      dao.updateChild(
        id: first.id,
        ownerUserId: 2,
        name: 'Tampered',
      ),
      throwsStateError,
    );
    await expectLater(
      dao.setArchived(first.id, true, ownerUserId: 2),
      throwsStateError,
    );
    expect(
      (await dao.getById(first.id, ownerUserId: 1))!.name,
      'Same Child',
    );
  });
}
