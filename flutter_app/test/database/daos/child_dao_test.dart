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

  test('findOrCreate creates new child', () async {
    final child = await dao.findOrCreate(
      name: 'Aarav',
      dateOfBirth: '2023-06-15',
      sex: 'M',
      ownerUserId: 1,
    );
    expect(child.id, greaterThan(0));
    expect(child.name, 'Aarav');
  });

  test('findOrCreate returns existing child', () async {
    final c1 = await dao.findOrCreate(
      name: 'Aarav',
      dateOfBirth: '2023-06-15',
      sex: 'M',
      ownerUserId: 1,
    );
    final c2 = await dao.findOrCreate(
      name: 'Aarav',
      dateOfBirth: '2023-06-15',
      sex: 'M',
      ownerUserId: 1,
    );
    expect(c1.id, c2.id);
  });

  test('watchForOwner returns only that owner children', () async {
    await dao.findOrCreate(
      name: 'A',
      dateOfBirth: '2023-01-01',
      sex: 'M',
      ownerUserId: 1,
    );
    await dao.findOrCreate(
      name: 'B',
      dateOfBirth: '2023-02-01',
      sex: 'F',
      ownerUserId: 1,
    );
    await dao.findOrCreate(
      name: 'Other',
      dateOfBirth: '2023-02-01',
      sex: 'F',
      ownerUserId: 2,
    );
    final all = await dao.watchForOwner(1).first;
    expect(all.length, 2);
  });

  test('watchForOwner filters by search query', () async {
    await dao.findOrCreate(
      name: 'Aarav',
      dateOfBirth: '2023-01-01',
      sex: 'M',
      ownerUserId: 1,
    );
    await dao.findOrCreate(
      name: 'Priya',
      dateOfBirth: '2023-02-01',
      sex: 'F',
      ownerUserId: 1,
    );
    final filtered = await dao.watchForOwner(1, search: 'pri').first;
    expect(filtered.length, 1);
    expect(filtered.first.name, 'Priya');
  });
}
