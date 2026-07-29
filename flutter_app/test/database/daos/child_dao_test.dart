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
        name: 'Aarav', dateOfBirth: '2023-06-15', sex: 'M');
    expect(child.id, greaterThan(0));
    expect(child.name, 'Aarav');
  });

  test('findOrCreate returns existing child', () async {
    final c1 = await dao.findOrCreate(
        name: 'Aarav', dateOfBirth: '2023-06-15', sex: 'M');
    final c2 = await dao.findOrCreate(
        name: 'Aarav', dateOfBirth: '2023-06-15', sex: 'M');
    expect(c1.id, c2.id);
  });

  test('watchAll returns all children', () async {
    await dao.findOrCreate(name: 'A', dateOfBirth: '2023-01-01', sex: 'M');
    await dao.findOrCreate(name: 'B', dateOfBirth: '2023-02-01', sex: 'F');
    final all = await dao.watchAll().first;
    expect(all.length, 2);
  });

  test('watchAll filters by search query', () async {
    await dao.findOrCreate(name: 'Aarav', dateOfBirth: '2023-01-01', sex: 'M');
    await dao.findOrCreate(name: 'Priya', dateOfBirth: '2023-02-01', sex: 'F');
    final filtered = await dao.watchAll(search: 'pri').first;
    expect(filtered.length, 1);
    expect(filtered.first.name, 'Priya');
  });
}
