import 'package:drift/drift.dart';
import '../database.dart';

class ChildDao {
  final AppDatabase _db;
  ChildDao(this._db);

  Future<ChildrenData> findOrCreate({
    required String name,
    required String dateOfBirth,
    required String sex,
    String? guardianName,
    String? location,
  }) async {
    final existing = await (_db.select(_db.children)
          ..where((c) =>
              c.name.equals(name) &
              c.dateOfBirth.equals(dateOfBirth) &
              c.sex.equals(sex)))
        .getSingleOrNull();
    if (existing != null) return existing;

    final id = await _db.into(_db.children).insert(
      ChildrenCompanion.insert(
        name: name,
        dateOfBirth: dateOfBirth,
        sex: sex,
        guardianName: Value(guardianName),
        location: Value(location),
      ),
    );
    return (_db.select(_db.children)..where((c) => c.id.equals(id))).getSingle();
  }

  Stream<List<ChildrenData>> watchAll({String? search}) {
    final query = _db.select(_db.children)
      ..orderBy([(c) => OrderingTerm.desc(c.updatedAt)]);
    if (search != null && search.isNotEmpty) {
      query.where((c) => c.name.like('%$search%'));
    }
    return query.watch();
  }

  Future<ChildrenData?> getById(int id) =>
      (_db.select(_db.children)..where((c) => c.id.equals(id))).getSingleOrNull();

  Stream<ChildrenData?> watchById(int id) =>
      (_db.select(_db.children)..where((c) => c.id.equals(id))).watchSingleOrNull();
}
