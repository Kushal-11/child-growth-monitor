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
    int? ownerUserId,
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
        ownerUserId: Value(ownerUserId),
      ),
    );
    return (_db.select(_db.children)..where((c) => c.id.equals(id))).getSingle();
  }

  Future<int> createChild({
    required String name,
    required String dateOfBirth,
    required String sex,
    String? guardianName,
    String? location,
    int? ownerUserId,
    String? photoPath,
  }) {
    return _db.into(_db.children).insert(
          ChildrenCompanion.insert(
            name: name,
            dateOfBirth: dateOfBirth,
            sex: sex,
            guardianName: Value(guardianName),
            location: Value(location),
            ownerUserId: Value(ownerUserId),
            photoPath: Value(photoPath),
          ),
        );
  }

  Future<void> updateChild({
    required int id,
    String? name,
    String? guardianName,
    String? location,
    String? photoPath,
  }) {
    return (_db.update(_db.children)..where((c) => c.id.equals(id))).write(
      ChildrenCompanion(
        name: name == null ? const Value.absent() : Value(name),
        guardianName: Value(guardianName),
        location: Value(location),
        photoPath: photoPath == null ? const Value.absent() : Value(photoPath),
        updatedAt: Value(DateTime.now()),
      ),
    );
  }

  Future<void> setArchived(int id, bool archived) {
    return (_db.update(_db.children)..where((c) => c.id.equals(id))).write(
      ChildrenCompanion(
        isArchived: Value(archived),
        updatedAt: Value(DateTime.now()),
      ),
    );
  }

  Stream<List<ChildrenData>> watchForOwner(int ownerUserId, {String? search}) {
    final query = _db.select(_db.children)
      ..where((c) => c.ownerUserId.equals(ownerUserId) & c.isArchived.equals(false))
      ..orderBy([(c) => OrderingTerm.desc(c.updatedAt)]);
    if (search != null && search.isNotEmpty) {
      query.where((c) => c.name.like('%$search%'));
    }
    return query.watch();
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
