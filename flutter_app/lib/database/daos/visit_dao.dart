import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';
import '../database.dart';

class VisitDao {
  final AppDatabase _db;
  VisitDao(this._db);

  static const _uuid = Uuid();

  Future<int> createWithMeasurement({
    required int childId,
    required double ageMonths,
    required String imagePath,
    String? sideImagePath,
    String? backImagePath,
    required MeasurementsCompanion measurement,
  }) async {
    return _db.transaction(() async {
      final visitId = await _db.into(_db.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: _uuid.v4(),
              ageMonths: ageMonths,
              imagePath: Value(imagePath),
              sideImagePath: Value(sideImagePath),
              backImagePath: Value(backImagePath),
            ),
          );
      await _db.into(_db.measurements).insert(
            measurement.copyWith(visitId: Value(visitId)),
          );
      return visitId;
    });
  }

  Stream<List<({Visit visit, Measurement? measurement})>> watchByChildId(
      int childId) {
    final query = _db.select(_db.visits).join([
      leftOuterJoin(
          _db.measurements, _db.measurements.visitId.equalsExp(_db.visits.id)),
    ])
      ..where(_db.visits.childId.equals(childId))
      ..orderBy([OrderingTerm.desc(_db.visits.visitDate)]);

    return query.watch().map((rows) => rows
        .map((row) => (
              visit: row.readTable(_db.visits),
              measurement: row.readTableOrNull(_db.measurements),
            ))
        .toList());
  }

  Future<({Visit visit, Measurement? measurement})?> getById(
      int visitId) async {
    final query = _db.select(_db.visits).join([
      leftOuterJoin(
          _db.measurements, _db.measurements.visitId.equalsExp(_db.visits.id)),
    ])
      ..where(_db.visits.id.equals(visitId));
    final row = await query.getSingleOrNull();
    if (row == null) return null;
    return (
      visit: row.readTable(_db.visits),
      measurement: row.readTableOrNull(_db.measurements)
    );
  }
}
