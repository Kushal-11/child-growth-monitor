import 'package:drift/drift.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/child.dart';
import '../models/child_detail.dart';
import 'database_provider.dart';

/// Watches all children from the local DB, with visit counts joined in.
final childrenProvider = StreamProvider<List<ChildSummary>>((ref) {
  final childDao = ref.watch(childDaoProvider);
  final db = ref.watch(databaseProvider);

  return childDao.watchAll().asyncMap((rows) async {
    return Future.wait(rows.map((c) async {
      final countExpr = db.visits.id.count();
      final visitCount = await (db.selectOnly(db.visits)
            ..addColumns([countExpr])
            ..where(db.visits.childId.equals(c.id)))
          .map((row) => row.read(countExpr) ?? 0)
          .getSingle();
      return ChildSummary(
        id: c.id,
        name: c.name,
        dateOfBirth: c.dateOfBirth,
        sex: c.sex,
        visitCount: visitCount,
      );
    }));
  });
});

/// Watches a single child + their visit history.
final childDetailProvider =
    StreamProvider.family<ChildDetail, int>((ref, childId) {
  final db = ref.watch(databaseProvider);
  final visitDao = ref.watch(visitDaoProvider);

  final childQuery = db.select(db.children)..where((t) => t.id.equals(childId));
  return childQuery.watchSingleOrNull().asyncMap((child) async {
    if (child == null) {
      throw StateError('Child $childId not found');
    }

    final visitRows = await visitDao.watchByChildId(childId).first;

    final visits = visitRows.map((pair) {
      final v = pair.visit;
      final m = pair.measurement;
      final directHeight = m?.manualHeightCm ??
          (m?.heightMethod == 'reference_object' ? m?.effectiveHeightCm : null);
      final directWeight = m?.manualWeightKg;
      return ChildVisit(
        visitId: v.id,
        visitDate: v.visitDate.toIso8601String(),
        ageMonths: v.ageMonths,
        measurement: m == null
            ? null
            : ChildVisitMeasurement(
                // Growth history must contain child measurements, never WHO,
                // ML, or WHZ-derived estimates presented as measured values.
                predictedHeightCm: directHeight,
                predictedWeightKg: directWeight,
                hazZscore: directHeight != null ? m.hazZscore : null,
                whzZscore: directHeight != null && directWeight != null
                    ? m.whzZscore
                    : null,
                hazStatus: directHeight != null ? m.hazStatus : null,
                whzStatus: directHeight != null && directWeight != null
                    ? m.whzStatus
                    : null,
                confidenceScore: null,
              ),
      );
    }).toList();

    return ChildDetail(
      id: child.id,
      name: child.name,
      dateOfBirth: child.dateOfBirth,
      sex: child.sex,
      guardianName: child.guardianName,
      location: child.location,
      visits: visits,
    );
  });
});
