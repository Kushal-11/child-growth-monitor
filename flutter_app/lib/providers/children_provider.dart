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
      return ChildVisit(
        visitId: v.id,
        visitDate: v.visitDate.toIso8601String(),
        ageMonths: v.ageMonths,
        measurement: m == null
            ? null
            : ChildVisitMeasurement(
                // Fall back to manual values when predicted are null so manual
                // visits surface in the detail row and growth chart.
                predictedHeightCm: m.predictedHeightCm ?? m.manualHeightCm,
                predictedWeightKg: m.predictedWeightKg ?? m.manualWeightKg,
                hazZscore: m.hazZscore,
                whzZscore: m.whzZscore,
                hazStatus: m.hazStatus,
                whzStatus: m.whzStatus,
                confidenceScore: m.confidenceScore,
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
