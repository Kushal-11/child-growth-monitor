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
      final displayedHeight =
          m?.manualHeightCm ?? m?.effectiveHeightCm ?? m?.predictedHeightCm;
      final displayedWeight =
          m?.manualWeightKg ?? m?.effectiveWeightKg ?? m?.predictedWeightKg;
      final heightMethod =
          m?.heightMethod ?? (m?.manualHeightCm != null ? 'manual' : null);
      final weightMethod =
          m?.weightMethod ?? (m?.manualWeightKg != null ? 'manual' : null);
      final heightIsDirect =
          heightMethod == 'manual' || heightMethod == 'reference_object';
      final weightIsDirect = weightMethod == 'manual';
      return ChildVisit(
        visitId: v.id,
        visitDate: v.visitDate.toIso8601String(),
        ageMonths: v.ageMonths,
        measurement: m == null
            ? null
            : ChildVisitMeasurement(
                predictedHeightCm: displayedHeight,
                predictedWeightKg: displayedWeight,
                heightMethod: heightMethod,
                weightMethod: weightMethod,
                muacCm: m.muacCm,
                muacMethod: m.muacMethod,
                hazZscore: heightIsDirect ? m.hazZscore : null,
                whzZscore:
                    heightIsDirect && weightIsDirect ? m.whzZscore : null,
                hazStatus: heightIsDirect ? m.hazStatus : null,
                whzStatus:
                    heightIsDirect && weightIsDirect ? m.whzStatus : null,
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
