import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/child.dart';
import '../models/child_detail.dart';
import 'auth_provider.dart';
import 'database_provider.dart';

/// Watches all children from the local DB, with visit counts joined in.
final childrenProvider = StreamProvider<List<ChildSummary>>((ref) {
  final childDao = ref.watch(childDaoProvider);
  final db = ref.watch(databaseProvider);
  final ownerUserId = ref.watch(authProvider).user?.id;
  if (ownerUserId == null) return Stream.value(const <ChildSummary>[]);

  return childDao.watchForOwner(ownerUserId).asyncMap((rows) async {
    return Future.wait(rows.map((c) async {
      final countExpr = db.visits.id.count();
      final visitCount = await (db.selectOnly(db.visits)
            ..addColumns([countExpr])
            ..where(
              db.visits.childId.equals(c.id) &
                  db.visits.ownerUserId.equals(ownerUserId),
            ))
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
  final childDao = ref.watch(childDaoProvider);
  final visitDao = ref.watch(visitDaoProvider);
  final ownerUserId = ref.watch(authProvider).user?.id;

  if (ownerUserId == null) {
    return Stream<ChildDetail>.error(
      StateError('Sign in to view this child.'),
    );
  }

  return childDao
      .watchById(childId, ownerUserId: ownerUserId)
      .asyncExpand((child) {
    if (child == null) {
      return Stream<ChildDetail>.error(
        StateError('Child $childId not found for the signed-in user.'),
      );
    }

    return visitDao
        .watchByChildId(childId, ownerUserId: ownerUserId)
        .map((visitRows) {
      final visits = visitRows.map((pair) {
        final v = pair.visit;
        final m = pair.measurement;
        return ChildVisit(
          visitId: v.id,
          visitDate: v.visitDate.toIso8601String(),
          ageMonths: v.ageMonths,
          entryMethod: v.entryMethod,
          measurement: m == null
              ? null
              : ChildVisitMeasurement(
                  predictedHeightCm: m.predictedHeightCm,
                  predictedWeightKg: m.predictedWeightKg,
                  manualHeightCm: m.manualHeightCm,
                  manualWeightKg: m.manualWeightKg,
                  effectiveHeightCm: m.effectiveHeightCm,
                  effectiveWeightKg: m.effectiveWeightKg,
                  hazZscore: m.hazZscore,
                  whzZscore: m.whzZscore,
                  hazStatus: m.hazStatus,
                  whzStatus: m.whzStatus,
                  confidenceScore: m.confidenceScore,
                  heightSource: m.heightSource,
                  weightSource: m.weightSource,
                  bmi: m.bmi,
                  bmiStatus: m.bmiStatus,
                  muacCm: m.muacCm,
                  muacStatus: m.muacStatus,
                  muacMethod: m.muacMethod,
                  poshanStatus: m.poshanStatus,
                  poshanTriggeredBy: _decodeTriggeredBy(m.poshanTriggeredBy),
                  classificationMethod: m.classificationMethod,
                  classificationRationale: m.classificationRationale,
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
});

List<String> _decodeTriggeredBy(String? value) {
  if (value == null || value.isEmpty) return const [];
  try {
    final decoded = jsonDecode(value);
    if (decoded is List) {
      return decoded.whereType<String>().toList(growable: false);
    }
  } on FormatException {
    // Legacy comma-separated values are accepted for local history only.
  }
  return value
      .split(',')
      .map((part) => part.trim())
      .where((part) => part.isNotEmpty)
      .toList(growable: false);
}
