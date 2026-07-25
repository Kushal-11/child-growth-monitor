import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';

import '../../services/poshan_setu_service.dart';
import '../database.dart';

/// Creates a manually-entered visit (no image, no ML) with its measurement and
/// a sync-queue entry, in a single transaction. Mirrors [VisitDao] but for the
/// manual-entry path (entry_method = 'manual').
///
/// The whole operation is atomic: if any insert throws, the transaction rolls
/// back and nothing partial is persisted.
class ManualVisitDao {
  ManualVisitDao(this._db);
  final AppDatabase _db;
  static const _uuid = Uuid();

  Future<int> createManualVisit({
    required int childId,
    required int ownerUserId,
    required double ageMonths,
    required DateTime visitDate,
    required double heightCm,
    required double weightKg,
    double? muacCm,
    double? hazZscore,
    double? whzZscore,
    String? hazStatus,
    String? whzStatus,
    String? muacMethod,
    String? notes,
  }) {
    return _db.transaction(() async {
      final resolvedMuacMethod =
          muacMethod ?? (muacCm == null ? 'unavailable' : 'manual');
      final child = await (_db.select(_db.children)
            ..where((candidate) =>
                candidate.id.equals(childId) &
                candidate.ownerUserId.equals(ownerUserId)))
          .getSingleOrNull();
      if (child == null) {
        throw StateError(
          'Child $childId was not found for the signed-in user.',
        );
      }
      final poshan = const PoshanSetuService().classify(
        sex: child.sex,
        ageMonths: ageMonths,
        heightCm: heightCm,
        heightSource: 'manual',
        weightKg: weightKg,
        weightSource: 'manual',
        muacCm: muacCm,
        muacSource: resolvedMuacMethod,
      );
      final visitId = await _db.into(_db.visits).insert(
            VisitsCompanion.insert(
              childId: childId,
              localUuid: _uuid.v4(),
              ageMonths: ageMonths,
              visitDate: Value(visitDate),
              notes: Value(notes),
              ownerUserId: Value(ownerUserId),
              entryMethod: const Value('manual'),
            ),
          );
      await _db.into(_db.measurements).insert(
            MeasurementsCompanion.insert(
              visitId: visitId,
              manualHeightCm: Value(heightCm),
              manualWeightKg: Value(weightKg),
              hazZscore: Value(hazZscore),
              whzZscore: Value(whzZscore),
              hazStatus: Value(hazStatus),
              whzStatus: Value(whzStatus),
              muacCm: Value(muacCm),
              muacStatus: Value(poshan.muacStatus),
              muacMethod: Value(resolvedMuacMethod),
              effectiveHeightCm: Value(heightCm),
              effectiveWeightKg: Value(weightKg),
              heightSource: const Value('manual'),
              weightSource: const Value('manual'),
              bmi: Value(poshan.bmi),
              bmiStatus: Value(poshan.bmiStatus),
              poshanStatus: Value(poshan.finalStatus),
              poshanTriggeredBy: Value(jsonEncode(poshan.triggeredBy)),
              classificationMethod: Value(poshan.classificationMethod),
              classificationRationale: Value(poshan.rationale),
            ),
          );
      await _db
          .into(_db.syncQueue)
          .insert(SyncQueueCompanion.insert(visitId: visitId));
      return visitId;
    });
  }
}
