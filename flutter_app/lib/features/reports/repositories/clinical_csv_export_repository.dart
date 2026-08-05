import 'package:drift/drift.dart';
import 'package:intl/intl.dart';

import '../../../database/database.dart';
import '../../guided_capture/domain/camera_screening_result.dart';
import '../domain/clinical_csv_record.dart';

abstract interface class ClinicalCsvExportRepository {
  Future<List<ClinicalCsvRecord>> loadSavedRecords({required int ownerUserId});
}

/// Reads every completed assessment/report owned by the signed-in field
/// worker. A completed record has either a measurement row or a persisted
/// camera result; incomplete capture drafts are intentionally excluded.
class DriftClinicalCsvExportRepository implements ClinicalCsvExportRepository {
  DriftClinicalCsvExportRepository(this._database);

  final AppDatabase _database;

  @override
  Future<List<ClinicalCsvRecord>> loadSavedRecords({
    required int ownerUserId,
  }) async {
    final query = _database.select(_database.visits).join([
      innerJoin(
        _database.children,
        _database.children.id.equalsExp(_database.visits.childId),
      ),
      leftOuterJoin(
        _database.measurements,
        _database.measurements.visitId.equalsExp(_database.visits.id),
      ),
    ])
      ..where(
        _database.visits.ownerUserId.equals(ownerUserId) |
            (_database.visits.ownerUserId.isNull() &
                _database.children.ownerUserId.equals(ownerUserId)),
      );
    final joinedRows = await query.get();
    final visitIds = joinedRows
        .map((row) => row.readTable(_database.visits).id)
        .toList(growable: false);

    final latestCameraResultByVisit = <int, CameraResult>{};
    if (visitIds.isNotEmpty) {
      final cameraResults = await (_database.select(_database.cameraResults)
            ..where((row) => row.visitId.isIn(visitIds))
            ..orderBy([
              (row) => OrderingTerm.desc(row.version),
              (row) => OrderingTerm.desc(row.createdAt),
            ]))
          .get();
      for (final result in cameraResults) {
        latestCameraResultByVisit.putIfAbsent(result.visitId, () => result);
      }
    }

    final records =
        <({ClinicalCsvRecord record, DateTime visitDate, int visitId})>[];
    for (final row in joinedRows) {
      final visit = row.readTable(_database.visits);
      final child = row.readTable(_database.children);
      final measurement = row.readTableOrNull(_database.measurements);
      final cameraResult = latestCameraResultByVisit[visit.id];
      if (measurement == null && cameraResult == null) continue;
      records.add((
        record: _toClinicalRecord(
          child: child,
          visit: visit,
          measurement: measurement,
          cameraResult: cameraResult,
        ),
        visitDate: visit.visitDate,
        visitId: visit.id,
      ));
    }
    records.sort((left, right) {
      final childOrder = left.record.childId.compareTo(right.record.childId);
      if (childOrder != 0) return childOrder;
      final dateOrder = left.visitDate.compareTo(right.visitDate);
      if (dateOrder != 0) return dateOrder;
      return left.visitId.compareTo(right.visitId);
    });
    return records.map((item) => item.record).toList(growable: false);
  }

  ClinicalCsvRecord _toClinicalRecord({
    required ChildrenData child,
    required Visit visit,
    required Measurement? measurement,
    required CameraResult? cameraResult,
  }) {
    final hasDirectMuac = _hasDirectMuac(measurement);
    final cameraUsesPopulationHeight =
        cameraResult?.heightSource == legacyWhoHeightSourceV1;
    final cameraUsesPopulationWeight =
        cameraResult?.weightSource == legacyWhoWeightSourceV1;
    final cameraHeightCm =
        cameraUsesPopulationHeight ? null : cameraResult?.estimatedHeightCm;
    final cameraWeightKg =
        cameraUsesPopulationWeight ? null : cameraResult?.estimatedWeightKg;
    final cameraHazZscore =
        cameraUsesPopulationHeight ? null : cameraResult?.estimatedHaz;
    final cameraWhzZscore =
        cameraUsesPopulationHeight || cameraUsesPopulationWeight
            ? null
            : cameraResult?.estimatedWhz;
    final cameraStuntingStatus = cameraUsesPopulationHeight
        ? null
        : cameraResult?.estimatedStuntingStatus;
    final cameraWastingStatus =
        cameraUsesPopulationHeight || cameraUsesPopulationWeight
            ? null
            : cameraResult?.estimatedWastingStatus;
    final predictedCategory = measurement?.poshanComplete == true
        ? measurement?.poshanStatus
        : cameraResult?.experimentalOverallCategory ??
            measurement?.combinedStatus ??
            measurement?.wastingStatus ??
            measurement?.whzStatus ??
            measurement?.poshanStatus;
    final measurementHeightCm = _isManualMethod(measurement?.heightMethod)
        ? null
        : measurement?.predictedHeightCm ?? measurement?.effectiveHeightCm;
    final calculatedHeightCm = cameraHeightCm ?? measurementHeightCm;
    final calculatedHeightMethod = cameraHeightCm != null
        ? _nonEmpty(cameraResult?.heightSource) ??
            _nonEmpty(cameraResult?.method)
        : measurementHeightCm != null
            ? _normaliseHeightEstimateMethod(
                _nonEmpty(measurement?.heightMethod) ??
                    _nonEmpty(measurement?.estimationMethod),
              )
            : null;
    final measurementWeightMethod = measurement?.weightMethod?.toLowerCase();
    final storedMlWeightKg = (_isManualMethod(measurementWeightMethod) ||
            measurementWeightMethod == 'ml_estimated')
        ? _positiveFinite(measurement?.mlEstimatedWeightKg)
        : null;
    final fallbackCalculatedWeightKg =
        _isManualMethod(measurement?.weightMethod)
            ? null
            : measurement?.predictedWeightKg ?? measurement?.effectiveWeightKg;
    final calculatedWeightKg =
        cameraWeightKg ?? storedMlWeightKg ?? fallbackCalculatedWeightKg;
    final calculatedWeightMethod = cameraWeightKg != null
        ? _nonEmpty(cameraResult?.weightSource) ??
            _nonEmpty(cameraResult?.method)
        : storedMlWeightKg != null
            ? experimentalMlWeightSourceV1
            : fallbackCalculatedWeightKg != null
                ? _normaliseWeightEstimateMethod(
                    _nonEmpty(measurement?.weightMethod),
                  )
                : null;
    return ClinicalCsvRecord(
      childId: child.id,
      childName: child.name,
      area: _nonEmpty(child.location),
      sex: child.sex.toUpperCase(),
      dateOfBirth: _formatStoredDate(child.dateOfBirth),
      measurementDate: DateFormat('yyyy-MM-dd').format(visit.visitDate),
      actualHeightCm: measurement?.manualHeightCm,
      calculatedHeightCm: calculatedHeightCm,
      calculatedHeightMethod: calculatedHeightMethod,
      actualWeightKg: measurement?.manualWeightKg,
      calculatedWeightKg: calculatedWeightKg,
      calculatedWeightMethod: calculatedWeightMethod,
      muacCm: hasDirectMuac ? measurement?.muacCm : null,
      calculatedMuacCm: hasDirectMuac ? null : measurement?.muacCm,
      muacStatus: _normaliseCategory(measurement?.muacStatus),
      muacMethod: _nonEmpty(measurement?.muacMethod),
      muacAgeInRange: measurement?.muacAgeInRange,
      muacConfidence: measurement?.muacConfidence,
      muacUncertaintyLowerCm: measurement?.muacUncertaintyLowerCm,
      muacUncertaintyUpperCm: measurement?.muacUncertaintyUpperCm,
      muacModelVersion: _nonEmpty(measurement?.muacModelVersion),
      muacCalibrationVersion: _nonEmpty(measurement?.muacCalibrationVersion),
      muacIsDirectMeasurement:
          measurement?.muacIsDirectMeasurement ?? (hasDirectMuac ? true : null),
      muacRequiresConfirmation: measurement?.muacRequiresConfirmation,
      muacReferralGuidance: _nonEmpty(measurement?.muacReferralGuidance),
      hazZscore: measurement?.hazZscore ?? cameraHazZscore,
      whzZscore: measurement?.whzZscore ?? cameraWhzZscore,
      // The mobile app does not capture the independent field-worker category.
      // Keep this blank rather than misrepresenting a computed classification
      // as manually observed ground truth.
      fieldCategory: null,
      predictedFieldCategory: _normaliseCategory(predictedCategory),
      stuntingPrediction: _normaliseCategory(
        measurement?.hazStatus ?? cameraStuntingStatus,
      ),
      wastingPrediction: _normaliseCategory(
        measurement?.whzStatus ??
            cameraWastingStatus ??
            measurement?.wastingStatus,
      ),
      notes: _buildNotes(
        visit: visit,
        measurement: measurement,
        cameraResult: cameraResult,
      ),
    );
  }

  bool _hasDirectMuac(Measurement? measurement) {
    if (measurement == null || measurement.muacCm == null) return false;
    if (measurement.muacIsDirectMeasurement == true) return true;
    return _isManualMethod(measurement.muacMethod) ||
        measurement.muacMethod?.toLowerCase() == 'tape_measured';
  }

  bool _isManualMethod(String? method) {
    final value = method?.trim().toLowerCase();
    return value == 'manual' || value == 'direct' || value == 'measured';
  }

  String _formatStoredDate(String raw) {
    final parsed = DateTime.tryParse(raw);
    return parsed == null ? raw : DateFormat('yyyy-MM-dd').format(parsed);
  }

  String? _normaliseCategory(String? value) {
    final trimmed = _nonEmpty(value);
    if (trimmed == null) return null;
    switch (trimmed.toUpperCase()) {
      case 'NORMAL':
        return 'Normal';
      case 'RISK_OVERWEIGHT':
      case 'RISK OVERWEIGHT':
        return 'Risk_Overweight';
      case 'OVERWEIGHT':
        return 'Overweight';
      case 'INDETERMINATE':
        return 'Indeterminate';
      default:
        return trimmed;
    }
  }

  String? _buildNotes({
    required Visit visit,
    required Measurement? measurement,
    required CameraResult? cameraResult,
  }) {
    final values = <String>[];
    void addOriginal(String? value) {
      final trimmed = _nonEmpty(value);
      if (trimmed != null && !values.contains(trimmed)) values.add(trimmed);
    }

    addOriginal(visit.notes);
    addOriginal(measurement?.measuredNotes);
    values.add('visit_uuid=${visit.localUuid}');
    values.add('entry_method=${visit.entryMethod}');
    if (_nonEmpty(measurement?.heightMethod) case final method?) {
      values.add('height_method=$method');
    }
    if (_nonEmpty(measurement?.weightMethod) case final method?) {
      values.add('weight_method=$method');
    }
    if (_nonEmpty(measurement?.muacMethod) case final method?) {
      values.add('muac_method=$method');
    }
    if (measurement?.muacRequiresConfirmation == true) {
      values.add('muac_requires_tape_confirmation=true');
    }
    if (measurement?.poshanComplete == false) {
      values.add('clinical_classification_incomplete=true');
    }
    if (cameraResult != null) {
      values.add('camera_method=${cameraResult.method}');
      values.add('model_version=${cameraResult.modelVersion}');
      if (cameraResult.nonClinical) values.add('non_clinical=true');
    }
    return values.isEmpty ? null : values.join('; ');
  }

  String? _nonEmpty(String? value) {
    final trimmed = value?.trim();
    return trimmed == null || trimmed.isEmpty ? null : trimmed;
  }

  String? _normaliseHeightEstimateMethod(String? method) {
    return method?.toLowerCase() == 'who_statistical'
        ? legacyWhoHeightSourceV1
        : method;
  }

  String? _normaliseWeightEstimateMethod(String? method) {
    return switch (method?.toLowerCase()) {
      'ml_estimated' => experimentalMlWeightSourceV1,
      'who_statistical' => legacyWhoWeightSourceV1,
      _ => method,
    };
  }

  double? _positiveFinite(double? value) {
    return value != null && value.isFinite && value > 0 ? value : null;
  }
}
