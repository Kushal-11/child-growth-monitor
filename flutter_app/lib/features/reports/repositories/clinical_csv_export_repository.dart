import 'package:drift/drift.dart';
import 'package:intl/intl.dart';

import '../../../constants/config.dart';
import '../../../database/database.dart';
import '../../../services/muac_service.dart';
import '../../../services/who_data_service.dart';
import '../../guided_capture/domain/camera_screening_result.dart';
import '../domain/clinical_csv_record.dart';

const String whoMuacForAgeMedianV1 = 'who_muac_for_age_median_v1';

abstract interface class ClinicalCsvExportRepository {
  Future<List<ClinicalCsvRecord>> loadSavedRecords({required int ownerUserId});
}

/// Reads every completed assessment/report owned by the signed-in field
/// worker. A completed record has either a measurement row or a persisted
/// camera result; incomplete capture drafts are intentionally excluded.
class DriftClinicalCsvExportRepository implements ClinicalCsvExportRepository {
  DriftClinicalCsvExportRepository(
    this._database, {
    WhoDataService? whoData,
  }) : _who = whoData ?? WhoDataService();

  final AppDatabase _database;
  final WhoDataService _who;
  Future<void>? _whoLoadFuture;

  @override
  Future<List<ClinicalCsvRecord>> loadSavedRecords({
    required int ownerUserId,
  }) async {
    if (!_who.isLoaded) {
      await (_whoLoadFuture ??= _who.loadFromAssets());
    }
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
    final referenceTargets = _who.getReferenceTargets(
      child.sex,
      visit.ageMonths,
    );
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
    final whoHeightCm = _positiveFinite(
      referenceTargets.heightForAge?.target,
    );
    final calculatedHeightCm =
        cameraHeightCm ?? measurementHeightCm ?? whoHeightCm;
    final calculatedHeightMethod = cameraHeightCm != null
        ? _nonEmpty(cameraResult?.heightSource) ??
            _nonEmpty(cameraResult?.method)
        : measurementHeightCm != null
            ? _normaliseHeightEstimateMethod(
                _nonEmpty(measurement?.heightMethod) ??
                    _nonEmpty(measurement?.estimationMethod),
              )
            : whoHeightCm != null
                ? legacyWhoHeightSourceV1
                : null;
    final measurementWeightMethod = measurement?.weightMethod?.toLowerCase();
    final weightReferenceHeightCm = _positiveFinite(
      measurement?.manualHeightCm ?? calculatedHeightCm,
    );
    final whoMedianWeightKg = weightReferenceHeightCm == null
        ? null
        : _positiveFinite(
            _who.getMedianWeightForHeight(
              child.sex.toUpperCase(),
              weightReferenceHeightCm,
              ageMonths: visit.ageMonths,
            ),
          );
    final rawStoredMlWeightKg =
        _positiveFinite(measurement?.mlEstimatedWeightKg);
    final storedMlWeightKg = (_isManualMethod(measurementWeightMethod) ||
                measurementWeightMethod == 'ml_estimated') &&
            _isPlausibleMlWeight(
              rawStoredMlWeightKg,
              whoMedianWeightKg,
            )
        ? rawStoredMlWeightKg
        : null;
    final fallbackCalculatedWeightKg =
        _isManualMethod(measurement?.weightMethod)
            ? null
            : measurement?.predictedWeightKg ?? measurement?.effectiveWeightKg;
    final whoCalculatedWeightKg = whoMedianWeightKg == null
        ? null
        : whoMedianWeightKg *
            bodyBuildWeightAdjustment(measurement?.bodyBuild ?? 'average');
    final calculatedWeightKg = cameraWeightKg ??
        storedMlWeightKg ??
        fallbackCalculatedWeightKg ??
        whoCalculatedWeightKg;
    final calculatedWeightMethod = cameraWeightKg != null
        ? _nonEmpty(cameraResult?.weightSource) ??
            _nonEmpty(cameraResult?.method)
        : storedMlWeightKg != null
            ? experimentalMlWeightSourceV1
            : fallbackCalculatedWeightKg != null
                ? _normaliseWeightEstimateMethod(
                    _nonEmpty(measurement?.weightMethod),
                  )
                : whoCalculatedWeightKg != null
                    ? legacyWhoWeightSourceV1
                    : null;
    final calculatedMuac = _calculatedMuac(
      child: child,
      visit: visit,
      measurement: measurement,
      hasDirectMuac: hasDirectMuac,
      whoMedianMuacCm: referenceTargets.muacForAge?.target,
      whoLowerMuacCm: referenceTargets.muacForAge?.lower2Sd,
      whoUpperMuacCm: referenceTargets.muacForAge?.upper2Sd,
    );
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
      muacStatus:
          hasDirectMuac ? _normaliseCategory(measurement?.muacStatus) : null,
      muacMethod: hasDirectMuac ? _nonEmpty(measurement?.muacMethod) : null,
      muacAgeInRange: measurement?.muacAgeInRange ??
          (visit.ageMonths >= 6 && visit.ageMonths <= 59.9),
      muacIsDirectMeasurement: hasDirectMuac ? true : null,
      calculatedMuacCm: calculatedMuac?.muacCm,
      calculatedMuacMethod: calculatedMuac?.muacMethod,
      muacConfidence: calculatedMuac?.confidence,
      muacUncertaintyLowerCm: calculatedMuac?.uncertaintyLowerCm,
      muacUncertaintyUpperCm: calculatedMuac?.uncertaintyUpperCm,
      muacModelVersion: calculatedMuac?.modelVersion,
      muacCalibrationVersion: calculatedMuac?.calibrationVersion,
      muacRequiresConfirmation: calculatedMuac?.requiresConfirmation,
      muacReferralGuidance: _nonEmpty(calculatedMuac?.referralGuidance),
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
        calculatedHeightMethod: calculatedHeightMethod,
        calculatedWeightMethod: calculatedWeightMethod,
        calculatedMuacMethod: calculatedMuac?.muacMethod,
      ),
    );
  }

  MuacResult? _calculatedMuac({
    required ChildrenData child,
    required Visit visit,
    required Measurement? measurement,
    required bool hasDirectMuac,
    required double? whoMedianMuacCm,
    required double? whoLowerMuacCm,
    required double? whoUpperMuacCm,
  }) {
    final storedEstimate =
        hasDirectMuac ? null : _positiveFinite(measurement?.muacCm);
    if (storedEstimate != null) {
      return MuacResult(
        muacCm: storedEstimate,
        muacStatus: null,
        muacMethod: _nonEmpty(measurement?.muacMethod) ?? 'estimated',
        ageInRange: measurement?.muacAgeInRange ??
            (visit.ageMonths >= 6 && visit.ageMonths <= 59.9),
        confidence: measurement?.muacConfidence,
        uncertaintyLowerCm: measurement?.muacUncertaintyLowerCm,
        uncertaintyUpperCm: measurement?.muacUncertaintyUpperCm,
        modelVersion: _nonEmpty(measurement?.muacModelVersion),
        calibrationVersion: _nonEmpty(measurement?.muacCalibrationVersion),
        requiresConfirmation: measurement?.muacRequiresConfirmation ?? true,
        referralGuidance: _nonEmpty(measurement?.muacReferralGuidance) ??
            'Calculated MUAC requires confirmation with a tape.',
      );
    }

    final whzEstimate = MuacService.estimate(
      ageMonths: visit.ageMonths,
      sex: child.sex,
      whz: measurement?.whzZscore,
    );
    if (whzEstimate.muacCm != null) return whzEstimate;

    final median = _positiveFinite(whoMedianMuacCm);
    if (median == null) return null;
    return MuacResult(
      muacCm: median,
      muacStatus: null,
      muacMethod: whoMuacForAgeMedianV1,
      ageInRange: visit.ageMonths >= 6 && visit.ageMonths <= 59.9,
      confidence: 0.0,
      uncertaintyLowerCm: _positiveFinite(whoLowerMuacCm),
      uncertaintyUpperCm: _positiveFinite(whoUpperMuacCm),
      modelVersion: 'who_acfa_lms_v1',
      calibrationVersion: 'who_official_excel_reference_v1',
      requiresConfirmation: true,
      referralGuidance: 'WHO age/sex reference only; confirm MUAC with a tape.',
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
    required String? calculatedHeightMethod,
    required String? calculatedWeightMethod,
    required String? calculatedMuacMethod,
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
    if ({
      calculatedHeightMethod,
      calculatedWeightMethod,
      calculatedMuacMethod,
    }.any((method) =>
        method?.contains('who_') == true ||
        method == 'estimated_from_whz' ||
        method == experimentalMlWeightSourceV1)) {
      values.add('calculated_values_for_comparison_only=true');
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

  bool _isPlausibleMlWeight(double? value, double? whoMedianKg) {
    if (value == null || whoMedianKg == null) return false;
    return value >= whoMedianKg * mlWeightLowerBound &&
        value <= whoMedianKg * mlWeightUpperBound;
  }
}
