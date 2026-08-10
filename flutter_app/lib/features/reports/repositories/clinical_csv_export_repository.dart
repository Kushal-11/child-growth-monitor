import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:intl/intl.dart';

import '../../../constants/config.dart';
import '../../../database/database.dart';
import '../../../services/muac_service.dart';
import '../../../services/nutrition_service.dart';
import '../../../services/poshan_setu_service.dart';
import '../../../services/who_data_service.dart';
import '../../ar_scan/domain/ar_scan_models.dart';
import '../../guided_capture/domain/camera_screening_result.dart';
import '../domain/clinical_csv_record.dart';

const _exportSchemaVersion = 'clinical_csv_v5_arcore_recovery';
const _whoStandardVersion = 'who_child_growth_standards_2006_lms';
const _whoActualAcuteMethod = 'who_imnci_measured_whz_muac_oedema_v1';
const _whoCalculatedAcuteMethod =
    'screening_only_calculated_whz_muac_no_oedema_v1';

abstract interface class ClinicalCsvExportRepository {
  Future<List<ClinicalCsvRecord>> loadSavedRecords({required int ownerUserId});
}

class _WhoScores {
  const _WhoScores({
    required this.bmi,
    required this.haz,
    required this.hazClassification,
    required this.hazQualityFlag,
    required this.whz,
    required this.whzClassification,
    required this.whzQualityFlag,
    required this.waz,
    required this.wazClassification,
    required this.wazQualityFlag,
    required this.baz,
    required this.bazClassification,
    required this.bazQualityFlag,
    required this.notes,
  });

  final double? bmi;
  final double? haz;
  final String? hazClassification;
  final String hazQualityFlag;
  final double? whz;
  final String? whzClassification;
  final String whzQualityFlag;
  final double? waz;
  final String? wazClassification;
  final String wazQualityFlag;
  final double? baz;
  final String? bazClassification;
  final String bazQualityFlag;
  final String? notes;
}

class _AcuteResult {
  const _AcuteResult({
    required this.status,
    required this.triggeredBy,
    required this.notes,
  });

  final String status;
  final List<String> triggeredBy;
  final String? notes;
}

typedef _PreviousActual = ({DateTime date, Measurement measurement});

/// Reads every completed assessment/report owned by the signed-in field
/// worker. Draft visits without a measurement or persisted camera result are
/// excluded. All WHO results are recomputed from same-basis values at export.
class DriftClinicalCsvExportRepository implements ClinicalCsvExportRepository {
  DriftClinicalCsvExportRepository(this._database, {WhoDataService? whoData})
    : _who = whoData ?? WhoDataService();

  final AppDatabase _database;
  final WhoDataService _who;
  late final NutritionService _nutrition = NutritionService(_who);
  Future<void>? _whoLoadFuture;

  @override
  Future<List<ClinicalCsvRecord>> loadSavedRecords({
    required int ownerUserId,
  }) async {
    if (!_who.isLoaded) {
      await (_whoLoadFuture ??= _who.loadFromAssets());
    }
    final query =
        _database.select(_database.visits).join([
          innerJoin(
            _database.children,
            _database.children.id.equalsExp(_database.visits.childId),
          ),
          leftOuterJoin(
            _database.measurements,
            _database.measurements.visitId.equalsExp(_database.visits.id),
          ),
        ])..where(
          _database.visits.ownerUserId.equals(ownerUserId) |
              (_database.visits.ownerUserId.isNull() &
                  _database.children.ownerUserId.equals(ownerUserId)),
        );
    final joinedRows = await query.get();
    final visitIds = joinedRows
        .map((row) => row.readTable(_database.visits).id)
        .toList(growable: false);

    final latestCameraResultByVisit = <int, CameraResult>{};
    final latestMeasuredRevisionByVisit = <int, MeasuredDetailRevision>{};
    if (visitIds.isNotEmpty) {
      final cameraResults =
          await (_database.select(_database.cameraResults)
                ..where((row) => row.visitId.isIn(visitIds))
                ..orderBy([
                  (row) => OrderingTerm.desc(row.version),
                  (row) => OrderingTerm.desc(row.createdAt),
                ]))
              .get();
      for (final result in cameraResults) {
        latestCameraResultByVisit.putIfAbsent(result.visitId, () => result);
      }

      final revisions =
          await (_database.select(_database.measuredDetailRevisions)
                ..where((row) => row.visitId.isIn(visitIds))
                ..orderBy([
                  (row) => OrderingTerm.desc(row.revisionNumber),
                  (row) => OrderingTerm.desc(row.createdAt),
                ]))
              .get();
      for (final revision in revisions) {
        latestMeasuredRevisionByVisit.putIfAbsent(
          revision.visitId,
          () => revision,
        );
      }
    }

    final chronologicalRows = [...joinedRows]
      ..sort((left, right) {
        final leftVisit = left.readTable(_database.visits);
        final rightVisit = right.readTable(_database.visits);
        final childOrder = leftVisit.childId.compareTo(rightVisit.childId);
        if (childOrder != 0) return childOrder;
        final dateOrder = leftVisit.visitDate.compareTo(rightVisit.visitDate);
        if (dateOrder != 0) return dateOrder;
        return leftVisit.id.compareTo(rightVisit.id);
      });
    final previousActualByChild = <int, _PreviousActual>{};
    final records =
        <({ClinicalCsvRecord record, DateTime visitDate, int visitId})>[];

    for (final row in chronologicalRows) {
      final visit = row.readTable(_database.visits);
      final child = row.readTable(_database.children);
      final measurement = row.readTableOrNull(_database.measurements);
      final cameraResult = latestCameraResultByVisit[visit.id];
      if (measurement == null && cameraResult == null) continue;
      final previousActual = previousActualByChild[child.id];
      records.add((
        record: _toClinicalRecord(
          child: child,
          visit: visit,
          measurement: measurement,
          cameraResult: cameraResult,
          measuredRevision: latestMeasuredRevisionByVisit[visit.id],
          previousActual: previousActual,
        ),
        visitDate: visit.visitDate,
        visitId: visit.id,
      ));
      if (_hasAnyActual(measurement)) {
        previousActualByChild[child.id] = (
          date: _dateOnly(visit.visitDate),
          measurement: measurement!,
        );
      }
    }

    records.sort((left, right) {
      final nameOrder = left.record.childName.toLowerCase().compareTo(
        right.record.childName.toLowerCase(),
      );
      if (nameOrder != 0) return nameOrder;
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
    required MeasuredDetailRevision? measuredRevision,
    required _PreviousActual? previousActual,
  }) {
    final visitDate = _dateOnly(visit.visitDate);
    final dateOfBirth = _parseDate(child.dateOfBirth);
    final rawAgeDays = dateOfBirth == null
        ? null
        : visitDate.difference(_dateOnly(dateOfBirth)).inDays;
    final ageDays = rawAgeDays != null && rawAgeDays >= 0 ? rawAgeDays : null;
    final whoAgeMonths = ageDays == null ? null : ageDays / 30.4375;
    final ageForWho = whoAgeMonths != null && whoAgeMonths < 60
        ? whoAgeMonths
        : null;
    final measurementMode = _normaliseMeasurementMode(
      measurement?.measurementMode,
    );
    final expectedMeasurementMode = ageDays == null
        ? null
        : ageDays < 731
        ? 'recumbent_length'
        : 'standing_height';
    final positionAdjustment = _positionAdjustment(
      ageDays: ageDays,
      measurementMode: measurementMode,
    );
    final oedema = _normaliseOedema(measurement?.oedema);
    final oedemaPresent = oedema == 'Yes';
    final arScan = _arScanFromMetadata(visit.deviceMetadataJson);
    final arHeightCm = _positiveFinite(arScan?.estimatedHeightCm);
    final arCameraResult =
        arScan != null &&
            cameraResult?.method == cameraScreeningContactlessMethodV2
        ? cameraResult
        : null;

    final hasDirectMuac = _hasDirectMuac(measurement);
    final actualHeightCm = _positiveFinite(measurement?.manualHeightCm);
    final actualWeightKg = _positiveFinite(measurement?.manualWeightKg);
    final actualMuacCm = hasDirectMuac
        ? _positiveFinite(measurement?.muacCm)
        : null;

    final cameraUsesPopulationHeight =
        cameraResult?.heightSource == legacyWhoHeightSourceV1;
    final cameraUsesPopulationWeight =
        cameraResult?.weightSource == legacyWhoWeightSourceV1;
    final cameraHeightCm = cameraUsesPopulationHeight
        ? null
        : _positiveFinite(cameraResult?.estimatedHeightCm);
    final cameraWeightKg = cameraUsesPopulationWeight
        ? null
        : _positiveFinite(cameraResult?.estimatedWeightKg);

    final storedPredictedHeightCm = _positiveFinite(
      measurement?.predictedHeightCm,
    );
    final storedHeightPredictionMethod = _storedHeightPredictionMethod(
      measurement,
    );
    final storedHeightDuplicatesManual =
        _isManualMethod(measurement?.heightMethod) &&
        _sameMeasurement(storedPredictedHeightCm, actualHeightCm);
    final storedHeightUsesPopulationReference = _isPopulationHeightMethod(
      storedHeightPredictionMethod,
    );
    final eligibleStoredPredictedHeightCm =
        storedHeightDuplicatesManual || storedHeightUsesPopulationReference
        ? null
        : storedPredictedHeightCm;
    final fallbackHeightMethod =
        _nonEmpty(measurement?.heightMethod) ??
        _nonEmpty(measurement?.estimationMethod);
    final fallbackHeightUsesPopulationReference = _isPopulationHeightMethod(
      fallbackHeightMethod,
    );
    final fallbackEstimatedHeightCm =
        _isManualMethod(measurement?.heightMethod) ||
            fallbackHeightUsesPopulationReference
        ? null
        : _positiveFinite(measurement?.effectiveHeightCm);
    final calculatedHeightCm =
        cameraHeightCm ??
        arHeightCm ??
        eligibleStoredPredictedHeightCm ??
        fallbackEstimatedHeightCm;
    final calculatedHeightMethod = cameraHeightCm != null
        ? _nonEmpty(cameraResult?.heightSource) ??
              _nonEmpty(cameraResult?.method)
        : arHeightCm != null
        ? arScan?.method
        : eligibleStoredPredictedHeightCm != null
        ? storedHeightPredictionMethod
        : fallbackEstimatedHeightCm != null
        ? _normaliseHeightEstimateMethod(
            _nonEmpty(measurement?.heightMethod) ??
                _nonEmpty(measurement?.estimationMethod),
          )
        : null;
    final calculatedHeightAvailability = calculatedHeightCm != null
        ? 'available'
        : cameraUsesPopulationHeight ||
              storedHeightUsesPopulationReference ||
              fallbackHeightUsesPopulationReference
        ? 'population_reference_suppressed'
        : storedHeightDuplicatesManual || actualHeightCm != null
        ? 'not_independently_recorded'
        : 'unavailable';

    // A measured or population-reference height may be used only to validate
    // the stored ML weight's broad safety bounds. It is never promoted into
    // calculated-height evidence or used for calculated WHZ/BAZ.
    final weightValidationHeightCm =
        calculatedHeightCm ??
        actualHeightCm ??
        _positiveFinite(measurement?.effectiveHeightCm);
    final whoMedianWeightKg =
        weightValidationHeightCm == null || ageForWho == null
        ? null
        : _positiveFinite(
            _who.getMedianWeightForHeight(
              child.sex.toUpperCase(),
              _adjustHeight(weightValidationHeightCm, positionAdjustment),
              ageMonths: ageForWho,
            ),
          );
    final rawStoredMlWeightKg = _positiveFinite(
      measurement?.mlEstimatedWeightKg,
    );
    final storedMlWeightKg =
        _isPlausibleMlWeight(rawStoredMlWeightKg, whoMedianWeightKg)
        ? rawStoredMlWeightKg
        : null;
    final storedPredictedWeightKg = _positiveFinite(
      measurement?.predictedWeightKg,
    );
    final predictedDuplicatesActual = _sameMeasurement(
      storedPredictedWeightKg,
      actualWeightKg,
    );
    final fallbackEstimatedWeightKg = _isManualMethod(measurement?.weightMethod)
        ? null
        : _positiveFinite(measurement?.effectiveWeightKg);
    final calculatedWeightCandidate =
        storedMlWeightKg ??
        (predictedDuplicatesActual ? null : storedPredictedWeightKg) ??
        fallbackEstimatedWeightKg;
    final calculatedWeightKg = cameraWeightKg ?? calculatedWeightCandidate;
    final calculatedWeightMethod = cameraWeightKg != null
        ? _nonEmpty(cameraResult?.weightSource) ??
              _nonEmpty(cameraResult?.method)
        : storedMlWeightKg != null
        ? experimentalMlWeightSourceV1
        : calculatedWeightCandidate != null
        ? _normaliseWeightEstimateMethod(
            _nonEmpty(measurement?.weightMethod) ??
                _nonEmpty(measurement?.estimationMethod),
          )
        : null;
    final calculatedWeightAvailability = calculatedWeightKg != null
        ? 'available'
        : rawStoredMlWeightKg != null
        ? 'implausible_or_unverifiable_ml_weight_suppressed'
        : actualWeightKg != null
        ? 'not_independently_recorded'
        : 'unavailable';

    final actualAdjustedHeight = _adjustNullableHeight(
      actualHeightCm,
      positionAdjustment,
    );
    final calculatedAdjustedHeight = _adjustNullableHeight(
      calculatedHeightCm,
      positionAdjustment,
    );
    final actualWho = _computeWhoScores(
      sex: child.sex,
      ageMonths: ageForWho,
      heightCm: actualAdjustedHeight,
      weightKg: actualWeightKg,
      suppressWeightScoresForOedema: oedemaPresent,
    );
    final calculatedWho = _computeWhoScores(
      sex: child.sex,
      ageMonths: ageForWho,
      heightCm: calculatedAdjustedHeight,
      weightKg: calculatedWeightKg,
      suppressWeightScoresForOedema: false,
    );

    final calculatedMuac = _calculatedMuac(
      measurement: measurement,
      cameraResult: cameraResult,
      arScan: arScan,
      hasDirectMuac: hasDirectMuac,
      ageMonths: ageForWho,
    );
    final muacAgeInRange =
        ageForWho != null &&
        ageForWho >= poshanMuacMinAgeMonths &&
        ageForWho < poshanMuacMaxAgeMonths;
    final actualMuacStatus = _classifyMuac(
      actualMuacCm,
      ageInRange: muacAgeInRange,
    );
    final calculatedMuacStatus = _classifyMuac(
      _positiveFinite(calculatedMuac?.muacCm),
      ageInRange: muacAgeInRange,
    );
    final calculatedMuacAvailability = calculatedMuac != null
        ? 'available'
        : hasDirectMuac
        ? 'not_independently_recorded'
        : 'unavailable';
    final actualAcute = _classifyActualAcute(
      ageMonths: ageForWho,
      whz: actualWho.whzQualityFlag == 'OK' ? actualWho.whz : null,
      muacCm: actualMuacCm,
      oedema: oedema,
    );
    final calculatedAcute = _classifyCalculatedAcute(
      ageMonths: ageForWho,
      whz: calculatedWho.whzQualityFlag == 'OK' ? calculatedWho.whz : null,
      muacCm: _positiveFinite(calculatedMuac?.muacCm),
    );
    final poshan = const PoshanSetuService().classify(
      sex: child.sex,
      ageMonths: ageForWho ?? visit.ageMonths,
      heightCm: actualHeightCm,
      heightSource: actualHeightCm == null ? 'unavailable' : 'manual',
      weightKg: actualWeightKg,
      weightSource: actualWeightKg == null ? 'unavailable' : 'manual',
      muacCm: actualMuacCm,
      muacSource: actualMuacCm == null ? 'unavailable' : 'tape',
    );

    final overallRaw = measurement?.poshanComplete == true
        ? measurement?.poshanStatus
        : measurement?.whoAcuteStatus ??
              measurement?.combinedStatus ??
              cameraResult?.experimentalOverallCategory ??
              measurement?.wastingStatus ??
              measurement?.whzStatus ??
              measurement?.poshanStatus;
    final previousMeasurement = previousActual?.measurement;
    final previousDate = previousActual?.date;
    final previousActualMuac = _hasDirectMuac(previousMeasurement)
        ? _positiveFinite(previousMeasurement?.muacCm)
        : null;

    return ClinicalCsvRecord(
      exportSchemaVersion: _exportSchemaVersion,
      childName: child.name,
      childId: child.id,
      guardianName: _nonEmpty(child.guardianName),
      area: _nonEmpty(child.location),
      sex: child.sex.toUpperCase(),
      dateOfBirth: _formatStoredDate(child.dateOfBirth),
      measurementDate: DateFormat('yyyy-MM-dd').format(visitDate),
      ageDays: ageDays,
      recordedAgeMonths: visit.ageMonths,
      whoAgeMonths: whoAgeMonths,
      visitUuid: visit.localUuid,
      entryMethod: visit.entryMethod,
      captureState: _nonEmpty(visit.captureState),
      consentVersion: _nonEmpty(visit.consentVersion),
      consentTimestamp: visit.consentTimestamp?.toIso8601String(),
      consentOperatorIdentifier: _nonEmpty(visit.consentOperatorIdentifier),
      measurementMode: measurementMode,
      whoExpectedMeasurementMode: expectedMeasurementMode,
      positionAdjustmentCm: positionAdjustment,
      oedema: oedema,
      // The current UI records presence, absence or not-checked, not grade.
      // Keep this blank rather than fabricating +/++/+++.
      oedemaGrade: null,
      measuredAt: measurement?.measuredAt?.toIso8601String(),
      measurementUpdateReason: _nonEmpty(measuredRevision?.reason),
      actualHeightCm: actualHeightCm,
      actualHeightMethod: actualHeightCm == null
          ? null
          : _normaliseDirectMethod(measurement?.heightMethod),
      actualWhoAdjustedHeightCm: actualAdjustedHeight,
      calculatedHeightCm: calculatedHeightCm,
      calculatedHeightMethod: calculatedHeightMethod,
      calculatedHeightConfidence: calculatedHeightCm == null
          ? null
          : arHeightCm != null && calculatedHeightCm == arHeightCm
          ? _finite(arScan?.qualityScore)
          : _finite(measurement?.heightConfidence) ??
                _finite(measurement?.confidenceScore),
      calculatedWhoAdjustedHeightCm: calculatedAdjustedHeight,
      calculatedHeightAvailability: calculatedHeightAvailability,
      heightErrorCm: _difference(calculatedHeightCm, actualHeightCm),
      actualWeightKg: actualWeightKg,
      actualWeightMethod: actualWeightKg == null
          ? null
          : _normaliseDirectMethod(measurement?.weightMethod),
      calculatedWeightKg: calculatedWeightKg,
      calculatedWeightMethod: calculatedWeightMethod,
      calculatedWeightConfidence: calculatedWeightKg == null
          ? null
          : calculatedWeightMethod == experimentalMlWeightSourceV1
          ? null
          : _finite(measurement?.weightConfidence) ??
                _finite(measurement?.confidenceScore),
      calculatedWeightAvailability: calculatedWeightAvailability,
      weightErrorKg: _difference(calculatedWeightKg, actualWeightKg),
      actualMuacCm: actualMuacCm,
      actualMuacStatus: actualMuacStatus,
      actualMuacMethod: actualMuacCm == null
          ? null
          : _normaliseDirectMethod(measurement?.muacMethod),
      actualMuacIsDirectMeasurement: actualMuacCm == null ? null : true,
      muacAgeInRange: ageForWho == null ? null : muacAgeInRange,
      calculatedMuacCm: _positiveFinite(calculatedMuac?.muacCm),
      calculatedMuacStatus: calculatedMuacStatus,
      calculatedMuacMethod: _nonEmpty(calculatedMuac?.muacMethod),
      calculatedMuacConfidence: _finite(calculatedMuac?.confidence),
      calculatedMuacUncertaintyLowerCm: _positiveFinite(
        calculatedMuac?.uncertaintyLowerCm,
      ),
      calculatedMuacUncertaintyUpperCm: _positiveFinite(
        calculatedMuac?.uncertaintyUpperCm,
      ),
      calculatedMuacModelVersion: _nonEmpty(calculatedMuac?.modelVersion),
      calculatedMuacCalibrationVersion: _nonEmpty(
        calculatedMuac?.calibrationVersion,
      ),
      calculatedMuacRequiresConfirmation: calculatedMuac?.requiresConfirmation,
      calculatedMuacReferralGuidance: _nonEmpty(
        calculatedMuac?.referralGuidance,
      ),
      calculatedMuacAvailability: calculatedMuacAvailability,
      muacErrorCm: _difference(
        _positiveFinite(calculatedMuac?.muacCm),
        actualMuacCm,
      ),
      actualBmi: actualWho.bmi,
      actualHazZscore: actualWho.haz,
      actualStuntingClassification: actualWho.hazClassification,
      actualHazQualityFlag: actualWho.hazQualityFlag,
      actualWhzZscore: actualWho.whz,
      actualWastingClassification: actualWho.whzClassification,
      actualWhzQualityFlag: actualWho.whzQualityFlag,
      actualWazZscore: actualWho.waz,
      actualUnderweightClassification: actualWho.wazClassification,
      actualWazQualityFlag: actualWho.wazQualityFlag,
      actualBazZscore: actualWho.baz,
      actualBmiForAgeClassification: actualWho.bazClassification,
      actualBazQualityFlag: actualWho.bazQualityFlag,
      actualWhoCalculationNotes: actualWho.notes,
      calculatedBmi: calculatedWho.bmi,
      calculatedHazZscore: calculatedWho.haz,
      calculatedStuntingPrediction: calculatedWho.hazClassification,
      calculatedHazQualityFlag: calculatedWho.hazQualityFlag,
      calculatedWhzZscore: calculatedWho.whz,
      calculatedWastingPrediction: calculatedWho.whzClassification,
      calculatedWhzQualityFlag: calculatedWho.whzQualityFlag,
      calculatedWazZscore: calculatedWho.waz,
      calculatedUnderweightPrediction: calculatedWho.wazClassification,
      calculatedWazQualityFlag: calculatedWho.wazQualityFlag,
      calculatedBazZscore: calculatedWho.baz,
      calculatedBmiForAgePrediction: calculatedWho.bazClassification,
      calculatedBazQualityFlag: calculatedWho.bazQualityFlag,
      calculatedWhoCalculationNotes: calculatedWho.notes,
      whoStandardVersion: _whoStandardVersion,
      actualAcuteNutritionClassification: actualAcute.status,
      actualAcuteTriggeredBy: _encodeTriggers(actualAcute.triggeredBy),
      actualAcuteMethod: _whoActualAcuteMethod,
      actualAcuteCalculationNotes: actualAcute.notes,
      calculatedAcuteNutritionPrediction: calculatedAcute.status,
      calculatedAcuteTriggeredBy: _encodeTriggers(calculatedAcute.triggeredBy),
      calculatedAcuteMethod: _whoCalculatedAcuteMethod,
      calculatedAcuteScreeningOnly: true,
      arcoreScanAvailable: arScan != null,
      arcoreMethod: arScan?.method,
      arcoreDepthHeightCm: arHeightCm,
      arcoreHeightUncertaintyCm: _finite(arScan?.uncertaintyCm),
      arcoreHeightRangeLowerCm: _positiveFinite(arScan?.heightRangeLowerCm),
      arcoreHeightRangeUpperCm: _positiveFinite(arScan?.heightRangeUpperCm),
      arcoreGeometryMlWeightKg: _positiveFinite(
        arCameraResult?.estimatedWeightKg,
      ),
      arcoreWeightRangeLowerKg: _positiveFinite(
        arCameraResult?.weightRangeLowerKg,
      ),
      arcoreWeightRangeUpperKg: _positiveFinite(
        arCameraResult?.weightRangeUpperKg,
      ),
      arcoreArmMuacCm: _positiveFinite(arScan?.estimatedMuacCm),
      arcoreMuacUncertaintyCm: _positiveFinite(arScan?.muacUncertaintyCm),
      arcoreMuacRangeLowerCm: _positiveFinite(arScan?.muacRangeLowerCm),
      arcoreMuacRangeUpperCm: _positiveFinite(arScan?.muacRangeUpperCm),
      arcoreQualityScore: _finite(arScan?.qualityScore),
      arcoreGeometryQualityScore: _finite(arScan?.geometryQualityScore),
      arcorePoseQualityScore: _finite(arScan?.poseQualityScore),
      arcoreAcceptedKeyframes: arScan?.acceptedKeyframes,
      arcoreDepthConfidence: _finite(arScan?.meanDepthConfidence),
      arcoreCoverageDegrees: _finite(arScan?.scanCoverageDegrees),
      arcoreFloorStabilityCm: _finite(arScan?.floorStabilityCm),
      arcoreShoulderWidthCm: _positiveFinite(arScan?.shoulderWidthCm),
      arcoreHipWidthCm: _positiveFinite(arScan?.hipWidthCm),
      arcoreTorsoLengthCm: _positiveFinite(arScan?.torsoLengthCm),
      arcoreUpperArmLengthCm: _positiveFinite(arScan?.upperArmLengthCm),
      arcoreChestDepthCm: _positiveFinite(arScan?.chestDepthCm),
      arcoreAbdomenDepthCm: _positiveFinite(arScan?.abdomenDepthCm),
      poshanSetuBmiStatus: poshan.bmiStatus,
      poshanSetuMuacStatus: poshan.muacStatus,
      poshanSetuFinalStatus: poshan.finalStatus,
      poshanSetuTriggeredBy: _encodeTriggers(poshan.triggeredBy),
      poshanSetuComplete: poshan.complete,
      poshanSetuVersion: PoshanSetuService.method,
      storedOverallNutritionPrediction: _normaliseOverallNutrition(overallRaw),
      storedOverallPredictionMethod: _nonEmpty(
        measurement?.classificationMethod ??
            measurement?.combinedMethod ??
            cameraResult?.method,
      ),
      storedOverallPredictionConfidence: _finite(
        measurement?.classificationConfidence ??
            measurement?.combinedConfidenceScore ??
            measurement?.confidenceScore,
      ),
      storedOverallPredictionRationale: _nonEmpty(
        measurement?.classificationRationale ??
            measurement?.whoAcuteRationale ??
            measurement?.combinedRationale,
      ),
      previousMeasurementDate: previousDate == null
          ? null
          : DateFormat('yyyy-MM-dd').format(previousDate),
      daysSincePreviousMeasurement: previousDate == null
          ? null
          : visitDate.difference(previousDate).inDays,
      actualHeightChangeCm: _difference(
        actualHeightCm,
        _positiveFinite(previousMeasurement?.manualHeightCm),
      ),
      actualWeightChangeKg: _difference(
        actualWeightKg,
        _positiveFinite(previousMeasurement?.manualWeightKg),
      ),
      actualMuacChangeCm: _difference(actualMuacCm, previousActualMuac),
      bodyBuild: _nonEmpty(measurement?.bodyBuild),
      estimationMethod: _nonEmpty(measurement?.estimationMethod),
      sideViewUsed: measurement?.sideViewUsed,
      mlEstimatedWeightKg: rawStoredMlWeightKg,
      mlWeightAcceptedForCalculation: storedMlWeightKg != null,
      mlWastingPrediction: _nonEmpty(measurement?.wastingStatus),
      mlWastingMethod: _nonEmpty(measurement?.wastingMethod),
      samProbability: _finite(measurement?.samProbability),
      mamProbability: _finite(measurement?.mamProbability),
      normalProbability: _finite(measurement?.normalProbability),
      riskOverweightProbability: _finite(
        measurement?.riskOverweightProbability,
      ),
      overweightProbability: _finite(measurement?.overweightProbability),
      visitNotes: _nonEmpty(visit.notes),
      measurementNotes: _nonEmpty(measurement?.measuredNotes),
      provenanceNotes: _buildProvenanceNotes(
        measurement: measurement,
        cameraResult: cameraResult,
        arScan: arScan,
        cameraUsesPopulationHeight: cameraUsesPopulationHeight,
        cameraUsesPopulationWeight: cameraUsesPopulationWeight,
        storedPopulationHeightSuppressed:
            storedHeightUsesPopulationReference ||
            fallbackHeightUsesPopulationReference,
        manualHeightDuplicateSuppressed: storedHeightDuplicatesManual,
        rejectedMlWeight:
            rawStoredMlWeightKg != null && storedMlWeightKg == null,
        mlWeightValidatedWithMeasuredHeight:
            rawStoredMlWeightKg != null &&
            calculatedHeightCm == null &&
            actualHeightCm != null,
        mlWeightValidatedWithPopulationReference:
            rawStoredMlWeightKg != null &&
            calculatedHeightCm == null &&
            actualHeightCm == null &&
            weightValidationHeightCm != null,
        mlWeightConfidenceUnavailable:
            calculatedWeightMethod == experimentalMlWeightSourceV1,
        calculatedMuacMethod: calculatedMuac?.muacMethod,
        directMuacHasNoStoredEstimate: hasDirectMuac && calculatedMuac == null,
        oedemaPresent: oedemaPresent,
        measurementModeMissing: measurementMode == null,
      ),
    );
  }

  _WhoScores _computeWhoScores({
    required String sex,
    required double? ageMonths,
    required double? heightCm,
    required double? weightKg,
    required bool suppressWeightScoresForOedema,
  }) {
    final notes = <String>[];
    if (ageMonths == null) {
      notes.add('WHO age unavailable or outside 0-59 months');
    }
    if (heightCm == null) notes.add('height or length unavailable');
    if (weightKg == null) notes.add('weight unavailable');
    if (suppressWeightScoresForOedema) {
      notes.add(
        'weight-related scores not interpreted because oedema is present',
      );
    }
    final bmi = heightCm == null || weightKg == null
        ? null
        : weightKg / ((heightCm / 100) * (heightCm / 100));
    final haz = ageMonths == null || heightCm == null
        ? null
        : _finite(_nutrition.computeHazForAge(sex, ageMonths, heightCm));
    final whz =
        suppressWeightScoresForOedema ||
            ageMonths == null ||
            heightCm == null ||
            weightKg == null
        ? null
        : _finite(_nutrition.computeWhz(sex, ageMonths, heightCm, weightKg));
    final waz =
        suppressWeightScoresForOedema || ageMonths == null || weightKg == null
        ? null
        : _finite(_nutrition.computeWaz(sex, ageMonths, weightKg));
    final baz =
        suppressWeightScoresForOedema || ageMonths == null || bmi == null
        ? null
        : _finite(_nutrition.computeBaz(sex, ageMonths, bmi));

    final hazFlag = _qualityFlag(
      zScore: haz,
      missingInput: ageMonths == null || heightCm == null,
      lower: -6,
      upper: 6,
    );
    final whzFlag = suppressWeightScoresForOedema
        ? 'NOT_INTERPRETABLE_OEDEMA'
        : _qualityFlag(
            zScore: whz,
            missingInput:
                ageMonths == null || heightCm == null || weightKg == null,
            lower: -5,
            upper: 5,
          );
    final wazFlag = suppressWeightScoresForOedema
        ? 'NOT_INTERPRETABLE_OEDEMA'
        : _qualityFlag(
            zScore: waz,
            missingInput: ageMonths == null || weightKg == null,
            lower: -6,
            upper: 5,
          );
    final bazFlag = suppressWeightScoresForOedema
        ? 'NOT_INTERPRETABLE_OEDEMA'
        : _qualityFlag(
            zScore: baz,
            missingInput: ageMonths == null || bmi == null,
            lower: -5,
            upper: 5,
          );
    if (hazFlag == 'IMPLAUSIBLE_REMEASURE') {
      notes.add('HAZ needs remeasurement');
    }
    if (whzFlag == 'IMPLAUSIBLE_REMEASURE') {
      notes.add('WHZ needs remeasurement');
    }
    if (wazFlag == 'IMPLAUSIBLE_REMEASURE') {
      notes.add('WAZ needs remeasurement');
    }
    if (bazFlag == 'IMPLAUSIBLE_REMEASURE') {
      notes.add('BAZ needs remeasurement');
    }

    return _WhoScores(
      bmi: bmi,
      haz: haz,
      hazClassification: hazFlag == 'OK' ? _classifyHaz(haz!) : null,
      hazQualityFlag: hazFlag,
      whz: whz,
      whzClassification: whzFlag == 'OK' ? _classifyWhz(whz!) : null,
      whzQualityFlag: whzFlag,
      waz: waz,
      wazClassification: wazFlag == 'OK' ? _classifyWaz(waz!) : null,
      wazQualityFlag: wazFlag,
      baz: baz,
      bazClassification: bazFlag == 'OK' ? _classifyBaz(baz!) : null,
      bazQualityFlag: bazFlag,
      notes: notes.isEmpty ? null : notes.join('; '),
    );
  }

  _AcuteResult _classifyActualAcute({
    required double? ageMonths,
    required double? whz,
    required double? muacCm,
    required String? oedema,
  }) {
    if (oedema == 'Yes') {
      return const _AcuteResult(
        status: 'SAM',
        triggeredBy: ['oedema'],
        notes: 'Bilateral pitting oedema independently determines SAM.',
      );
    }
    if (ageMonths == null || ageMonths < 6 || ageMonths >= 60) {
      return const _AcuteResult(
        status: 'Indeterminate',
        triggeredBy: [],
        notes: 'The WHZ/MUAC combined rule is limited to age 6-59 months.',
      );
    }
    final triggers = <String>[];
    if (whz != null && whz < -3) triggers.add('whz');
    if (muacCm != null && muacCm < 11.5) triggers.add('muac');
    if (triggers.isNotEmpty) {
      return _AcuteResult(status: 'SAM', triggeredBy: triggers, notes: null);
    }
    if (oedema != 'No') {
      return const _AcuteResult(
        status: 'Indeterminate',
        triggeredBy: [],
        notes: 'Oedema must be checked before MAM or normal can be finalized.',
      );
    }
    if (whz != null && whz >= -3 && whz < -2) triggers.add('whz');
    if (muacCm != null && muacCm >= 11.5 && muacCm < 12.5) {
      triggers.add('muac');
    }
    if (triggers.isNotEmpty) {
      return _AcuteResult(status: 'MAM', triggeredBy: triggers, notes: null);
    }
    if (whz != null && whz >= -2 && muacCm != null && muacCm >= 12.5) {
      return const _AcuteResult(
        status: 'No Acute Malnutrition',
        triggeredBy: ['whz', 'muac', 'oedema'],
        notes: null,
      );
    }
    return const _AcuteResult(
      status: 'Indeterminate',
      triggeredBy: [],
      notes: 'Normal requires WHZ >= -2, MUAC >= 12.5 cm and no oedema.',
    );
  }

  _AcuteResult _classifyCalculatedAcute({
    required double? ageMonths,
    required double? whz,
    required double? muacCm,
  }) {
    if (ageMonths == null || ageMonths < 6 || ageMonths >= 60) {
      return const _AcuteResult(
        status: 'Indeterminate',
        triggeredBy: [],
        notes: 'Calculated acute screening is limited to age 6-59 months.',
      );
    }
    final triggers = <String>[];
    if (whz != null && whz < -3) triggers.add('calculated_whz');
    if (muacCm != null && muacCm < 11.5) triggers.add('calculated_muac');
    if (triggers.isNotEmpty) {
      return _AcuteResult(status: 'SAM', triggeredBy: triggers, notes: null);
    }
    if (whz != null && whz >= -3 && whz < -2) {
      triggers.add('calculated_whz');
    }
    if (muacCm != null && muacCm >= 11.5 && muacCm < 12.5) {
      triggers.add('calculated_muac');
    }
    if (triggers.isNotEmpty) {
      return _AcuteResult(status: 'MAM', triggeredBy: triggers, notes: null);
    }
    if (whz != null && whz >= -2 && muacCm != null && muacCm >= 12.5) {
      return const _AcuteResult(
        status: 'No Acute Malnutrition',
        triggeredBy: ['calculated_whz', 'calculated_muac'],
        notes: null,
      );
    }
    return const _AcuteResult(
      status: 'Indeterminate',
      triggeredBy: [],
      notes: 'Calculated WHZ and calculated MUAC evidence is incomplete.',
    );
  }

  MuacResult? _calculatedMuac({
    required Measurement? measurement,
    required CameraResult? cameraResult,
    required FullArScanResult? arScan,
    required bool hasDirectMuac,
    required double? ageMonths,
  }) {
    final cameraEstimate = _positiveFinite(cameraResult?.estimatedMuacCm);
    if (cameraEstimate != null) {
      return MuacResult(
        muacCm: cameraEstimate,
        muacStatus: null,
        muacMethod:
            _nonEmpty(cameraResult?.muacSource) ??
            _nonEmpty(cameraResult?.method) ??
            'camera_estimated',
        ageInRange: ageMonths != null && ageMonths >= 6 && ageMonths < 60,
        confidence: _finite(arScan?.geometryQualityScore),
        uncertaintyLowerCm: _positiveFinite(cameraResult?.muacRangeLowerCm),
        uncertaintyUpperCm: _positiveFinite(cameraResult?.muacRangeUpperCm),
        modelVersion: _nonEmpty(cameraResult?.modelVersion),
        calibrationVersion: _nonEmpty(cameraResult?.method),
        requiresConfirmation: true,
        referralGuidance: 'Calculated camera MUAC requires tape confirmation.',
      );
    }
    final arEstimate = _positiveFinite(arScan?.estimatedMuacCm);
    if (arEstimate != null) {
      return MuacResult(
        muacCm: arEstimate,
        muacStatus: null,
        muacMethod: arcoreArmMuacSourceV3,
        ageInRange: ageMonths != null && ageMonths >= 6 && ageMonths < 60,
        confidence: _finite(arScan?.geometryQualityScore),
        uncertaintyLowerCm: _positiveFinite(arScan?.muacRangeLowerCm),
        uncertaintyUpperCm: _positiveFinite(arScan?.muacRangeUpperCm),
        modelVersion: arScan?.method,
        calibrationVersion: arScan?.method,
        requiresConfirmation: true,
        referralGuidance: 'ARCore MUAC estimate requires tape confirmation.',
      );
    }
    final storedEstimate = hasDirectMuac
        ? null
        : _positiveFinite(measurement?.muacCm);
    if (storedEstimate != null) {
      return MuacResult(
        muacCm: storedEstimate,
        muacStatus: null,
        muacMethod: _nonEmpty(measurement?.muacMethod) ?? 'estimated',
        ageInRange:
            measurement?.muacAgeInRange ??
            (ageMonths != null && ageMonths >= 6 && ageMonths < 60),
        confidence: measurement?.muacConfidence,
        uncertaintyLowerCm: measurement?.muacUncertaintyLowerCm,
        uncertaintyUpperCm: measurement?.muacUncertaintyUpperCm,
        modelVersion: _nonEmpty(measurement?.muacModelVersion),
        calibrationVersion: _nonEmpty(measurement?.muacCalibrationVersion),
        requiresConfirmation: measurement?.muacRequiresConfirmation ?? true,
        referralGuidance:
            _nonEmpty(measurement?.muacReferralGuidance) ??
            'Calculated MUAC requires confirmation with a tape.',
      );
    }
    return null;
  }

  bool _hasAnyActual(Measurement? measurement) {
    if (measurement == null) return false;
    return _positiveFinite(measurement.manualHeightCm) != null ||
        _positiveFinite(measurement.manualWeightKg) != null ||
        _hasDirectMuac(measurement);
  }

  bool _hasDirectMuac(Measurement? measurement) {
    if (measurement == null || _positiveFinite(measurement.muacCm) == null) {
      return false;
    }
    if (measurement.muacIsDirectMeasurement == true) return true;
    return _isManualMethod(measurement.muacMethod);
  }

  String? _classifyMuac(double? value, {required bool ageInRange}) {
    if (value == null || !ageInRange) return null;
    if (value < 11.5) return 'SAM';
    if (value < 12.5) return 'MAM';
    return 'Normal';
  }

  String _classifyHaz(double zScore) {
    if (zScore < -3) return 'Severely Stunted';
    if (zScore < -2) return 'Stunted';
    return 'Not Stunted';
  }

  String _classifyWhz(double zScore) {
    if (zScore < -3) return 'Severely Wasted';
    if (zScore < -2) return 'Moderately Wasted';
    if (zScore <= 1) return 'Normal';
    if (zScore <= 2) return 'Possible Risk of Overweight';
    if (zScore <= 3) return 'Overweight';
    return 'Obese';
  }

  String _classifyWaz(double zScore) {
    if (zScore < -3) return 'Severely Underweight';
    if (zScore < -2) return 'Underweight';
    return 'Not Underweight';
  }

  String _classifyBaz(double zScore) {
    if (zScore < -3) return 'Severely Low BMI-for-Age';
    if (zScore < -2) return 'Low BMI-for-Age';
    if (zScore <= 1) return 'Normal';
    if (zScore <= 2) return 'Possible Risk of Overweight';
    if (zScore <= 3) return 'Overweight';
    return 'Obese';
  }

  String _qualityFlag({
    required double? zScore,
    required bool missingInput,
    required double lower,
    required double upper,
  }) {
    if (missingInput) return 'UNAVAILABLE_MISSING_INPUT';
    if (zScore == null) return 'UNAVAILABLE_REFERENCE';
    if (zScore < lower || zScore > upper) return 'IMPLAUSIBLE_REMEASURE';
    return 'OK';
  }

  String? _normaliseOverallNutrition(String? value) {
    final canonical = _canonical(value);
    return switch (canonical) {
      'SAM' || 'SEVERELY WASTED' || 'SEVERE WASTING' => 'SAM',
      'MAM' || 'MODERATELY WASTED' || 'MODERATE WASTING' => 'MAM',
      'NORMAL' || 'NO ACUTE MALNUTRITION' => 'No Acute Malnutrition',
      'RISK OVERWEIGHT' || 'RISK OF OVERWEIGHT' => 'Risk of Overweight',
      'OVERWEIGHT' => 'Overweight',
      'OBESE' => 'Obese',
      'INDETERMINATE' || 'UNKNOWN' => 'Indeterminate',
      _ => null,
    };
  }

  String? _normaliseOedema(String? value) {
    final canonical = _canonical(value);
    return switch (canonical) {
      'YES' => 'Yes',
      'NO' => 'No',
      'NOT CHECKED' || 'UNKNOWN' => 'Not checked',
      _ => null,
    };
  }

  String? _normaliseMeasurementMode(String? value) {
    return switch (value?.trim().toLowerCase()) {
      'standing_height' || 'standing' || 'height' => 'standing_height',
      'recumbent_length' || 'recumbent' || 'length' => 'recumbent_length',
      _ => null,
    };
  }

  FullArScanResult? _arScanFromMetadata(String? encoded) {
    if (encoded == null || encoded.isEmpty) return null;
    try {
      final metadata = jsonDecode(encoded);
      if (metadata is! Map<String, dynamic>) return null;
      final raw = metadata['arcore_depth_scan'];
      if (raw is! Map) return null;
      return FullArScanResult.fromJson(Map<String, dynamic>.from(raw));
    } on Object {
      return null;
    }
  }

  double? _positionAdjustment({
    required int? ageDays,
    required String? measurementMode,
  }) {
    if (ageDays == null || measurementMode == null) return null;
    if (ageDays < 731 && measurementMode == 'standing_height') return 0.7;
    if (ageDays >= 731 && measurementMode == 'recumbent_length') return -0.7;
    return 0;
  }

  double _adjustHeight(double heightCm, double? adjustmentCm) {
    return heightCm + (adjustmentCm ?? 0);
  }

  double? _adjustNullableHeight(double? heightCm, double? adjustmentCm) {
    return heightCm == null ? null : _adjustHeight(heightCm, adjustmentCm);
  }

  String? _buildProvenanceNotes({
    required Measurement? measurement,
    required CameraResult? cameraResult,
    required FullArScanResult? arScan,
    required bool cameraUsesPopulationHeight,
    required bool cameraUsesPopulationWeight,
    required bool storedPopulationHeightSuppressed,
    required bool manualHeightDuplicateSuppressed,
    required bool rejectedMlWeight,
    required bool mlWeightValidatedWithMeasuredHeight,
    required bool mlWeightValidatedWithPopulationReference,
    required bool mlWeightConfidenceUnavailable,
    required String? calculatedMuacMethod,
    required bool directMuacHasNoStoredEstimate,
    required bool oedemaPresent,
    required bool measurementModeMissing,
  }) {
    final values = <String>[];
    if (measurement?.muacRequiresConfirmation == true) {
      values.add('muac_requires_tape_confirmation=true');
    }
    if (cameraResult != null) {
      values.add('camera_method=${cameraResult.method}');
      values.add('camera_model_version=${cameraResult.modelVersion}');
      if (cameraResult.nonClinical) values.add('camera_non_clinical=true');
      if (cameraResult.estimatedHaz != null ||
          cameraResult.estimatedWhz != null) {
        values.add('camera_zscores_recomputed_from_same_basis_values=true');
      }
    }
    if (arScan != null) {
      values.add('arcore_depth_scan_available=true');
      values.add('arcore_method=${arScan.method}');
      values.add('arcore_non_clinical=true');
      values.add('arcore_raw_depth_not_retained=true');
    }
    if (cameraUsesPopulationHeight) {
      values.add('camera_population_height_suppressed=true');
    }
    if (cameraUsesPopulationWeight) {
      values.add('camera_population_weight_suppressed=true');
    }
    if (storedPopulationHeightSuppressed) {
      values.add('stored_population_height_suppressed=true');
    }
    if (manualHeightDuplicateSuppressed) {
      values.add('manual_height_duplicate_suppressed=true');
    }
    if (rejectedMlWeight) {
      values.add('implausible_ml_weight_suppressed=true');
    }
    if (mlWeightValidatedWithMeasuredHeight) {
      values.add('ml_weight_bounds_checked_with_measured_height=true');
    }
    if (mlWeightValidatedWithPopulationReference) {
      values.add('ml_weight_bounds_checked_with_population_reference=true');
    }
    if (mlWeightConfidenceUnavailable) {
      values.add('ml_weight_confidence_not_recorded=true');
    }
    if (calculatedMuacMethod == 'estimated_from_whz') {
      values.add('calculated_muac_generated_from_calculated_whz=true');
    }
    if (directMuacHasNoStoredEstimate) {
      values.add('calculated_muac_not_independently_recorded=true');
    }
    if (oedemaPresent) values.add('oedema_grade_not_collected=true');
    if (measurementModeMissing) {
      values.add('measurement_position_assumed_from_age=true');
    }
    return values.isEmpty ? null : values.join('; ');
  }

  String? _storedHeightPredictionMethod(Measurement? measurement) {
    final heightMethod = _nonEmpty(measurement?.heightMethod);
    if (_isManualMethod(heightMethod)) {
      return _nonEmpty(measurement?.estimationMethod) ?? 'stored_prediction';
    }
    return _normaliseHeightEstimateMethod(
      heightMethod ?? _nonEmpty(measurement?.estimationMethod),
    );
  }

  String? _normaliseDirectMethod(String? method) {
    return switch (method?.trim().toLowerCase()) {
      'tape' || 'tape_measured' => 'tape',
      'manual' || 'direct' || 'measured' => 'manual',
      _ => _nonEmpty(method) ?? 'manual',
    };
  }

  String? _normaliseHeightEstimateMethod(String? method) {
    return method?.trim().toLowerCase() == 'who_statistical'
        ? legacyWhoHeightSourceV1
        : _nonEmpty(method);
  }

  String? _normaliseWeightEstimateMethod(String? method) {
    return switch (method?.trim().toLowerCase()) {
      'ml_estimated' => experimentalMlWeightSourceV1,
      'who_statistical' => legacyWhoWeightSourceV1,
      _ => _nonEmpty(method),
    };
  }

  bool _isManualMethod(String? method) {
    final value = method?.trim().toLowerCase();
    return value == 'manual' ||
        value == 'direct' ||
        value == 'measured' ||
        value == 'tape' ||
        value == 'tape_measured';
  }

  bool _isPopulationHeightMethod(String? method) {
    final value = method?.trim().toLowerCase();
    return value == 'who_statistical' ||
        value == 'who_median_estimated' ||
        value == legacyWhoHeightSourceV1;
  }

  DateTime? _parseDate(String raw) => DateTime.tryParse(raw);

  DateTime _dateOnly(DateTime value) {
    return DateTime(value.year, value.month, value.day);
  }

  String _formatStoredDate(String raw) {
    final parsed = _parseDate(raw);
    return parsed == null ? raw : DateFormat('yyyy-MM-dd').format(parsed);
  }

  String? _encodeTriggers(Iterable<String> triggers) {
    final values = triggers.toList(growable: false);
    return values.isEmpty ? null : jsonEncode(values);
  }

  String? _nonEmpty(String? value) {
    final trimmed = value?.trim();
    return trimmed == null || trimmed.isEmpty ? null : trimmed;
  }

  String? _canonical(String? value) {
    final trimmed = _nonEmpty(value);
    if (trimmed == null) return null;
    return trimmed
        .toUpperCase()
        .replaceAll(RegExp(r'[_-]+'), ' ')
        .replaceAll(RegExp(r'\s+'), ' ');
  }

  double? _positiveFinite(double? value) {
    return value != null && value.isFinite && value > 0 ? value : null;
  }

  double? _finite(double? value) {
    return value != null && value.isFinite ? value : null;
  }

  double? _difference(double? calculated, double? actual) {
    return calculated == null || actual == null ? null : calculated - actual;
  }

  bool _isPlausibleMlWeight(double? value, double? whoMedianKg) {
    if (value == null || whoMedianKg == null) return false;
    return value >= whoMedianKg * mlWeightLowerBound &&
        value <= whoMedianKg * mlWeightUpperBound;
  }

  bool _sameMeasurement(double? left, double? right) {
    if (left == null || right == null) return false;
    return (left - right).abs() < 0.000001;
  }
}
