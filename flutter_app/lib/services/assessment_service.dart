import 'dart:convert';
import 'dart:io';

import 'package:drift/drift.dart';

import '../constants/config.dart';
import '../database/daos/child_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';
import '../database/database.dart';
import '../models/assessment_result.dart' as ar;
import '../models/body_measurements.dart';
import '../models/wasting_features.dart';
import 'age_service.dart';
import 'measurement_service.dart';
import 'ml_inference_service.dart';
import 'muac_service.dart';
import 'nutrition_service.dart';
import 'pose_source.dart';
import 'poshan_setu_service.dart';
import 'who_data_service.dart';

/// Thrown when on-device pose detection fails to produce a usable result.
/// Surfaces to the UI so the worker can retake the photo.
class PoseDetectionFailedException implements Exception {
  PoseDetectionFailedException(this.message);
  final String message;
  @override
  String toString() => 'PoseDetectionFailedException: $message';
}

class InvalidAssessmentInputException implements Exception {
  InvalidAssessmentInputException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Function signature for moving an image into permanent storage.
/// Real impl: `ImageStorageService.persist`. Tests can pass an identity fn.
typedef ImagePersister = Future<String> Function(String tempPath);

class AssessmentService {
  AssessmentService({
    required ChildDao childDao,
    required VisitDao visitDao,
    required SyncQueueDao syncQueueDao,
    required PoseSource pose,
    required MeasurementService measurement,
    required NutritionService nutrition,
    required WhoDataService who,
    required MlInferenceService ml,
    required ImagePersister persistImage,
  })  : _childDao = childDao,
        _visitDao = visitDao,
        _pose = pose,
        _measurement = measurement,
        _nutrition = nutrition,
        _who = who,
        _ml = ml,
        _persistImage = persistImage;

  final ChildDao _childDao;
  final VisitDao _visitDao;
  final PoseSource _pose;
  final MeasurementService _measurement;
  final NutritionService _nutrition;
  final WhoDataService _who;
  final MlInferenceService _ml;
  final ImagePersister _persistImage;

  Future<ar.AssessmentResult> runAssessment({
    required String frontImagePath,
    String? sideImagePath,
    String? backImagePath,
    required String childName,
    required String dateOfBirth,
    required String sex,
    double? manualWeightKg,
    double? manualHeightCm,
    double? manualMuacCm,
    String? guardianName,
    String? location,
    int? ownerUserId,
  }) async {
    final dob = DateTime.tryParse(dateOfBirth);
    if (dob == null) {
      throw InvalidAssessmentInputException('Select a valid date of birth.');
    }
    final assessedOn = DateTime.now();
    final ageMonths = AgeService.ageMonthsAt(dob, assessedOn);
    final completedAgeMonths = AgeService.completedMonths(dob, assessedOn);
    _validateInputs(
      ageMonths: ageMonths,
      sex: sex,
      manualHeightCm: manualHeightCm,
      manualWeightKg: manualWeightKg,
      manualMuacCm: manualMuacCm,
    );
    if (ownerUserId == null) {
      throw InvalidAssessmentInputException(
        'Sign in before saving an assessment.',
      );
    }

    // Validate pose quality on temporary capture files before copying anything
    // into permanent app storage. Failed/low-confidence captures therefore do
    // not leave orphaned image files.
    final segments = await _detectFront(frontImagePath);
    if (segments.totalHeightPx == null || segments.totalHeightPx! <= 0) {
      throw PoseDetectionFailedException(
        'Could not detect a complete body in the photo. '
        'Make sure the child is fully visible (head to heels) and try again.',
      );
    }
    final sideSegments =
        sideImagePath != null ? await _detectSide(sideImagePath) : null;
    final poseConfidence = await _pose.confidenceFor(frontImagePath);
    if (!poseConfidence.isFinite || poseConfidence < minConfidenceThreshold) {
      throw PoseDetectionFailedException(
        'Pose confidence is too low for a reliable assessment. '
        'Make sure the full body is clear and retake the photo.',
      );
    }

    final m = _measurement.compute(
      segments: segments,
      sideSegments: sideSegments,
      ageMonths: ageMonths,
      sex: sex,
      manualHeightCm: manualHeightCm,
      poseConfidence: poseConfidence,
    );

    WastingPrediction? prediction;
    try {
      final features = WastingFeatures(
        ageMonths: ageMonths,
        sexBinary: sex.toUpperCase() == 'M' ? 1 : 0,
        heightCm: m.effectiveHeightCm,
        shoulderWidthCm: m.shoulderWidthCm,
        hipWidthCm: m.hipWidthCm,
        torsoLengthCm: m.torsoLengthCm,
        upperArmLengthCm: m.upperArmLengthCm,
        shoulderHeightRatio: m.shoulderWidthCm / m.effectiveHeightCm,
        hipHeightRatio: m.hipWidthCm / m.effectiveHeightCm,
        bodyBuildScore: m.bodyBuildScore,
        chestDepthCm: m.chestDepthCm,
        abdDepthCm: m.abdDepthCm,
      );
      prediction = _ml.predict(features);
    } catch (e, st) {
      // ML failure → WHO median fallback. Log diagnostics so production
      // bugs (corrupt model, bad feature vector) don't masquerade as
      // routine fallback usage.
      // ignore: avoid_print
      print(
          'AssessmentService: ML prediction failed, falling back to WHO median. $e\n$st');
      prediction = null;
    }

    final whoMedianWeight = _who.getMedianWeightForHeight(
      sex,
      m.effectiveHeightCm,
      ageMonths: ageMonths,
    );
    final resolvedWeight = _resolveWeight(
      manualWeightKg: manualWeightKg,
      ml: prediction,
      whoMedianKg: whoMedianWeight,
      build: m.bodyBuild,
    );
    final effectiveWeight = resolvedWeight.value;
    final heightSource = PoshanSetuService.normalizeSource(m.estimationMethod);
    final weightSource =
        PoshanSetuService.normalizeSource(resolvedWeight.source);

    final measuredHeight =
        PoshanSetuService.isEligibleBodyMeasurementSource(heightSource);
    final measuredWeight =
        PoshanSetuService.isEligibleBodyMeasurementSource(weightSource);
    // WHO z-scores are clinical outputs only when their anthropometric inputs
    // were actually measured. Statistical/ML estimates stay visible as
    // secondary screening data and cannot manufacture a Normal result.
    final haz = measuredHeight
        ? _nutrition.computeHaz(sex, completedAgeMonths, m.effectiveHeightCm)
        : null;
    final whz = measuredHeight && measuredWeight && effectiveWeight != null
        ? _nutrition.computeWhz(
            sex, ageMonths, m.effectiveHeightCm, effectiveWeight)
        : null;

    final muacResult = MuacService.estimate(
      ageMonths: ageMonths,
      sex: sex,
      whz: whz,
      manualMuacCm: manualMuacCm,
    );

    final hazStatus = haz != null ? classifyHaz(haz) : null;
    final whzStatus = whz != null ? classifyWhz(whz) : null;
    final poshan = const PoshanSetuService().classify(
      sex: sex,
      ageMonths: ageMonths,
      heightCm: m.effectiveHeightCm,
      heightSource: heightSource,
      weightKg: effectiveWeight,
      weightSource: weightSource,
      muacCm: muacResult.muacCm,
      muacSource: muacResult.muacMethod,
    );

    String? frontPath;
    String? sidePath;
    String? backPath;
    late final ChildrenData child;
    try {
      frontPath = await _persistImage(frontImagePath);
      sidePath =
          sideImagePath != null ? await _persistImage(sideImagePath) : null;
      backPath =
          backImagePath != null ? await _persistImage(backImagePath) : null;

      child = await _childDao.findOrCreate(
        name: childName,
        dateOfBirth: dateOfBirth,
        sex: sex,
        guardianName: guardianName,
        location: location,
        ownerUserId: ownerUserId,
      );

      await _visitDao.createWithMeasurement(
        childId: child.id,
        ageMonths: ageMonths,
        imagePath: frontPath,
        sideImagePath: sidePath,
        backImagePath: backPath,
        ownerUserId: ownerUserId,
        enqueueForSync: true,
        measurement: MeasurementsCompanion(
          predictedHeightCm:
              Value(manualHeightCm == null ? m.effectiveHeightCm : null),
          predictedWeightKg:
              Value(manualWeightKg == null ? effectiveWeight : null),
          manualHeightCm: Value(manualHeightCm),
          manualWeightKg: Value(manualWeightKg),
          hazZscore: Value(haz),
          whzZscore: Value(whz),
          hazStatus: Value(hazStatus),
          whzStatus: Value(whzStatus),
          confidenceScore: Value(poseConfidence),
          bodyBuild: Value(m.bodyBuild),
          estimationMethod: Value(m.estimationMethod),
          sideViewUsed: Value(m.sideViewUsed),
          chestDepthCm: Value(m.chestDepthCm),
          abdDepthCm: Value(m.abdDepthCm),
          mlEstimatedWeightKg: Value(prediction?.estimatedWeightKg),
          samProbability: Value(prediction?.samProbability),
          mamProbability: Value(prediction?.mamProbability),
          normalProbability: Value(prediction?.normalProbability),
          riskOverweightProbability: Value(prediction?.riskProbability),
          overweightProbability: Value(prediction?.overweightProbability),
          wastingStatus: Value(prediction?.wastingStatus),
          muacCm: Value(muacResult.muacCm),
          muacStatus: Value(poshan.muacStatus),
          muacMethod: Value(muacResult.muacMethod),
          effectiveHeightCm: Value(m.effectiveHeightCm),
          effectiveWeightKg: Value(effectiveWeight),
          heightSource: Value(heightSource),
          weightSource: Value(weightSource),
          bmi: Value(poshan.bmi),
          bmiStatus: Value(poshan.bmiStatus),
          poshanStatus: Value(poshan.finalStatus),
          poshanTriggeredBy: Value(jsonEncode(poshan.triggeredBy)),
          classificationMethod: Value(poshan.classificationMethod),
          classificationRationale: Value(poshan.rationale),
          mlModelVersion: Value(prediction?.modelVersion),
          mlNonClinical: Value(prediction?.nonClinical),
          mlTrainingData: Value(prediction?.trainingData),
        ),
      );
    } catch (_) {
      await _deletePersistedCopy(frontPath, frontImagePath);
      await _deletePersistedCopy(sidePath, sideImagePath);
      await _deletePersistedCopy(backPath, backImagePath);
      rethrow;
    }

    return ar.AssessmentResult(
      childName: childName,
      sex: sex,
      ageMonths: ageMonths,
      summary: poshan.finalStatus,
      measurement: ar.Measurement(
        predictedHeightCm: manualHeightCm == null ? m.effectiveHeightCm : null,
        predictedWeightKg: manualWeightKg == null ? effectiveWeight : null,
        manualHeightCm: manualHeightCm,
        manualWeightKg: manualWeightKg,
        confidenceScore: poseConfidence,
        estimationMethod: m.estimationMethod,
        bodyBuild: m.bodyBuild,
        sideViewUsed: m.sideViewUsed,
        chestDepthCm: m.chestDepthCm,
        abdDepthCm: m.abdDepthCm,
        effectiveHeightCm: m.effectiveHeightCm,
        effectiveWeightKg: effectiveWeight,
        heightSource: heightSource,
        weightSource: weightSource,
      ),
      nutrition: ar.Nutrition(
        hazZscore: haz,
        whzZscore: whz,
        hazStatus: hazStatus,
        whzStatus: whzStatus,
        ageMonths: ageMonths,
      ),
      mlPrediction: prediction == null
          ? null
          : ar.MlPrediction(
              estimatedWeightKg: prediction.estimatedWeightKg,
              samProbability: prediction.samProbability,
              mamProbability: prediction.mamProbability,
              normalProbability: prediction.normalProbability,
              riskProbability: prediction.riskProbability,
              overweightProbability: prediction.overweightProbability,
              wastingStatus: prediction.wastingStatus,
              modelVersion: prediction.modelVersion,
              nonClinical: prediction.nonClinical,
              trainingData: prediction.trainingData,
            ),
      muac: ar.MuacDetail(
        muacCm: muacResult.muacCm,
        muacStatus: poshan.muacStatus,
        muacMethod: muacResult.muacMethod,
        ageInRange: muacResult.ageInRange,
      ),
      poshan: ar.PoshanDetail(
        bmi: poshan.bmi,
        bmiStatus: poshan.bmiStatus,
        muacStatus: poshan.muacStatus,
        finalStatus: poshan.finalStatus,
        triggeredBy: poshan.triggeredBy,
        classificationMethod: poshan.classificationMethod,
        rationale: poshan.rationale,
        complete: poshan.complete,
      ),
    );
  }

  // --- Helpers ----------------------------------------------------------

  Future<BodySegments> _detectFront(String path) => _pose.segmentsFor(path);
  Future<SideViewSegments?> _detectSide(String path) =>
      _pose.sideSegmentsFor(path);

  Future<void> _deletePersistedCopy(
    String? persistedPath,
    String? originalPath,
  ) async {
    if (persistedPath == null ||
        originalPath == null ||
        persistedPath == originalPath) {
      return;
    }
    try {
      final file = File(persistedPath);
      if (await file.exists()) await file.delete();
    } on FileSystemException {
      // Best-effort cleanup only. The assessment failure remains the primary
      // error surfaced to the worker.
    }
  }

  ({double? value, String source}) _resolveWeight({
    required double? manualWeightKg,
    required WastingPrediction? ml,
    required double? whoMedianKg,
    required String build,
  }) {
    if (manualWeightKg != null && manualWeightKg > 0) {
      return (value: manualWeightKg, source: 'manual');
    }
    if (ml?.estimatedWeightKg != null && whoMedianKg != null) {
      final ok = _ml.weightWithinBounds(
        predictedKg: ml!.estimatedWeightKg!,
        whoMedianKg: whoMedianKg,
      );
      if (ok) {
        return (value: ml.estimatedWeightKg, source: 'ml_estimated');
      }
    }
    if (whoMedianKg == null) {
      return (value: null, source: 'unavailable');
    }
    return (
      value: whoMedianKg * bodyBuildWeightAdjustment(build),
      source: 'who_statistical',
    );
  }

  void _validateInputs({
    required double ageMonths,
    required String sex,
    required double? manualHeightCm,
    required double? manualWeightKg,
    required double? manualMuacCm,
  }) {
    if (!ageMonths.isFinite ||
        ageMonths < 0 ||
        ageMonths >= maxUnderFiveAgeMonths) {
      throw InvalidAssessmentInputException(
          'Assessments are limited to children under five years old.');
    }
    final normalizedSex = sex.trim().toUpperCase();
    if (normalizedSex != 'M' && normalizedSex != 'F') {
      throw InvalidAssessmentInputException('Select the child\'s sex.');
    }
    _validateOptionalRange(
      manualHeightCm,
      label: 'Height',
      min: minPlausibleHeightCm,
      max: maxPlausibleHeightCm,
      unit: 'cm',
    );
    _validateOptionalRange(
      manualWeightKg,
      label: 'Weight',
      min: minPlausibleWeightKg,
      max: maxPlausibleWeightKg,
      unit: 'kg',
    );
    _validateOptionalRange(
      manualMuacCm,
      label: 'MUAC',
      min: minPlausibleMuacCm,
      max: maxPlausibleMuacCm,
      unit: 'cm',
    );
  }

  void _validateOptionalRange(
    double? value, {
    required String label,
    required double min,
    required double max,
    required String unit,
  }) {
    if (value == null) return;
    if (!value.isFinite || value < min || value > max) {
      throw InvalidAssessmentInputException(
          '$label must be between $min and $max $unit.');
    }
  }
}
