import 'dart:convert';

import 'package:drift/drift.dart';

import '../constants/config.dart';
import '../database/daos/child_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';
import '../database/database.dart';
import '../models/assessment_result.dart' as ar;
import '../models/body_measurements.dart';
import '../models/wasting_features.dart';
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

/// Function signature for moving an image into permanent storage.
/// Real impl: `ImageStorageService.persist`. Tests can pass an identity fn.
typedef ImagePersister = Future<String> Function(String tempPath);

class AssessmentService {
  static const protocolVersion = 'WHO-CMAM-OR-2009/2013-v1';
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
        _syncQueueDao = syncQueueDao,
        _pose = pose,
        _measurement = measurement,
        _nutrition = nutrition,
        _who = who,
        _ml = ml,
        _persistImage = persistImage;

  final ChildDao _childDao;
  final VisitDao _visitDao;
  final SyncQueueDao _syncQueueDao;
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
    final dob = DateTime.parse(dateOfBirth);
    final ageMonths = DateTime.now().difference(dob).inDays / daysPerMonth;

    final frontPath = await _persistImage(frontImagePath);
    final sidePath =
        sideImagePath != null ? await _persistImage(sideImagePath) : null;
    final backPath =
        backImagePath != null ? await _persistImage(backImagePath) : null;

    final segments = await _detectFront(frontPath);
    if (segments.totalHeightPx == null || segments.totalHeightPx! <= 0) {
      throw PoseDetectionFailedException(
        'Could not detect a complete body in the photo. '
        'Make sure the child is fully visible (head to heels) and try again.',
      );
    }
    final sideSegments = sidePath != null ? await _detectSide(sidePath) : null;
    final poseConfidence = await _pose.confidenceFor(frontPath);

    final m = _measurement.compute(
      segments: segments,
      sideSegments: sideSegments,
      ageMonths: ageMonths,
      sex: sex,
      manualHeightCm: manualHeightCm,
      poseConfidence: poseConfidence,
    );
    final heightMethod = manualHeightCm != null ? 'manual' : m.estimationMethod;

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
        'AssessmentService: ML prediction failed, falling back to WHO median. $e\n$st',
      );
      prediction = null;
    }

    final whoMedianWeight = _who.getMedianWeightForHeight(
      sex,
      m.effectiveHeightCm,
      ageMonths: ageMonths,
    );
    final effectiveWeight = _resolveWeight(
      manualWeightKg: manualWeightKg,
      ml: prediction,
      whoMedianKg: whoMedianWeight,
      build: m.bodyBuild,
    );

    final haz = _nutrition.computeHaz(
      sex,
      ageMonths.round(),
      m.effectiveHeightCm,
    );
    final whz = effectiveWeight != null
        ? _nutrition.computeWhz(
            sex,
            ageMonths,
            m.effectiveHeightCm,
            effectiveWeight,
          )
        : null;

    final muacResult = MuacService.estimate(
      ageMonths: ageMonths,
      sex: sex,
      whz: whz,
      manualMuacCm: manualMuacCm,
    );

    final hazStatus = haz != null ? classifyHaz(haz) : null;
    final whzStatus = whz != null ? classifyWhz(whz) : null;
    final weightMethod = manualWeightKg != null
        ? 'manual'
        : prediction?.estimatedWeightKg == effectiveWeight
            ? 'ml_estimated'
            : effectiveWeight != null
                ? 'who_statistical'
                : 'unavailable';
    final poshan = const PoshanSetuService().classify(
      sex: sex,
      ageMonths: ageMonths,
      heightCm: m.effectiveHeightCm,
      heightSource: manualHeightCm != null ? 'manual' : 'unavailable',
      weightKg: effectiveWeight,
      weightSource: weightMethod,
      muacCm: muacResult.muacCm,
      muacSource: muacResult.muacMethod,
    );

    // Only WHZ and independently tape-measured MUAC define the headline
    // clinical verdict. ML and WHZ-derived MUAC remain decision support.
    final summaryStatus = combineNutritionStatus(
      whzStatus: whzStatus,
      muacStatus: muacResult.muacStatus,
      muacMethod: muacResult.muacMethod,
      isDirectMeasurement: muacResult.isDirectMeasurement,
    );
    final triggeredBy = <String>[
      if (muacResult.isDirectMeasurement &&
          muacResult.muacStatus == summaryStatus &&
          (summaryStatus == 'SAM' || summaryStatus == 'MAM'))
        'muac',
      if (whzStatus == summaryStatus && summaryStatus != 'NORMAL') 'whz',
    ];
    final rationale = triggeredBy.isEmpty
        ? 'No direct MUAC or WHZ flag triggered'
        : '$summaryStatus flagged by ${triggeredBy.join(' or ')} (WHO OR-rule)';
    final bmi = poshan.bmi;

    final child = await _childDao.findOrCreate(
      name: childName,
      dateOfBirth: dateOfBirth,
      sex: sex,
      guardianName: guardianName,
      location: location,
      ownerUserId: ownerUserId,
    );

    final visitId = await _visitDao.createWithMeasurement(
      childId: child.id,
      ageMonths: ageMonths,
      imagePath: frontPath,
      sideImagePath: sidePath,
      backImagePath: backPath,
      measurement: MeasurementsCompanion(
        predictedHeightCm: Value(m.effectiveHeightCm),
        predictedWeightKg: Value(effectiveWeight),
        manualHeightCm: Value(manualHeightCm),
        manualWeightKg: Value(manualWeightKg),
        effectiveHeightCm: Value(m.effectiveHeightCm),
        effectiveWeightKg: Value(effectiveWeight),
        heightMethod: Value(heightMethod),
        weightMethod: Value(weightMethod),
        bmi: Value(bmi),
        bmiStatus: Value(poshan.bmiStatus),
        hazZscore: Value(haz),
        whzZscore: Value(whz),
        hazStatus: Value(hazStatus),
        whzStatus: Value(whzStatus),
        confidenceScore: Value(poseConfidence),
        heightConfidence: Value(manualHeightCm != null ? 1.0 : null),
        weightConfidence: Value(manualWeightKg != null ? 1.0 : null),
        classificationConfidence: Value(
          triggeredBy.contains('muac') ? 1.0 : poseConfidence,
        ),
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
        wastingStatus: Value(prediction?.wastingStatus ?? 'who_fallback'),
        wastingMethod:
            Value(prediction == null ? 'unavailable' : 'ml_classifier'),
        muacCm: Value(muacResult.muacCm),
        muacStatus: Value(muacResult.muacStatus),
        muacMethod: Value(muacResult.muacMethod),
        muacAgeInRange: Value(muacResult.ageInRange),
        muacConfidence: Value(muacResult.confidence),
        muacUncertaintyLowerCm: Value(muacResult.uncertaintyLowerCm),
        muacUncertaintyUpperCm: Value(muacResult.uncertaintyUpperCm),
        muacModelVersion: Value(muacResult.modelVersion),
        muacCalibrationVersion: Value(muacResult.calibrationVersion),
        muacIsDirectMeasurement: Value(muacResult.isDirectMeasurement),
        muacRequiresConfirmation: Value(muacResult.requiresConfirmation),
        muacReferralGuidance: Value(muacResult.referralGuidance),
        combinedStatus: Value(summaryStatus),
        combinedTriggeredBy: Value(jsonEncode(triggeredBy)),
        combinedRationale: Value(rationale),
        combinedMethod: const Value('who_muac_whz_or_rule'),
        combinedConfidenceScore: Value(
          triggeredBy.contains('muac') ? 1.0 : poseConfidence,
        ),
        combinedProtocolVersion: const Value(protocolVersion),
        poshanStatus: Value(poshan.finalStatus),
        poshanTriggeredBy: Value(jsonEncode(poshan.triggeredBy)),
        classificationMethod: Value(poshan.classificationMethod),
        classificationRationale: Value(poshan.rationale),
        poshanComplete: Value(poshan.complete),
      ),
    );
    await _syncQueueDao.enqueue(visitId);

    return ar.AssessmentResult(
      childName: childName,
      sex: sex,
      ageMonths: ageMonths,
      summary: poshan.finalStatus,
      combinedNutrition: ar.CombinedNutritionDetail(
        status: summaryStatus,
        triggeredBy: triggeredBy,
        rationale: triggeredBy.isEmpty
            ? 'No direct MUAC or WHZ flag triggered'
            : '$summaryStatus flagged by ${triggeredBy.join(' or ')} (WHO OR-rule)',
        method: 'who_muac_whz_or_rule',
        confidenceScore: poseConfidence,
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
      measurement: ar.Measurement(
        effectiveHeightCm: m.effectiveHeightCm,
        heightMethod: heightMethod,
        predictedHeightCm: m.effectiveHeightCm,
        predictedWeightKg: effectiveWeight,
        manualHeightCm: manualHeightCm,
        manualWeightKg: manualWeightKg,
        effectiveWeightKg: effectiveWeight,
        weightMethod: weightMethod,
        heightConfidence: manualHeightCm != null ? 1.0 : null,
        weightConfidence: manualWeightKg != null ? 1.0 : null,
        confidenceScore: poseConfidence,
        estimationMethod: m.estimationMethod,
        bodyBuild: m.bodyBuild,
        sideViewUsed: m.sideViewUsed,
        chestDepthCm: m.chestDepthCm,
        abdDepthCm: m.abdDepthCm,
      ),
      nutrition: ar.Nutrition(
        hazZscore: haz,
        whzZscore: whz,
        hazStatus: hazStatus,
        whzStatus: whzStatus,
        ageMonths: ageMonths,
        bmi: poshan.bmi,
        bmiStatus: poshan.bmiStatus,
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
            ),
      muac: ar.MuacDetail(
        muacCm: muacResult.muacCm,
        muacStatus: muacResult.muacStatus,
        muacMethod: muacResult.muacMethod,
        ageInRange: muacResult.ageInRange,
        confidence: muacResult.confidence,
        uncertaintyLowerCm: muacResult.uncertaintyLowerCm,
        uncertaintyUpperCm: muacResult.uncertaintyUpperCm,
        modelVersion: muacResult.modelVersion,
        calibrationVersion: muacResult.calibrationVersion,
        isDirectMeasurement: muacResult.isDirectMeasurement,
        requiresConfirmation: muacResult.requiresConfirmation,
        referralGuidance: muacResult.referralGuidance,
      ),
    );
  }

  // --- Helpers ----------------------------------------------------------

  Future<BodySegments> _detectFront(String path) => _pose.segmentsFor(path);
  Future<SideViewSegments?> _detectSide(String path) =>
      _pose.sideSegmentsFor(path);

  double? _resolveWeight({
    required double? manualWeightKg,
    required WastingPrediction? ml,
    required double? whoMedianKg,
    required String build,
  }) {
    if (manualWeightKg != null && manualWeightKg > 0) return manualWeightKg;
    if (ml?.estimatedWeightKg != null && whoMedianKg != null) {
      final ok = _ml.weightWithinBounds(
        predictedKg: ml!.estimatedWeightKg!,
        whoMedianKg: whoMedianKg,
      );
      if (ok) return ml.estimatedWeightKg;
    }
    if (whoMedianKg == null) return null;
    return whoMedianKg * bodyBuildWeightAdjustment(build);
  }
}
