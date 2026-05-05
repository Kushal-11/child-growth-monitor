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
  AssessmentService({
    required AppDatabase db,
    required ChildDao childDao,
    required VisitDao visitDao,
    required SyncQueueDao syncQueueDao,
    required PoseSource pose,
    required MeasurementService measurement,
    required NutritionService nutrition,
    required WhoDataService who,
    required MlInferenceService ml,
    required ImagePersister persistImage,
  })  : _db = db,
        _childDao = childDao,
        _visitDao = visitDao,
        _syncQueueDao = syncQueueDao,
        _pose = pose,
        _measurement = measurement,
        _nutrition = nutrition,
        _who = who,
        _ml = ml,
        _persistImage = persistImage;

  final AppDatabase _db;
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
  }) async {
    final dob = DateTime.parse(dateOfBirth);
    final ageMonths =
        DateTime.now().difference(dob).inDays / daysPerMonth;

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
      print('AssessmentService: ML prediction failed, falling back to WHO median. $e\n$st');
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

    final haz =
        _nutrition.computeHaz(sex, ageMonths.round(), m.effectiveHeightCm);
    final whz = effectiveWeight != null
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

    final child = await _childDao.findOrCreate(
      name: childName,
      dateOfBirth: dateOfBirth,
      sex: sex,
      guardianName: guardianName,
      location: location,
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
        wastingStatus:
            Value(prediction?.wastingStatus ?? 'who_fallback'),
        muacCm: Value(muacResult.muacCm),
        muacStatus: Value(muacResult.muacStatus),
        muacMethod: Value(muacResult.muacMethod),
      ),
    );
    await _syncQueueDao.enqueue(visitId);

    final summaryStatus = whzStatus ?? hazStatus ?? 'Unknown';

    return ar.AssessmentResult(
      childName: childName,
      sex: sex,
      ageMonths: ageMonths,
      summary: summaryStatus,
      measurement: ar.Measurement(
        predictedHeightCm: m.effectiveHeightCm,
        predictedWeightKg: effectiveWeight,
        manualHeightCm: manualHeightCm,
        manualWeightKg: manualWeightKg,
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
      ),
    );
  }

  // --- Helpers ----------------------------------------------------------

  Future<BodySegments> _detectFront(String path) => _pose.segmentsFor(path);
  Future<SideViewSegments?> _detectSide(String path) => _pose.sideSegmentsFor(path);

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
