import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';

import '../../../constants/config.dart';
import '../../../database/daos/camera_result_dao.dart';
import '../../../database/daos/guided_visit_dao.dart';
import '../../../database/database.dart';
import '../../../models/wasting_features.dart';
import '../../../services/measurement_service.dart';
import '../../../services/ml_inference_service.dart';
import '../../../services/nutrition_service.dart';
import '../../../services/pose_source.dart';
import '../../../services/who_data_service.dart';
import '../domain/camera_screening_result.dart';
import '../domain/capture_models.dart';

abstract interface class CameraMlInference {
  CameraModelMetadata get metadata;

  WastingPrediction predict(WastingFeatures features);

  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  });
}

class MlCameraInferenceAdapter implements CameraMlInference {
  MlCameraInferenceAdapter(this._service);

  final MlInferenceService _service;

  @override
  CameraModelMetadata get metadata => CameraModelMetadata(
        modelVersion: _service.modelVersion,
        manifestChecksum: _service.manifestChecksum,
        trainingDataLabel: _service.trainingDataLabel,
      );

  @override
  WastingPrediction predict(WastingFeatures features) =>
      _service.predict(features);

  @override
  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  }) =>
      _service.weightWithinBounds(
        predictedKg: predictedKg,
        whoMedianKg: whoMedianKg,
      );
}

abstract interface class CameraScreeningRunner {
  Future<CameraScreeningResult> run({
    required CameraScreeningVisit visit,
    required List<CameraScreeningAsset> acceptedAssets,
    required int version,
    String? supersedesResultUuid,
  });
}

class CameraScreeningService implements CameraScreeningRunner {
  CameraScreeningService({
    required PoseSource pose,
    required MeasurementService measurement,
    required NutritionService nutrition,
    required WhoDataService who,
    required CameraMlInference ml,
    String Function()? newUuid,
    DateTime Function()? now,
  })  : _pose = pose,
        _measurement = measurement,
        _nutrition = nutrition,
        _who = who,
        _ml = ml,
        _newUuid = newUuid ?? const Uuid().v4,
        _now = now ?? DateTime.now;

  final PoseSource _pose;
  final MeasurementService _measurement;
  final NutritionService _nutrition;
  final WhoDataService _who;
  final CameraMlInference _ml;
  final String Function() _newUuid;
  final DateTime Function() _now;

  @override
  Future<CameraScreeningResult> run({
    required CameraScreeningVisit visit,
    required List<CameraScreeningAsset> acceptedAssets,
    required int version,
    String? supersedesResultUuid,
  }) async {
    final front = _requiredAsset(acceptedAssets, CaptureAssetRole.front);
    final side = _requiredAsset(acceptedAssets, CaptureAssetRole.side);
    final frontSegments = await _pose.segmentsFor(front.localPath);
    if (frontSegments.totalHeightPx == null ||
        !frontSegments.totalHeightPx!.isFinite ||
        frontSegments.totalHeightPx! <= 0) {
      throw StateError('Accepted front asset has no usable full-body pose');
    }
    final sideSegments = await _pose.sideSegmentsFor(side.localPath);
    final poseConfidence = await _pose.confidenceFor(front.localPath);
    final measurements = _measurement.computeCameraEstimate(
      segments: frontSegments,
      sideSegments: sideSegments,
      ageMonths: visit.ageMonths,
      sex: visit.sex,
      poseConfidence: poseConfidence.isFinite ? poseConfidence : 0,
    );

    final features = WastingFeatures(
      ageMonths: visit.ageMonths,
      sexBinary: visit.sex.toUpperCase() == 'M' ? 1 : 0,
      heightCm: measurements.effectiveHeightCm,
      shoulderWidthCm: measurements.shoulderWidthCm,
      hipWidthCm: measurements.hipWidthCm,
      torsoLengthCm: measurements.torsoLengthCm,
      upperArmLengthCm: measurements.upperArmLengthCm,
      shoulderHeightRatio:
          measurements.shoulderWidthCm / measurements.effectiveHeightCm,
      hipHeightRatio: measurements.hipWidthCm / measurements.effectiveHeightCm,
      bodyBuildScore: measurements.bodyBuildScore,
      chestDepthCm: measurements.chestDepthCm,
      abdDepthCm: measurements.abdDepthCm,
    );

    WastingPrediction? prediction;
    try {
      prediction = _ml.predict(features);
    } on Object {
      prediction = null;
    }

    final whoMedianWeight = _who.getMedianWeightForHeight(
      visit.sex,
      measurements.effectiveHeightCm,
      ageMonths: visit.ageMonths,
    );
    final (estimatedWeightKg, weightSource) = _resolveEstimatedWeight(
      prediction: prediction,
      whoMedianWeight: whoMedianWeight,
      bodyBuild: measurements.bodyBuild,
    );
    final classifier = _validatedClassifier(prediction);

    final estimatedHaz = _finiteOrNull(
      _nutrition.computeHaz(
        visit.sex,
        visit.ageMonths.round(),
        measurements.effectiveHeightCm,
      ),
    );
    final estimatedWhz = estimatedWeightKg == null
        ? null
        : _finiteOrNull(
            _nutrition.computeWhz(
              visit.sex,
              visit.ageMonths,
              measurements.effectiveHeightCm,
              estimatedWeightKg,
            ),
          );
    final metadata = _ml.metadata;

    return CameraScreeningResult(
      resultUuid: _newUuid(),
      version: version,
      supersedesResultUuid: supersedesResultUuid,
      estimatedHeightCm: measurements.effectiveHeightCm,
      estimatedWeightKg: estimatedWeightKg,
      heightSource: measurements.estimationMethod,
      weightSource: weightSource,
      estimatedHaz: estimatedHaz,
      estimatedWhz: estimatedWhz,
      estimatedStuntingStatus:
          estimatedHaz == null ? null : classifyHaz(estimatedHaz),
      estimatedWastingStatus:
          estimatedWhz == null ? null : classifyWhz(estimatedWhz),
      experimentalOverallCategory: classifier?.category,
      componentProbabilities: classifier?.probabilities ?? const {},
      bodyProportionFeatures: {
        'shoulder_width_cm': measurements.shoulderWidthCm,
        'hip_width_cm': measurements.hipWidthCm,
        'torso_length_cm': measurements.torsoLengthCm,
        'upper_arm_length_cm': measurements.upperArmLengthCm,
        'shoulder_height_ratio':
            measurements.shoulderWidthCm / measurements.effectiveHeightCm,
        'hip_height_ratio':
            measurements.hipWidthCm / measurements.effectiveHeightCm,
        'body_build': measurements.bodyBuild,
        'side_view_used': measurements.sideViewUsed,
        if (measurements.chestDepthCm != null)
          'chest_depth_cm': measurements.chestDepthCm,
        if (measurements.abdDepthCm != null)
          'abd_depth_cm': measurements.abdDepthCm,
      },
      captureQualitySummary: _qualitySummary(acceptedAssets),
      method: cameraScreeningMethodV1,
      modelVersion: metadata.modelVersion,
      manifestChecksum: metadata.manifestChecksum,
      trainingDataLabel: metadata.trainingDataLabel,
      createdAt: _now(),
    );
  }

  CameraScreeningAsset _requiredAsset(
    List<CameraScreeningAsset> assets,
    CaptureAssetRole role,
  ) {
    final candidates = assets.where(
      (asset) => asset.role == role && asset.localPath.isNotEmpty,
    );
    if (candidates.isEmpty) {
      throw StateError('Accepted ${role.wireValue} asset is required');
    }
    return candidates.first;
  }

  (double?, String?) _resolveEstimatedWeight({
    required WastingPrediction? prediction,
    required double? whoMedianWeight,
    required String bodyBuild,
  }) {
    final predicted = prediction?.estimatedWeightKg;
    final medianIsValid = whoMedianWeight != null &&
        whoMedianWeight.isFinite &&
        whoMedianWeight > 0;
    if (predicted != null &&
        predicted.isFinite &&
        predicted > 0 &&
        medianIsValid &&
        _ml.weightWithinBounds(
          predictedKg: predicted,
          whoMedianKg: whoMedianWeight,
        )) {
      return (predicted, 'ml_weight_estimator_v1');
    }
    if (!medianIsValid) return (null, null);
    return (
      whoMedianWeight * bodyBuildWeightAdjustment(bodyBuild),
      'who_weight_for_height_median_body_build_v1',
    );
  }

  _ValidatedClassifier? _validatedClassifier(WastingPrediction? prediction) {
    if (prediction == null ||
        !cameraClassifierCategories.contains(prediction.wastingStatus)) {
      return null;
    }
    final probabilities = <String, double>{
      'SAM': prediction.samProbability,
      'MAM': prediction.mamProbability,
      'Normal': prediction.normalProbability,
      'Risk_Overweight': prediction.riskProbability,
      'Overweight': prediction.overweightProbability,
    };
    if (probabilities.values.any(
      (value) => !value.isFinite || value < 0 || value > 1,
    )) {
      return null;
    }
    final sum =
        probabilities.values.fold<double>(0, (total, value) => total + value);
    if ((sum - 1).abs() > 0.02) return null;
    final highest = probabilities.entries.reduce(
      (left, right) => left.value >= right.value ? left : right,
    );
    if (highest.key != prediction.wastingStatus) return null;
    return _ValidatedClassifier(
      category: prediction.wastingStatus,
      probabilities: probabilities,
    );
  }

  Map<String, Object?> _qualitySummary(
    List<CameraScreeningAsset> assets,
  ) {
    final used = assets
        .where(
          (asset) =>
              asset.role == CaptureAssetRole.front ||
              asset.role == CaptureAssetRole.side,
        )
        .toList()
      ..sort((left, right) => left.role.index.compareTo(right.role.index));
    double? average(Iterable<double?> scores) {
      final finite =
          scores.whereType<double>().where((score) => score.isFinite).toList();
      if (finite.isEmpty) return null;
      return finite.reduce((left, right) => left + right) / finite.length;
    }

    return {
      'overall': average(used.map((asset) => asset.overallScore)),
      'pose': average(used.map((asset) => asset.poseScore)),
      'coverage': average(used.map((asset) => asset.coverageScore)),
      'orientation': average(used.map((asset) => asset.orientationScore)),
      'sharpness': average(used.map((asset) => asset.sharpnessScore)),
      'lighting': average(used.map((asset) => asset.lightingScore)),
      'used_views':
          used.map((asset) => asset.role.wireValue).toList(growable: false),
    };
  }

  double? _finiteOrNull(double? value) =>
      value != null && value.isFinite ? value : null;
}

class CameraScreeningWorkflow {
  CameraScreeningWorkflow({
    required AppDatabase database,
    required GuidedVisitDao visitDao,
    required CameraResultDao cameraResultDao,
    required CameraScreeningRunner runner,
  })  : _database = database,
        _visitDao = visitDao,
        _cameraResultDao = cameraResultDao,
        _runner = runner;

  final AppDatabase _database;
  final GuidedVisitDao _visitDao;
  final CameraResultDao _cameraResultDao;
  final CameraScreeningRunner _runner;

  Future<CameraScreeningResult> process({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    var processingStarted = false;
    try {
      final visit = await _visitDao.beginCameraProcessing(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
      );
      processingStarted = true;
      final child = await (_database.select(_database.children)
            ..where(
              (row) =>
                  row.id.equals(visit.childId) &
                  row.ownerUserId.equals(ownerUserId),
            ))
          .getSingleOrNull();
      if (child == null) {
        throw StateError('Owner-scoped child was not found');
      }
      final storedAssets = await (_database.select(_database.captureAssets)
            ..where(
              (row) =>
                  row.visitId.equals(visit.id) &
                  row.qualityVerdict.equals('accepted'),
            )
            ..orderBy([
              (row) => OrderingTerm.asc(row.selectedRank),
              (row) => OrderingTerm.asc(row.capturedAt),
            ]))
          .get();
      final assets = storedAssets
          .where(
            (asset) => asset.localPath != null && asset.localPath!.isNotEmpty,
          )
          .map(
            (asset) => CameraScreeningAsset(
              role: CaptureAssetRole.fromWire(asset.role),
              localPath: asset.localPath!,
              poseScore: asset.poseScore,
              coverageScore: asset.coverageScore,
              orientationScore: asset.orientationScore,
              sharpnessScore: asset.sharpnessScore,
              lightingScore: asset.lightingScore,
              overallScore: asset.overallScore,
            ),
          )
          .toList(growable: false);
      final previous = await _cameraResultDao.getVersions(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
      );
      final result = await _runner.run(
        visit: CameraScreeningVisit(
          visitUuid: visitUuid,
          ownerUserId: ownerUserId,
          ageMonths: visit.ageMonths,
          sex: child.sex,
        ),
        acceptedAssets: assets,
        version: previous.length + 1,
        supersedesResultUuid:
            previous.isEmpty ? null : previous.last.resultUuid,
      );
      await _cameraResultDao.appendCameraResult(
        ownerUserId: ownerUserId,
        visitUuid: visitUuid,
        result: _companionFor(visit.id, result),
        payloadJson: jsonEncode(result.toJson()),
      );
      return result;
    } on Object {
      if (processingStarted) {
        await _visitDao.markCameraProcessingFailed(
          ownerUserId: ownerUserId,
          visitUuid: visitUuid,
        );
      }
      rethrow;
    }
  }

  CameraResultsCompanion _companionFor(
    int visitId,
    CameraScreeningResult result,
  ) {
    return CameraResultsCompanion.insert(
      resultUuid: result.resultUuid,
      visitId: visitId,
      version: result.version,
      supersedesResultUuid: Value(result.supersedesResultUuid),
      estimatedHeightCm: Value(result.estimatedHeightCm),
      estimatedWeightKg: Value(result.estimatedWeightKg),
      heightSource: Value(result.heightSource),
      weightSource: Value(result.weightSource),
      estimatedHaz: Value(result.estimatedHaz),
      estimatedWhz: Value(result.estimatedWhz),
      estimatedStuntingStatus: Value(result.estimatedStuntingStatus),
      estimatedWastingStatus: Value(result.estimatedWastingStatus),
      experimentalOverallCategory: Value(result.experimentalOverallCategory),
      componentProbabilitiesJson:
          Value(jsonEncode(result.componentProbabilities)),
      bodyProportionFeaturesJson:
          Value(jsonEncode(result.bodyProportionFeatures)),
      captureQualitySummaryJson:
          Value(jsonEncode(result.captureQualitySummary)),
      method: result.method,
      modelVersion: result.modelVersion,
      manifestChecksum: result.manifestChecksum,
      trainingDataLabel: result.trainingDataLabel,
      nonClinical: const Value(true),
      createdAt: Value(result.createdAt),
    );
  }
}

class _ValidatedClassifier {
  const _ValidatedClassifier({
    required this.category,
    required this.probabilities,
  });

  final String category;
  final Map<String, double> probabilities;
}
