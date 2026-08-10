import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';

import '../../../constants/config.dart';
import '../../../database/daos/camera_result_dao.dart';
import '../../../database/daos/guided_visit_dao.dart';
import '../../../database/database.dart';
import '../../ar_scan/domain/ar_scan_models.dart';
import '../../../models/wasting_features.dart';
import '../../../services/measurement_service.dart';
import '../../../services/ml_inference_service.dart';
import '../../../services/muac_service.dart';
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
    required WhoDataService who,
    required CameraMlInference ml,
    String Function()? newUuid,
    DateTime Function()? now,
  })  : _pose = pose,
        _measurement = measurement,
        _who = who,
        _ml = ml,
        _newUuid = newUuid ?? const Uuid().v4,
        _now = now ?? DateTime.now;

  final PoseSource _pose;
  final MeasurementService _measurement;
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

    final arScan = visit.arScan;
    final usesArGeometry = arScan?.hasWeightGeometry == true;
    final heightCm =
        arScan?.estimatedHeightCm ?? measurements.effectiveHeightCm;
    final photoGeometryScale = arScan != null &&
            measurements.effectiveHeightCm.isFinite &&
            measurements.effectiveHeightCm > 0
        ? heightCm / measurements.effectiveHeightCm
        : 1.0;
    final shoulderWidthCm = usesArGeometry
        ? arScan!.shoulderWidthCm!
        : measurements.shoulderWidthCm * photoGeometryScale;
    final hipWidthCm = usesArGeometry
        ? arScan!.hipWidthCm!
        : measurements.hipWidthCm * photoGeometryScale;
    final torsoLengthCm = usesArGeometry
        ? arScan!.torsoLengthCm!
        : measurements.torsoLengthCm * photoGeometryScale;
    final upperArmLengthCm = usesArGeometry
        ? arScan!.upperArmLengthCm!
        : measurements.upperArmLengthCm * photoGeometryScale;
    final chestDepthCm = usesArGeometry
        ? arScan!.chestDepthCm
        : measurements.chestDepthCm == null
            ? null
            : measurements.chestDepthCm! * photoGeometryScale;
    final abdomenDepthCm = usesArGeometry
        ? arScan!.abdomenDepthCm
        : measurements.abdDepthCm == null
            ? null
            : measurements.abdDepthCm! * photoGeometryScale;
    final features = WastingFeatures(
      ageMonths: visit.ageMonths,
      sexBinary: visit.sex.toUpperCase() == 'M' ? 1 : 0,
      heightCm: heightCm,
      shoulderWidthCm: shoulderWidthCm,
      hipWidthCm: hipWidthCm,
      torsoLengthCm: torsoLengthCm,
      upperArmLengthCm: upperArmLengthCm,
      shoulderHeightRatio: shoulderWidthCm / heightCm,
      hipHeightRatio: hipWidthCm / heightCm,
      bodyBuildScore: _bodyBuildScore(
        shoulderWidthCm: shoulderWidthCm,
        heightCm: heightCm,
        ageMonths: visit.ageMonths,
      ),
      chestDepthCm: chestDepthCm,
      abdDepthCm: abdomenDepthCm,
    );

    WastingPrediction? prediction;
    try {
      prediction = _ml.predict(features);
    } on Object {
      prediction = null;
    }

    final whoMedianWeight = _who.getMedianWeightForHeight(
      visit.sex,
      heightCm,
      ageMonths: visit.ageMonths,
    );
    final (estimatedWeightKg, weightSource) = _resolveEstimatedWeight(
      prediction: prediction,
      whoMedianWeight: whoMedianWeight,
      source: usesArGeometry
          ? arcoreGeometryWeightSourceV3
          : arScan != null
              ? arcoreHeightPhotoGeometryWeightSourceV3
              : experimentalMlWeightSourceV1,
    );
    final weightRange = estimatedWeightKg != null
        ? _weightRange(
            estimateKg: estimatedWeightKg,
            base: features,
            whoMedianWeight: whoMedianWeight,
            geometryQuality: arScan?.geometryQualityScore,
          )
        : null;
    final muac = arScan?.estimatedMuacCm != null
        ? MuacResult(
            muacCm: arScan!.estimatedMuacCm,
            muacStatus: null,
            muacMethod: arcoreArmMuacSourceV3,
            ageInRange: visit.ageMonths >= 6 && visit.ageMonths <= 59.9,
            confidence: arScan.geometryQualityScore,
            uncertaintyLowerCm: arScan.muacRangeLowerCm,
            uncertaintyUpperCm: arScan.muacRangeUpperCm,
            modelVersion: contactlessArMethodV3,
            calibrationVersion: contactlessArMethodV3,
            requiresConfirmation: false,
          )
        : MuacService.estimate(
            ageMonths: visit.ageMonths,
            sex: visit.sex,
            whz: null,
            upperArmLengthCm: upperArmLengthCm,
            shoulderWidthCm: shoulderWidthCm,
            heightCm: heightCm,
            landmarkVisibility: poseConfidence,
            muacMedianCm: _who
                .getReferenceTargets(visit.sex, visit.ageMonths)
                .muacForAge
                ?.target,
          );
    final nutrition = NutritionService(_who);
    final estimatedHaz = nutrition.computeHaz(
      visit.sex,
      visit.ageMonths.round(),
      heightCm,
    );
    final estimatedWhz = estimatedWeightKg == null
        ? null
        : nutrition.computeWhz(
            visit.sex,
            visit.ageMonths,
            heightCm,
            estimatedWeightKg,
          );
    final classifier = _validatedClassifier(prediction);
    final metadata = _ml.metadata;

    return CameraScreeningResult(
      resultUuid: _newUuid(),
      version: version,
      supersedesResultUuid: supersedesResultUuid,
      estimatedHeightCm: heightCm,
      estimatedWeightKg: estimatedWeightKg,
      estimatedMuacCm: muac.muacCm,
      heightSource: arScan == null
          ? legacyWhoHeightSourceV1
          : arScan.method == contactlessArMethodV3
              ? arcoreDepthHeightSourceV3
              : arcoreDepthHeightSourceV2,
      weightSource: weightSource,
      muacSource: muac.muacMethod,
      heightRangeLowerCm: arScan?.heightRangeLowerCm,
      heightRangeUpperCm: arScan?.heightRangeUpperCm,
      weightRangeLowerKg: weightRange?.$1,
      weightRangeUpperKg: weightRange?.$2,
      muacRangeLowerCm: muac.uncertaintyLowerCm,
      muacRangeUpperCm: muac.uncertaintyUpperCm,
      estimatedHaz: estimatedHaz,
      estimatedWhz: estimatedWhz,
      estimatedStuntingStatus:
          estimatedHaz == null ? null : classifyHaz(estimatedHaz),
      estimatedWastingStatus:
          estimatedWhz == null ? null : classifyWhz(estimatedWhz),
      experimentalOverallCategory: classifier?.category,
      componentProbabilities: classifier?.probabilities ?? const {},
      bodyProportionFeatures: {
        'height_cm': heightCm,
        'shoulder_width_cm': shoulderWidthCm,
        'hip_width_cm': hipWidthCm,
        'torso_length_cm': torsoLengthCm,
        'upper_arm_length_cm': upperArmLengthCm,
        'shoulder_height_ratio': shoulderWidthCm / heightCm,
        'hip_height_ratio': hipWidthCm / heightCm,
        'body_build_score': features.bodyBuildScore,
        'side_view_used': usesArGeometry || measurements.sideViewUsed,
        'feature_scaling_height_source': arScan != null
            ? arcoreDepthHeightSourceV3
            : whoReferenceFeatureScalingV1,
        'geometry_source': usesArGeometry
            ? contactlessArMethodV3
            : arScan != null
                ? arcoreHeightPhotoGeometryWeightSourceV3
                : cameraScreeningMethodV1,
        'clinical_measurement_eligible': false,
        if (chestDepthCm != null) 'chest_depth_cm': chestDepthCm,
        if (abdomenDepthCm != null) 'abd_depth_cm': abdomenDepthCm,
      },
      captureQualitySummary: _qualitySummary(acceptedAssets, arScan: arScan),
      method: arScan != null
          ? cameraScreeningContactlessMethodV2
          : cameraScreeningMethodV1,
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
    required String source,
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
      return (predicted, source);
    }
    // A WHO population median is a reference value, not a measurement of this
    // child. Invalid or unavailable ML output therefore fails closed.
    return (null, null);
  }

  (double, double) _weightRange({
    required double estimateKg,
    required WastingFeatures base,
    required double? whoMedianWeight,
    required double? geometryQuality,
  }) {
    final quality = (geometryQuality ?? 0).clamp(0.0, 1.0);
    final fraction = contactlessGeometryPerturbationBase +
        contactlessGeometryPerturbationQualityPenalty * (1.0 - quality);
    final predictions = <double>[estimateKg];
    for (final factor in [1.0 - fraction, 1.0 + fraction]) {
      try {
        final prediction = _ml
            .predict(
              WastingFeatures(
                ageMonths: base.ageMonths,
                sexBinary: base.sexBinary,
                heightCm: base.heightCm * factor,
                shoulderWidthCm: base.shoulderWidthCm * factor,
                hipWidthCm: base.hipWidthCm * factor,
                torsoLengthCm: base.torsoLengthCm * factor,
                upperArmLengthCm: base.upperArmLengthCm * factor,
                shoulderHeightRatio: base.shoulderHeightRatio,
                hipHeightRatio: base.hipHeightRatio,
                bodyBuildScore: base.bodyBuildScore,
                chestDepthCm: base.chestDepthCm == null
                    ? null
                    : base.chestDepthCm! * factor,
                abdDepthCm:
                    base.abdDepthCm == null ? null : base.abdDepthCm! * factor,
              ),
            )
            .estimatedWeightKg;
        if (prediction != null &&
            prediction.isFinite &&
            prediction > 0 &&
            (whoMedianWeight == null ||
                _ml.weightWithinBounds(
                  predictedKg: prediction,
                  whoMedianKg: whoMedianWeight,
                ))) {
          predictions.add(prediction);
        }
      } on Object {
        // The base estimate remains available when a perturbation run fails.
      }
    }
    final observedHalfWidth = predictions
        .map((value) => (value - estimateKg).abs())
        .fold<double>(0, (largest, value) => value > largest ? value : largest);
    final halfWidth =
        observedHalfWidth > contactlessWeightRangeMinimumHalfWidthKg
            ? observedHalfWidth
            : contactlessWeightRangeMinimumHalfWidthKg;
    return (
      (estimateKg - halfWidth).clamp(0.1, double.infinity),
      estimateKg + halfWidth,
    );
  }

  int _bodyBuildScore({
    required double shoulderWidthCm,
    required double heightCm,
    required double ageMonths,
  }) {
    final observed = shoulderWidthCm / heightCm;
    final expected = expectedShoulderRatio(ageMonths);
    if (observed < expected - bodyBuildThresholdMl) return -1;
    if (observed > expected + bodyBuildThresholdMl) return 1;
    return 0;
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
    List<CameraScreeningAsset> assets, {
    FullArScanResult? arScan,
  }) {
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
      if (arScan != null) ...{
        'ar_method': arScan.method,
        'ar_overall': arScan.qualityScore,
        'ar_geometry': arScan.geometryQualityScore,
        'ar_pose': arScan.poseQualityScore,
        'ar_keyframes': arScan.acceptedKeyframes,
        'ar_coverage_degrees': arScan.scanCoverageDegrees,
        'ar_floor_stability_cm': arScan.floorStabilityCm,
        'ar_depth_confidence': arScan.meanDepthConfidence,
      },
    };
  }
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
      processingStarted = visit.captureState == 'processing';
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
          arScan: _arScanFromMetadata(visit.deviceMetadataJson),
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

  FullArScanResult? _arScanFromMetadata(String? encoded) {
    if (encoded == null || encoded.isEmpty) return null;
    try {
      final metadata = jsonDecode(encoded) as Map<String, dynamic>;
      final raw = metadata['arcore_depth_scan'];
      if (raw is! Map) return null;
      return FullArScanResult.fromJson(Map<String, dynamic>.from(raw));
    } on Object {
      return null;
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
      estimatedMuacCm: Value(result.estimatedMuacCm),
      heightSource: Value(result.heightSource),
      weightSource: Value(result.weightSource),
      muacSource: Value(result.muacSource),
      heightRangeLowerCm: Value(result.heightRangeLowerCm),
      heightRangeUpperCm: Value(result.heightRangeUpperCm),
      weightRangeLowerKg: Value(result.weightRangeLowerKg),
      weightRangeUpperKg: Value(result.weightRangeUpperKg),
      muacRangeLowerCm: Value(result.muacRangeLowerCm),
      muacRangeUpperCm: Value(result.muacRangeUpperCm),
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
