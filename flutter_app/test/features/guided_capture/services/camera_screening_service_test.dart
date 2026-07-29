import 'package:child_growth_monitor_app/database/daos/camera_result_dao.dart';
import 'package:child_growth_monitor_app/database/daos/capture_asset_dao.dart';
import 'package:child_growth_monitor_app/database/daos/guided_visit_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/camera_screening_service.dart';
import 'package:child_growth_monitor_app/models/body_measurements.dart';
import 'package:child_growth_monitor_app/models/wasting_features.dart';
import 'package:child_growth_monitor_app/services/measurement_service.dart';
import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/pose_source.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';
import 'package:drift/drift.dart' show Value;
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

class FakeWhoDataService extends WhoDataService {
  FakeWhoDataService({
    this.medianHeightCm = 90,
    this.medianWeightKg = 12,
  });

  double? medianHeightCm;
  double? medianWeightKg;

  @override
  double? getMedianHeightForAge(String sex, int ageMonths) => medianHeightCm;

  @override
  (double, double, double)? getHazLms(String sex, int ageMonths) {
    final median = medianHeightCm;
    return median == null ? null : (1, median, 0.1);
  }

  @override
  double? getMedianWeightForHeight(
    String sex,
    double heightCm, {
    double ageMonths = 36,
  }) =>
      medianWeightKg;

  @override
  (double, double, double)? getWfhLms(
    String sex,
    double heightCm,
    double ageMonths,
  ) {
    final median = medianWeightKg;
    return median == null ? null : (1, median, 0.1);
  }
}

class FakePoseSource implements PoseSource {
  @override
  Future<double> confidenceFor(String path) async => 0.9;

  @override
  Future<BodySegments> segmentsFor(String path) async => const BodySegments(
        totalHeightPx: 800,
        shoulderWidthPx: 160,
        hipWidthPx: 140,
        torsoLengthPx: 240,
        upperArmLengthPx: 120,
      );

  @override
  Future<SideViewSegments?> sideSegmentsFor(String path) async =>
      const SideViewSegments(
        totalHeightPx: 800,
        chestDepthPx: 60,
        abdDepthPx: 70,
      );
}

class FakeCameraMlInference implements CameraMlInference {
  FakeCameraMlInference({
    this.prediction = const WastingPrediction(
      estimatedWeightKg: 11,
      samProbability: 0.1,
      mamProbability: 0.6,
      normalProbability: 0.2,
      riskProbability: 0.05,
      overweightProbability: 0.05,
      wastingStatus: 'MAM',
    ),
    this.error,
  });

  WastingPrediction prediction;
  Object? error;

  @override
  CameraModelMetadata get metadata => const CameraModelMetadata(
        modelVersion: 'synthetic-who-v1',
        manifestChecksum:
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
        trainingDataLabel: 'synthetic_who_research_only',
      );

  @override
  WastingPrediction predict(WastingFeatures features) {
    if (error case final error?) throw error;
    return prediction;
  }

  @override
  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  }) {
    final ratio = predictedKg / whoMedianKg;
    return ratio >= 0.45 && ratio <= 1.8;
  }
}

CameraScreeningVisit screeningVisit() => const CameraScreeningVisit(
      visitUuid: '10000000-0000-0000-0000-000000000001',
      ownerUserId: 7,
      ageMonths: 30,
      sex: 'F',
    );

List<CameraScreeningAsset> acceptedAssets() => const [
      CameraScreeningAsset(
        role: CaptureAssetRole.front,
        localPath: '/visit/front.jpg',
        overallScore: 0.9,
        poseScore: 0.9,
      ),
      CameraScreeningAsset(
        role: CaptureAssetRole.side,
        localPath: '/visit/side.jpg',
        overallScore: 0.8,
        poseScore: 0.85,
      ),
    ];

CameraScreeningService service({
  FakeWhoDataService? who,
  FakeCameraMlInference? ml,
}) {
  final whoService = who ?? FakeWhoDataService();
  return CameraScreeningService(
    pose: FakePoseSource(),
    measurement: MeasurementService(whoService),
    nutrition: NutritionService(whoService),
    who: whoService,
    ml: ml ?? FakeCameraMlInference(),
    newUuid: () => '30000000-0000-0000-0000-000000000001',
    now: () => DateTime.utc(2026, 7, 29, 10),
  );
}

void main() {
  group('CameraScreeningService', () {
    test('requires accepted front and side assets', () async {
      await expectLater(
        service().run(
          visit: screeningVisit(),
          acceptedAssets: [acceptedAssets().first],
          version: 1,
        ),
        throwsStateError,
      );
    });

    test('returns isolated non-clinical estimates with full provenance',
        () async {
      final result = await service().run(
        visit: screeningVisit(),
        acceptedAssets: acceptedAssets(),
        version: 1,
      );

      expect(result.method, cameraScreeningMethodV1);
      expect(result.nonClinical, isTrue);
      expect(result.heightSource, 'who_height_for_age_median_v1');
      expect(result.weightSource, 'ml_weight_estimator_v1');
      expect(result.modelVersion, 'synthetic-who-v1');
      expect(result.manifestChecksum, hasLength(64));
      expect(result.trainingDataLabel, 'synthetic_who_research_only');
      expect(result.experimentalOverallCategory, 'MAM');
      expect(result.componentProbabilities, hasLength(5));
      expect(result.captureQualitySummary['used_views'], ['front', 'side']);
    });

    test('keeps a height estimate when weight cannot be estimated', () async {
      final who = FakeWhoDataService(medianWeightKg: null);
      final ml = FakeCameraMlInference(
        prediction: const WastingPrediction(
          estimatedWeightKg: null,
          samProbability: double.nan,
          mamProbability: double.nan,
          normalProbability: double.nan,
          riskProbability: double.nan,
          overweightProbability: double.nan,
          wastingStatus: 'MAM',
        ),
      );

      final result = await service(who: who, ml: ml).run(
        visit: screeningVisit(),
        acceptedAssets: acceptedAssets(),
        version: 1,
      );

      expect(result.estimatedHeightCm, 90);
      expect(result.estimatedWeightKg, isNull);
      expect(result.estimatedWhz, isNull);
      expect(result.experimentalOverallCategory, isNull);
    });

    test('keeps valid weight but omits an invalid classifier category',
        () async {
      final ml = FakeCameraMlInference(
        prediction: const WastingPrediction(
          estimatedWeightKg: 11,
          samProbability: double.nan,
          mamProbability: 0.6,
          normalProbability: 0.2,
          riskProbability: 0.1,
          overweightProbability: 0.1,
          wastingStatus: 'MAM',
        ),
      );

      final result = await service(ml: ml).run(
        visit: screeningVisit(),
        acceptedAssets: acceptedAssets(),
        version: 1,
      );

      expect(result.estimatedWeightKg, 11);
      expect(result.experimentalOverallCategory, isNull);
      expect(result.componentProbabilities, isEmpty);
    });

    test('labels WHO statistical weight fallback and fabricates no category',
        () async {
      final result = await service(
        ml: FakeCameraMlInference(error: StateError('model unavailable')),
      ).run(
        visit: screeningVisit(),
        acceptedAssets: acceptedAssets(),
        version: 1,
      );

      expect(result.estimatedWeightKg, 12);
      expect(
        result.weightSource,
        'who_weight_for_height_median_body_build_v1',
      );
      expect(result.experimentalOverallCategory, isNull);
    });

    test('non-finite model weight uses the explicit WHO fallback', () async {
      final ml = FakeCameraMlInference(
        prediction: const WastingPrediction(
          estimatedWeightKg: double.infinity,
          samProbability: 0.1,
          mamProbability: 0.6,
          normalProbability: 0.2,
          riskProbability: 0.05,
          overweightProbability: 0.05,
          wastingStatus: 'MAM',
        ),
      );

      final result = await service(ml: ml).run(
        visit: screeningVisit(),
        acceptedAssets: acceptedAssets(),
        version: 1,
      );

      expect(result.estimatedWeightKg, 12);
      expect(
        result.weightSource,
        'who_weight_for_height_median_body_build_v1',
      );
    });
  });

  group('CameraScreeningWorkflow', () {
    late AppDatabase db;
    late GuidedVisitDao visitDao;
    late CaptureAssetDao assetDao;
    late CameraResultDao resultDao;
    late int childId;

    setUp(() async {
      db = AppDatabase.forTesting(NativeDatabase.memory());
      visitDao = GuidedVisitDao(db);
      assetDao = CaptureAssetDao(db);
      resultDao = CameraResultDao(db);
      childId = await db.into(db.children).insert(
            ChildrenCompanion.insert(
              name: 'Child 001',
              dateOfBirth: '2024-01-01',
              sex: 'F',
              ownerUserId: const Value(7),
            ),
          );
      await visitDao.createDraft(
        childId: childId,
        ownerUserId: 7,
        localUuid: screeningVisit().visitUuid,
        visitDate: DateTime(2026, 7, 29),
        ageMonths: 30,
        deviceMetadataJson: '{}',
        consentVersion: 'guided_capture_consent_v1',
        consentTimestamp: DateTime(2026, 7, 29),
        consentOperatorIdentifier: 'worker-7',
      );
      await assetDao.saveAcceptedAssets(
        ownerUserId: 7,
        visitUuid: screeningVisit().visitUuid,
        assets: [
          AcceptedCaptureAsset(
            assetUuid: '20000000-0000-0000-0000-000000000001',
            role: 'front',
            localPath: '/visit/front.jpg',
            capturedAt: DateTime(2026, 7, 29, 10),
            overallScore: 0.9,
            payloadJson: '{"role":"front"}',
          ),
          AcceptedCaptureAsset(
            assetUuid: '20000000-0000-0000-0000-000000000002',
            role: 'side',
            localPath: '/visit/side.jpg',
            capturedAt: DateTime(2026, 7, 29, 10, 1),
            overallScore: 0.8,
            payloadJson: '{"role":"side"}',
          ),
        ],
      );
    });

    tearDown(() => db.close());

    test('reprocessing appends version 2 without changing version 1', () async {
      final workflow = CameraScreeningWorkflow(
        database: db,
        visitDao: visitDao,
        cameraResultDao: resultDao,
        runner: _VersionedRunner(),
      );

      await workflow.process(
        ownerUserId: 7,
        visitUuid: screeningVisit().visitUuid,
      );
      await workflow.process(
        ownerUserId: 7,
        visitUuid: screeningVisit().visitUuid,
      );

      final versions = await resultDao.getVersions(
        ownerUserId: 7,
        visitUuid: screeningVisit().visitUuid,
      );
      expect(versions.map((result) => result.version), [1, 2]);
      expect(versions.map((result) => result.estimatedHeightCm), [88, 89]);
      expect(versions[1].supersedesResultUuid, versions[0].resultUuid);
    });

    test('failure keeps accepted assets and marks processing_failed', () async {
      final workflow = CameraScreeningWorkflow(
        database: db,
        visitDao: visitDao,
        cameraResultDao: resultDao,
        runner: _FailingRunner(),
      );

      await expectLater(
        workflow.process(
          ownerUserId: 7,
          visitUuid: screeningVisit().visitUuid,
        ),
        throwsStateError,
      );

      final visit = await visitDao.getByUuid(
        ownerUserId: 7,
        visitUuid: screeningVisit().visitUuid,
      );
      expect(visit!.captureState, 'processing_failed');
      expect(await db.select(db.captureAssets).get(), hasLength(2));
      expect(await db.select(db.cameraResults).get(), isEmpty);
    });
  });
}

class _VersionedRunner implements CameraScreeningRunner {
  @override
  Future<CameraScreeningResult> run({
    required CameraScreeningVisit visit,
    required List<CameraScreeningAsset> acceptedAssets,
    required int version,
    String? supersedesResultUuid,
  }) async {
    return CameraScreeningResult(
      resultUuid:
          '30000000-0000-0000-0000-${version.toString().padLeft(12, '0')}',
      version: version,
      supersedesResultUuid: supersedesResultUuid,
      estimatedHeightCm: 87.0 + version,
      heightSource: 'who_height_for_age_median_v1',
      captureQualitySummary: const {
        'used_views': ['front', 'side']
      },
      method: cameraScreeningMethodV1,
      modelVersion: 'synthetic-who-v1',
      manifestChecksum: 'a' * 64,
      trainingDataLabel: 'synthetic_who_research_only',
      createdAt: DateTime.utc(2026, 7, 29),
    );
  }
}

class _FailingRunner implements CameraScreeningRunner {
  @override
  Future<CameraScreeningResult> run({
    required CameraScreeningVisit visit,
    required List<CameraScreeningAsset> acceptedAssets,
    required int version,
    String? supersedesResultUuid,
  }) {
    throw StateError('inference failed');
  }
}
