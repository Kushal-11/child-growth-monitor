import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/database/daos/child_dao.dart';
import 'package:child_growth_monitor_app/database/daos/sync_queue_dao.dart';
import 'package:child_growth_monitor_app/database/daos/visit_dao.dart';
import 'package:child_growth_monitor_app/models/body_measurements.dart';
import 'package:child_growth_monitor_app/models/wasting_features.dart';
import 'package:child_growth_monitor_app/services/assessment_service.dart';
import 'package:child_growth_monitor_app/services/measurement_service.dart';
import 'package:child_growth_monitor_app/services/pose_source.dart';
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';
import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

import '../fixtures/who_test_data.dart';

class _StubPose implements PoseSource {
  @override
  Future<BodySegments> segmentsFor(String _) async => const BodySegments(
        headHeightPx: 100,
        torsoLengthPx: 240,
        legLengthPx: 380,
        shoulderWidthPx: 160,
        hipWidthPx: 140,
        upperArmLengthPx: 120,
        totalHeightPx: 800,
        headTopY: 0,
        chinY: 100,
        shoulderMidpointY: 200,
        hipMidpointY: 440,
        heelY: 800,
        headConfidence: 1,
        torsoConfidence: 1,
        legConfidence: 1,
        hipConfidence: 1,
        armConfidence: 1,
      );
  @override
  Future<SideViewSegments?> sideSegmentsFor(String _) async => null;
  @override
  Future<double> confidenceFor(String _) async => 0.9;
}

class _DegradedPose implements PoseSource {
  @override
  Future<BodySegments> segmentsFor(String _) async => const BodySegments(
        headHeightPx: null,
        torsoLengthPx: null,
        legLengthPx: null,
        shoulderWidthPx: null,
        hipWidthPx: null,
        upperArmLengthPx: null,
        totalHeightPx: null,
        headTopY: null,
        chinY: null,
        shoulderMidpointY: null,
        hipMidpointY: null,
        heelY: null,
        headConfidence: 0,
        torsoConfidence: 0,
        legConfidence: 0,
        hipConfidence: 0,
        armConfidence: 0,
      );
  @override
  Future<SideViewSegments?> sideSegmentsFor(String _) async => null;
  @override
  Future<double> confidenceFor(String _) async => 0;
}

class _LowConfidencePose extends _StubPose {
  @override
  Future<double> confidenceFor(String _) async => 0.49;
}

class _StubMl extends MlInferenceService {
  WastingPrediction? canned;
  Object? throwOnPredict;

  @override
  Future<void> load() async {}

  @override
  WastingPrediction predict(WastingFeatures features) {
    if (throwOnPredict != null) throw throwOnPredict!;
    return canned ??
        const WastingPrediction(
          estimatedWeightKg: 11.0,
          samProbability: 0.02,
          mamProbability: 0.05,
          normalProbability: 0.90,
          riskProbability: 0.02,
          overweightProbability: 0.01,
          wastingStatus: 'Normal',
        );
  }

  @override
  bool weightWithinBounds({
    required double predictedKg,
    required double whoMedianKg,
  }) =>
      true;
}

void main() {
  late AppDatabase db;
  late AssessmentService svc;
  late _StubMl ml;

  setUp(() async {
    db = AppDatabase.forTesting(NativeDatabase.memory());
    final who = WhoDataService();
    await loadWhoForTests(who);
    ml = _StubMl();
    svc = AssessmentService(
      childDao: ChildDao(db),
      visitDao: VisitDao(db),
      syncQueueDao: SyncQueueDao(db),
      pose: _StubPose(),
      measurement: MeasurementService(who),
      nutrition: NutritionService(who),
      who: who,
      ml: ml,
      persistImage: (path) async => path,
    );
  });

  tearDown(() async => db.close());

  test('estimated-only path is Indeterminate and enqueues atomically',
      () async {
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Aisha',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      ownerUserId: 1,
    );

    expect(result.nutrition.hazStatus, isNull);
    expect(result.nutrition.whzStatus, isNull);
    expect(result.mlPrediction, isNotNull);
    expect(result.mlPrediction!.wastingStatus, 'Normal');
    expect(result.poshan!.finalStatus, 'Indeterminate');
    expect(result.summary, 'Indeterminate');

    final pending = await db.select(db.syncQueue).get();
    expect(pending.length, 1);
    expect(pending.first.status, 'pending');

    final visits = await db.select(db.visits).get();
    expect(visits.length, 1);
    expect(visits.first.localUuid.length, 36);
    final stored = await db.select(db.measurements).getSingle();
    expect(stored.heightSource, 'who_statistical');
    expect(stored.weightSource, 'ml_estimated');
    expect(stored.muacStatus, 'Indeterminate');
    expect(stored.poshanStatus, 'Indeterminate');
    expect(stored.classificationMethod, 'poshan_setu_v1');
  });

  test('ML failure stores no fabricated ML classification', () async {
    ml.throwOnPredict = StateError('boom');
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Bilal',
      dateOfBirth: '2024-06-01',
      sex: 'M',
      ownerUserId: 1,
    );

    expect(result.mlPrediction, isNull);
    expect(result.measurement.estimationMethod, isNotNull);
    expect(result.muac, isNotNull);
    final stored = await db.select(db.measurements).get();
    expect(stored.length, 1);
    expect(stored.first.whzStatus, isNull);
    expect(stored.first.wastingStatus, isNull);
    expect(stored.first.weightSource, 'who_statistical');
  });

  test('throws PoseDetectionFailedException when totalHeightPx missing',
      () async {
    final who = WhoDataService();
    var persistedImages = 0;
    final svcDegraded = AssessmentService(
      childDao: ChildDao(db),
      visitDao: VisitDao(db),
      syncQueueDao: SyncQueueDao(db),
      pose: _DegradedPose(),
      measurement: MeasurementService(who),
      nutrition: NutritionService(who),
      who: who,
      ml: ml,
      persistImage: (path) async {
        persistedImages++;
        return path;
      },
    );

    // The service should fail BEFORE touching the WHO/measurement services,
    // so passing un-loaded WhoDataService instances is fine.
    await expectLater(
      svcDegraded.runAssessment(
        frontImagePath: '/tmp/front.jpg',
        childName: 'Carmen',
        dateOfBirth: '2024-03-01',
        sex: 'F',
        ownerUserId: 1,
      ),
      throwsA(isA<PoseDetectionFailedException>()),
    );
    expect(
      persistedImages,
      0,
      reason: 'invalid temporary captures must not reach permanent storage',
    );
  });

  test('rejects pose confidence below the safety floor', () async {
    final who = WhoDataService();
    await loadWhoForTests(who);
    final lowConfidence = AssessmentService(
      childDao: ChildDao(db),
      visitDao: VisitDao(db),
      syncQueueDao: SyncQueueDao(db),
      pose: _LowConfidencePose(),
      measurement: MeasurementService(who),
      nutrition: NutritionService(who),
      who: who,
      ml: ml,
      persistImage: (path) async => path,
    );

    await expectLater(
      lowConfidence.runAssessment(
        frontImagePath: '/tmp/front.jpg',
        childName: 'Low confidence',
        dateOfBirth: '2024-03-01',
        sex: 'F',
        ownerUserId: 1,
      ),
      throwsA(isA<PoseDetectionFailedException>()),
    );
    expect(await db.select(db.visits).get(), isEmpty);
  });

  test('MUAC in SAM range escalates the summary to SAM even when WHZ is Normal',
      () async {
    // Tape-measured SAM (MUAC < 11.5) with an otherwise-normal weight: the
    // WHO OR-rule must surface SAM, not the green "Normal" the WHZ alone gives.
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Fatima',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      manualMuacCm: 10.0,
      ownerUserId: 1,
    );

    expect(result.muac!.muacStatus, 'SAM');
    final whz = result.nutrition.whzStatus;
    expect(whz == null || !whz.contains('SAM'), isTrue,
        reason: 'precondition: WHZ itself is not SAM in this scenario');
    expect(result.summary, 'SAM');
  });

  test('manual MUAC is not rounded before Poshan classification', () async {
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Boundary child',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      manualMuacCm: 11.49,
      ownerUserId: 1,
    );

    expect(result.muac!.muacCm, 11.49);
    expect(result.poshan!.muacStatus, 'SAM');
    expect(result.poshan!.finalStatus, 'SAM');
    final stored = await db.select(db.measurements).getSingle();
    expect(stored.muacCm, 11.49);
    expect(stored.poshanStatus, 'SAM');
  });

  test('ML wasting SAM stays secondary to an Indeterminate Poshan result',
      () async {
    ml.canned = const WastingPrediction(
      estimatedWeightKg: 11.0,
      samProbability: 0.85,
      mamProbability: 0.08,
      normalProbability: 0.04,
      riskProbability: 0.02,
      overweightProbability: 0.01,
      wastingStatus: 'SAM',
    );

    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Gita',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      ownerUserId: 1,
    );

    expect(result.mlPrediction!.wastingStatus, 'SAM');
    final whz = result.nutrition.whzStatus;
    expect(whz == null || !whz.contains('SAM'), isTrue,
        reason: 'precondition: WHZ itself is not SAM in this scenario');
    expect(result.summary, 'Indeterminate',
        reason: 'ML remains secondary and cannot determine Poshan Setu');
  });

  test('manual height weight and tape MUAC produce a final Normal result',
      () async {
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Measured child',
      dateOfBirth: '2024-01-01',
      sex: 'F',
      manualHeightCm: 100,
      manualWeightKg: 13.5,
      manualMuacCm: 12.5,
      ownerUserId: 1,
    );

    expect(result.poshan!.bmiStatus, 'Normal');
    expect(result.poshan!.muacStatus, 'Normal');
    expect(result.poshan!.finalStatus, 'Normal');
    expect(result.poshan!.complete, true);
    final stored = await db.select(db.measurements).getSingle();
    expect(stored.effectiveHeightCm, 100);
    expect(stored.effectiveWeightKg, 13.5);
    expect(stored.heightSource, 'manual');
    expect(stored.weightSource, 'manual');
    expect(stored.predictedHeightCm, isNull);
    expect(stored.predictedWeightKg, isNull);
    expect(stored.manualHeightCm, 100);
    expect(stored.manualWeightKg, 13.5);
    expect(stored.poshanStatus, 'Normal');
  });

  test('runAssessment tags created child with ownerUserId', () async {
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Owned Assessment Child',
      dateOfBirth: '2022-06-01',
      sex: 'M',
      ownerUserId: 9001,
    );
    expect(result.childName, 'Owned Assessment Child');

    final children = await db.select(db.children).get();
    final created =
        children.firstWhere((c) => c.name == 'Owned Assessment Child');
    expect(created.ownerUserId, 9001);
    final visit = await db.select(db.visits).getSingle();
    expect(visit.ownerUserId, 9001);
  });
}
