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
      db: db,
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

  test('happy path returns Normal result and enqueues a sync row', () async {
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Aisha',
      dateOfBirth: '2024-01-01',
      sex: 'F',
    );

    expect(result.nutrition.whzStatus, isNotNull);
    expect(result.mlPrediction, isNotNull);
    expect(result.mlPrediction!.wastingStatus, 'Normal');

    final pending = await db.select(db.syncQueue).get();
    expect(pending.length, 1);
    expect(pending.first.status, 'pending');

    final visits = await db.select(db.visits).get();
    expect(visits.length, 1);
    expect(visits.first.localUuid.length, 36);
  });

  test('ML failure produces a result labelled who_fallback', () async {
    ml.throwOnPredict = StateError('boom');
    final result = await svc.runAssessment(
      frontImagePath: '/tmp/front.jpg',
      childName: 'Bilal',
      dateOfBirth: '2024-06-01',
      sex: 'M',
    );

    expect(result.mlPrediction, isNull);
    expect(result.measurement.estimationMethod, isNotNull);
    expect(result.muac, isNotNull);
    final stored = await db.select(db.measurements).get();
    expect(stored.length, 1);
    expect(stored.first.whzStatus, isNotNull);
    expect(stored.first.wastingStatus, 'who_fallback');
  });

  test('throws PoseDetectionFailedException when totalHeightPx missing', () async {
    final who = WhoDataService();
    final svcDegraded = AssessmentService(
      db: db,
      childDao: ChildDao(db),
      visitDao: VisitDao(db),
      syncQueueDao: SyncQueueDao(db),
      pose: _DegradedPose(),
      measurement: MeasurementService(who),
      nutrition: NutritionService(who),
      who: who,
      ml: ml,
      persistImage: (path) async => path,
    );

    // The service should fail BEFORE touching the WHO/measurement services,
    // so passing un-loaded WhoDataService instances is fine.
    await expectLater(
      svcDegraded.runAssessment(
        frontImagePath: '/tmp/front.jpg',
        childName: 'Carmen',
        dateOfBirth: '2024-03-01',
        sex: 'F',
      ),
      throwsA(isA<PoseDetectionFailedException>()),
    );
  });
}
