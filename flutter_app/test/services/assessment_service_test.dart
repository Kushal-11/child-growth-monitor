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
import 'package:child_growth_monitor_app/services/ml_inference_service.dart';
import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

import '../fixtures/who_test_data.dart';

class _StubPose {
  BodySegments segmentsFor(String _) => const BodySegments(
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
  SideViewSegments? sideSegmentsFor(String _) => null;
  double confidenceFor(String _) => 0.9;
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
}
