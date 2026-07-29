import 'dart:convert';
import 'dart:io';

import 'package:child_growth_monitor_app/constants/feature_flags.dart';
import 'package:child_growth_monitor_app/database/daos/camera_result_dao.dart';
import 'package:child_growth_monitor_app/database/daos/capture_asset_dao.dart';
import 'package:child_growth_monitor_app/database/daos/guided_visit_dao.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/camera_screening_result.dart';
import 'package:child_growth_monitor_app/features/guided_capture/domain/capture_models.dart';
import 'package:child_growth_monitor_app/features/guided_capture/repositories/guided_capture_repository.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/camera_screening_service.dart';
import 'package:child_growth_monitor_app/features/guided_capture/services/guided_camera_controller.dart';
import 'package:child_growth_monitor_app/features/reports/providers/visit_report_provider.dart';
import 'package:drift/drift.dart';
import 'package:drift/native.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

class _DeviceFakeCameraRunner implements CameraScreeningRunner {
  @override
  Future<CameraScreeningResult> run({
    required CameraScreeningVisit visit,
    required List<CameraScreeningAsset> acceptedAssets,
    required int version,
    String? supersedesResultUuid,
  }) async {
    expect(
      acceptedAssets.map((asset) => asset.role).toSet(),
      containsAll({CaptureAssetRole.front, CaptureAssetRole.side}),
    );
    return CameraScreeningResult(
      resultUuid: '30000000-0000-0000-0000-000000000001',
      version: version,
      supersedesResultUuid: supersedesResultUuid,
      estimatedHeightCm: 88,
      estimatedWeightKg: 11,
      heightSource: 'device_integration_fake',
      weightSource: 'device_integration_fake',
      estimatedHaz: -1.2,
      estimatedWhz: -0.8,
      estimatedStuntingStatus: 'Normal',
      estimatedWastingStatus: 'NORMAL',
      captureQualitySummary: const {
        'overall': 0.9,
        'used_views': ['front', 'side'],
      },
      method: cameraScreeningMethodV1,
      modelVersion: 'device-integration-fake-v1',
      manifestChecksum: 'a' * 64,
      trainingDataLabel: 'test_only',
      createdAt: DateTime.now().toUtc(),
    );
  }
}

GuidedRetainedFrame _retained(String path, CaptureAssetRole role) =>
    GuidedRetainedFrame(
      localPath: path,
      role: role,
      capturedAt: DateTime.now().toUtc(),
      selectedRank: 1,
      poseScore: 0.9,
      coverageScore: 0.9,
      orientationScore: 0.9,
      sharpnessScore: 0.9,
      lightingScore: 0.9,
      overallScore: 0.9,
      qualityThresholdVersion: 'guided_capture_quality_v1',
      imageWidth: 1080,
      imageHeight: 1920,
      exifOrientation: 1,
      displayOrientation: 0,
      cameraIdentifier: 'integration-fake',
      lensDirection: 'back',
      deviceMetadataJson: jsonEncode({'platform': Platform.operatingSystem}),
    );

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets(
      'device persists offline capture, estimate and typed outbox across restart',
      (_) async {
    expect(FeatureFlags.liveCaptureEnabled, isTrue);
    final stopwatch = Stopwatch()..start();
    final root = await Directory.systemTemp.createTemp(
      'guided-device-integration-',
    );
    addTearDown(() async {
      if (await root.exists()) await root.delete(recursive: true);
    });
    final databaseFile = File('${root.path}/guided.sqlite');
    final front = File('${root.path}/front.jpg')
      ..writeAsBytesSync(List<int>.filled(1024, 1));
    final side = File('${root.path}/side.jpg')
      ..writeAsBytesSync(List<int>.filled(1024, 2));
    const visitUuid = '10000000-0000-0000-0000-000000000001';

    var database = AppDatabase.forTesting(NativeDatabase(databaseFile));
    final visitDao = GuidedVisitDao(database);
    final assetDao = CaptureAssetDao(database);
    final resultDao = CameraResultDao(database);
    final childId = await database.into(database.children).insert(
          ChildrenCompanion.insert(
            name: 'Device Test Child',
            dateOfBirth: '2024-01-29',
            sex: 'F',
            ownerUserId: const Value(7),
          ),
        );
    final repository = DriftGuidedCaptureRepository(
      database: database,
      visitDao: visitDao,
      captureAssetDao: assetDao,
    );
    final child = await repository.getOwnerChild(
      childId: childId,
      ownerUserId: 7,
    );
    await repository.createDraft(
      child: child!,
      visitUuid: visitUuid,
      visitDate: DateTime(2026, 7, 29),
      deviceMetadataJson: jsonEncode({'platform': Platform.operatingSystem}),
      consentVersion: 'guided_capture_consent_v1',
      consentTimestamp: DateTime.now().toUtc(),
      consentOperatorIdentifier: 'device-test',
    );
    await repository.saveAcceptedFrames(
      ownerUserId: 7,
      visitUuid: visitUuid,
      frames: [
        _retained(front.path, CaptureAssetRole.front),
        _retained(side.path, CaptureAssetRole.side),
      ],
    );
    final workflow = CameraScreeningWorkflow(
      database: database,
      visitDao: visitDao,
      cameraResultDao: resultDao,
      runner: _DeviceFakeCameraRunner(),
    );
    await workflow.process(ownerUserId: 7, visitUuid: visitUuid);
    await database.close();

    database = AppDatabase.forTesting(NativeDatabase(databaseFile));
    addTearDown(database.close);
    final report = await DriftVisitReportRepository(database).load(
      ownerUserId: 7,
      visitUuid: visitUuid,
    );
    final outbox = await database.select(database.syncOutbox).get();
    stopwatch.stop();

    expect(report.captureState, CaptureState.estimatedReport);
    expect(report.latestCameraResult?.estimatedHeightCm, 88);
    expect(report.acceptedAssetCount, 2);
    expect(outbox, hasLength(4));
    expect(
      outbox.map((entry) => entry.entityType),
      containsAll(['visit', 'capture_asset', 'camera_result']),
    );
    expect(outbox.every((entry) => entry.status == 'pending'), isTrue);
    expect(front.existsSync(), isTrue);
    expect(side.existsSync(), isTrue);
    expect(
      stopwatch.elapsed,
      lessThan(const Duration(seconds: 15)),
      reason: 'Local fake-inference persistence should remain field-usable',
    );
  });
}
