import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:uuid/uuid.dart';

import '../../../database/daos/capture_asset_dao.dart';
import '../../../database/daos/guided_visit_dao.dart';
import '../../../database/database.dart';
import '../../../services/age_service.dart';
import '../domain/capture_models.dart';
import '../services/guided_camera_controller.dart';

class GuidedCaptureChild {
  const GuidedCaptureChild({
    required this.id,
    required this.ownerUserId,
    required this.name,
    required this.dateOfBirth,
    required this.sex,
  });

  final int id;
  final int ownerUserId;
  final String name;
  final String dateOfBirth;
  final String sex;
}

class GuidedCaptureSnapshot {
  const GuidedCaptureSnapshot({
    required this.child,
    required this.visitUuid,
    required this.captureState,
    required this.acceptedFrames,
  });

  final GuidedCaptureChild child;
  final String visitUuid;
  final CaptureState captureState;
  final Map<CaptureAssetRole, List<GuidedRetainedFrame>> acceptedFrames;
}

abstract interface class GuidedCaptureRepository {
  Future<GuidedCaptureChild?> getOwnerChild({
    required int childId,
    required int ownerUserId,
  });

  Future<GuidedCaptureSnapshot> createDraft({
    required GuidedCaptureChild child,
    required String visitUuid,
    required DateTime visitDate,
    required String deviceMetadataJson,
    required String consentVersion,
    required DateTime consentTimestamp,
    required String consentOperatorIdentifier,
  });

  Future<GuidedCaptureSnapshot?> loadDraft({
    required int ownerUserId,
    required String visitUuid,
  });

  Future<void> saveAcceptedFrames({
    required int ownerUserId,
    required String visitUuid,
    required List<GuidedRetainedFrame> frames,
  });

  Future<void> markIncomplete({
    required int ownerUserId,
    required String visitUuid,
  });
}

class DriftGuidedCaptureRepository implements GuidedCaptureRepository {
  DriftGuidedCaptureRepository({
    required AppDatabase database,
    required GuidedVisitDao visitDao,
    required CaptureAssetDao captureAssetDao,
    Uuid uuid = const Uuid(),
  })  : _database = database,
        _visitDao = visitDao,
        _captureAssetDao = captureAssetDao,
        _uuid = uuid;

  final AppDatabase _database;
  final GuidedVisitDao _visitDao;
  final CaptureAssetDao _captureAssetDao;
  final Uuid _uuid;

  @override
  Future<GuidedCaptureChild?> getOwnerChild({
    required int childId,
    required int ownerUserId,
  }) async {
    final child = await (_database.select(_database.children)
          ..where(
            (row) =>
                row.id.equals(childId) &
                row.ownerUserId.equals(ownerUserId) &
                row.isArchived.equals(false),
          ))
        .getSingleOrNull();
    if (child == null || child.ownerUserId == null) return null;
    return _childFromRow(child);
  }

  @override
  Future<GuidedCaptureSnapshot> createDraft({
    required GuidedCaptureChild child,
    required String visitUuid,
    required DateTime visitDate,
    required String deviceMetadataJson,
    required String consentVersion,
    required DateTime consentTimestamp,
    required String consentOperatorIdentifier,
  }) async {
    final dateOfBirth = DateTime.parse(child.dateOfBirth);
    late final double ageMonths;
    try {
      ageMonths = AgeService.ageMonthsAt(dateOfBirth, visitDate);
    } on ArgumentError {
      throw StateError('Visit date cannot be before date of birth');
    }
    await _visitDao.createDraft(
      childId: child.id,
      ownerUserId: child.ownerUserId,
      localUuid: visitUuid,
      visitDate: visitDate,
      ageMonths: ageMonths,
      deviceMetadataJson: deviceMetadataJson,
      consentVersion: consentVersion,
      consentTimestamp: consentTimestamp,
      consentOperatorIdentifier: consentOperatorIdentifier,
    );
    return GuidedCaptureSnapshot(
      child: child,
      visitUuid: visitUuid,
      captureState: CaptureState.draftCapture,
      acceptedFrames: const {},
    );
  }

  @override
  Future<GuidedCaptureSnapshot?> loadDraft({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    final visit = await _visitDao.getByUuid(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    if (visit == null || visit.entryMethod != 'guided_capture') return null;
    final child = await getOwnerChild(
      childId: visit.childId,
      ownerUserId: ownerUserId,
    );
    if (child == null) return null;
    final assets = await _captureAssetDao.getForVisit(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
    final accepted = <CaptureAssetRole, List<GuidedRetainedFrame>>{};
    for (final asset in assets) {
      final role = CaptureAssetRole.fromWire(asset.role);
      accepted.putIfAbsent(role, () => []).add(
            GuidedRetainedFrame(
              localPath: asset.localPath ?? '',
              role: role,
              capturedAt: asset.capturedAt,
              selectedRank: asset.selectedRank ?? 1,
              poseScore: asset.poseScore ?? 0,
              coverageScore: asset.coverageScore ?? 0,
              orientationScore: asset.orientationScore ?? 0,
              sharpnessScore: asset.sharpnessScore ?? 0,
              lightingScore: asset.lightingScore ?? 0,
              overallScore: asset.overallScore ?? 0,
              qualityThresholdVersion:
                  asset.qualityThresholdVersion ?? 'unknown',
              imageWidth: asset.imageWidth ?? 0,
              imageHeight: asset.imageHeight ?? 0,
              exifOrientation: asset.exifOrientation,
              displayOrientation: asset.displayOrientation ?? 0,
              cameraIdentifier: _cameraMetadataValue(
                asset.deviceCameraMetadataJson,
                'camera_identifier',
              ),
              lensDirection: _cameraMetadataValue(
                asset.deviceCameraMetadataJson,
                'lens_direction',
              ),
              deviceMetadataJson: asset.deviceCameraMetadataJson ?? '{}',
            ),
          );
    }
    return GuidedCaptureSnapshot(
      child: child,
      visitUuid: visitUuid,
      captureState: CaptureState.fromWire(
        visit.captureState ?? CaptureState.draftCapture.wireValue,
      ),
      acceptedFrames: accepted,
    );
  }

  @override
  Future<void> saveAcceptedFrames({
    required int ownerUserId,
    required String visitUuid,
    required List<GuidedRetainedFrame> frames,
  }) async {
    if (frames.isEmpty) {
      throw ArgumentError.value(frames, 'frames', 'must not be empty');
    }
    final accepted = <AcceptedCaptureAsset>[];
    for (final frame in frames) {
      final assetUuid = _uuid.v4();
      final payload = jsonEncode({
        'asset_uuid': assetUuid,
        'visit_uuid': visitUuid,
        'role': frame.role.wireValue,
        'captured_at': frame.capturedAt.toIso8601String(),
        'selected_rank': frame.selectedRank,
        'quality': {
          'pose': frame.poseScore,
          'coverage': frame.coverageScore,
          'orientation': frame.orientationScore,
          'sharpness': frame.sharpnessScore,
          'lighting': frame.lightingScore,
          'overall': frame.overallScore,
          'threshold_version': frame.qualityThresholdVersion,
        },
        'image_width': frame.imageWidth,
        'image_height': frame.imageHeight,
        'exif_orientation': frame.exifOrientation,
        'display_orientation': frame.displayOrientation,
        'device_camera_metadata': jsonDecode(frame.deviceMetadataJson),
      });
      accepted.add(
        AcceptedCaptureAsset(
          assetUuid: assetUuid,
          role: frame.role.wireValue,
          localPath: frame.localPath,
          capturedAt: frame.capturedAt,
          payloadJson: payload,
          selectedRank: frame.selectedRank,
          poseScore: frame.poseScore,
          coverageScore: frame.coverageScore,
          orientationScore: frame.orientationScore,
          sharpnessScore: frame.sharpnessScore,
          lightingScore: frame.lightingScore,
          overallScore: frame.overallScore,
          qualityThresholdVersion: frame.qualityThresholdVersion,
          imageWidth: frame.imageWidth,
          imageHeight: frame.imageHeight,
          exifOrientation: frame.exifOrientation,
          displayOrientation: frame.displayOrientation,
          deviceCameraMetadataJson: frame.deviceMetadataJson,
        ),
      );
    }
    await _captureAssetDao.saveAcceptedAssets(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
      assets: accepted,
    );
  }

  @override
  Future<void> markIncomplete({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    await _visitDao.markIncompleteCapture(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
  }

  GuidedCaptureChild _childFromRow(ChildrenData child) => GuidedCaptureChild(
        id: child.id,
        ownerUserId: child.ownerUserId!,
        name: child.name,
        dateOfBirth: child.dateOfBirth,
        sex: child.sex,
      );

  String _cameraMetadataValue(String? raw, String key) {
    if (raw == null) return 'unknown';
    try {
      return (jsonDecode(raw) as Map<String, dynamic>)[key]?.toString() ??
          'unknown';
    } on Object {
      return 'unknown';
    }
  }
}
