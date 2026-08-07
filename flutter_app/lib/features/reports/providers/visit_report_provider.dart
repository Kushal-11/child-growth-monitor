import 'dart:convert';

import 'package:drift/drift.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../database/database.dart';
import '../../../features/guided_capture/domain/camera_screening_result.dart';
import '../../../features/guided_capture/domain/capture_models.dart';
import '../../../providers/assessment_service_provider.dart';
import '../../../providers/database_provider.dart';

class VisitReportSnapshot {
  const VisitReportSnapshot({
    required this.visitUuid,
    required this.visitDate,
    required this.captureState,
    required this.latestCameraResult,
    required this.acceptedAssetCount,
    this.measuredReport,
  });

  final String visitUuid;
  final DateTime visitDate;
  final CaptureState captureState;
  final CameraScreeningResult? latestCameraResult;
  final int acceptedAssetCount;
  final MeasuredReportSnapshot? measuredReport;
}

class MeasuredReportSnapshot {
  const MeasuredReportSnapshot({
    this.heightCm,
    this.weightKg,
    this.muacCm,
    this.hazZscore,
    this.hazStatus,
    this.whzZscore,
    this.whzStatus,
    this.muacStatus,
    this.muacEligible,
    this.oedema,
    this.whoAcuteStatus,
    this.whoAcuteTriggeredBy = const [],
    this.whoAcuteRationale,
    this.poshanStatus,
    this.poshanTriggeredBy = const [],
    this.poshanComplete,
    this.classificationMethod,
    this.classificationRationale,
    this.measuredAt,
  });

  final double? heightCm;
  final double? weightKg;
  final double? muacCm;
  final double? hazZscore;
  final String? hazStatus;
  final double? whzZscore;
  final String? whzStatus;
  final String? muacStatus;
  final bool? muacEligible;
  final String? oedema;
  final String? whoAcuteStatus;
  final List<String> whoAcuteTriggeredBy;
  final String? whoAcuteRationale;
  final String? poshanStatus;
  final List<String> poshanTriggeredBy;
  final bool? poshanComplete;
  final String? classificationMethod;
  final String? classificationRationale;
  final DateTime? measuredAt;
}

class VisitReportRequest {
  const VisitReportRequest({
    required this.visitUuid,
    required this.ownerUserId,
  });

  final String visitUuid;
  final int ownerUserId;

  @override
  bool operator ==(Object other) =>
      other is VisitReportRequest &&
      other.visitUuid == visitUuid &&
      other.ownerUserId == ownerUserId;

  @override
  int get hashCode => Object.hash(visitUuid, ownerUserId);
}

abstract interface class VisitReportRepository {
  Future<VisitReportSnapshot> load({
    required int ownerUserId,
    required String visitUuid,
  });
}

class DriftVisitReportRepository implements VisitReportRepository {
  DriftVisitReportRepository(this._database);

  final AppDatabase _database;

  @override
  Future<VisitReportSnapshot> load({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    final visit = await (_database.select(_database.visits)
          ..where(
            (row) =>
                row.localUuid.equals(visitUuid) &
                row.ownerUserId.equals(ownerUserId) &
                row.entryMethod.equals('guided_capture'),
          ))
        .getSingleOrNull();
    if (visit == null || visit.captureState == null) {
      throw StateError('Owner-scoped guided visit was not found');
    }
    final results = await (_database.select(_database.cameraResults)
          ..where((row) => row.visitId.equals(visit.id))
          ..orderBy([(row) => OrderingTerm.desc(row.version)])
          ..limit(1))
        .get();
    final acceptedAssets = await (_database.select(_database.captureAssets)
          ..where(
            (row) =>
                row.visitId.equals(visit.id) &
                row.qualityVerdict.equals('accepted'),
          ))
        .get();
    final measurement = await (_database.select(_database.measurements)
          ..where((row) => row.visitId.equals(visit.id)))
        .getSingleOrNull();
    return VisitReportSnapshot(
      visitUuid: visit.localUuid,
      visitDate: visit.visitDate,
      captureState: CaptureState.fromWire(visit.captureState!),
      latestCameraResult:
          results.isEmpty ? null : _cameraResultFromRow(results.single),
      acceptedAssetCount: acceptedAssets.length,
      measuredReport:
          measurement == null ? null : _measuredReportFromRow(measurement),
    );
  }

  MeasuredReportSnapshot _measuredReportFromRow(Measurement row) {
    return MeasuredReportSnapshot(
      heightCm: row.manualHeightCm,
      weightKg: row.manualWeightKg,
      muacCm: row.muacCm,
      hazZscore: row.hazZscore,
      hazStatus: row.hazStatus,
      whzZscore: row.whzZscore,
      whzStatus: row.whzStatus,
      muacStatus: row.muacStatus,
      muacEligible: row.muacAgeInRange,
      oedema: row.oedema,
      whoAcuteStatus: row.whoAcuteStatus,
      whoAcuteTriggeredBy: _decodeStringList(row.whoAcuteTriggeredBy),
      whoAcuteRationale: row.whoAcuteRationale,
      poshanStatus: row.poshanStatus,
      poshanTriggeredBy: _decodeStringList(row.poshanTriggeredBy),
      poshanComplete: row.poshanComplete,
      classificationMethod: row.classificationMethod,
      classificationRationale: row.classificationRationale,
      measuredAt: row.measuredAt,
    );
  }

  CameraScreeningResult _cameraResultFromRow(CameraResult row) {
    return CameraScreeningResult(
      resultUuid: row.resultUuid,
      version: row.version,
      supersedesResultUuid: row.supersedesResultUuid,
      estimatedHeightCm: row.estimatedHeightCm,
      estimatedWeightKg: row.estimatedWeightKg,
      estimatedMuacCm: row.estimatedMuacCm,
      heightSource: row.heightSource,
      weightSource: row.weightSource,
      muacSource: row.muacSource,
      heightRangeLowerCm: row.heightRangeLowerCm,
      heightRangeUpperCm: row.heightRangeUpperCm,
      weightRangeLowerKg: row.weightRangeLowerKg,
      weightRangeUpperKg: row.weightRangeUpperKg,
      muacRangeLowerCm: row.muacRangeLowerCm,
      muacRangeUpperCm: row.muacRangeUpperCm,
      estimatedHaz: row.estimatedHaz,
      estimatedWhz: row.estimatedWhz,
      estimatedStuntingStatus: row.estimatedStuntingStatus,
      estimatedWastingStatus: row.estimatedWastingStatus,
      experimentalOverallCategory: row.experimentalOverallCategory,
      componentProbabilities: _decodeDoubleMap(
        row.componentProbabilitiesJson,
      ),
      bodyProportionFeatures: _decodeObjectMap(
        row.bodyProportionFeaturesJson,
      ),
      captureQualitySummary: _decodeObjectMap(
        row.captureQualitySummaryJson,
      ),
      method: row.method,
      modelVersion: row.modelVersion,
      manifestChecksum: row.manifestChecksum,
      trainingDataLabel: row.trainingDataLabel,
      createdAt: row.createdAt,
    );
  }

  Map<String, Object?> _decodeObjectMap(String? raw) {
    if (raw == null) return const {};
    final value = jsonDecode(raw);
    if (value is! Map) return const {};
    return {
      for (final entry in value.entries) entry.key.toString(): entry.value,
    };
  }

  Map<String, double> _decodeDoubleMap(String? raw) {
    final values = _decodeObjectMap(raw);
    return {
      for (final entry in values.entries)
        if (entry.value is num) entry.key: (entry.value as num).toDouble(),
    };
  }

  List<String> _decodeStringList(String? raw) {
    if (raw == null) return const [];
    final value = jsonDecode(raw);
    if (value is! List) return const [];
    return value.map((item) => item.toString()).toList(growable: false);
  }
}

abstract interface class CameraScreeningProcessor {
  Future<void> process({
    required int ownerUserId,
    required String visitUuid,
  });
}

class ProviderCameraScreeningProcessor implements CameraScreeningProcessor {
  ProviderCameraScreeningProcessor(this._ref);

  final Ref _ref;

  @override
  Future<void> process({
    required int ownerUserId,
    required String visitUuid,
  }) async {
    final workflow = await _ref.read(cameraScreeningWorkflowProvider.future);
    await workflow.process(
      ownerUserId: ownerUserId,
      visitUuid: visitUuid,
    );
  }
}

final visitReportRepositoryProvider = Provider<VisitReportRepository>((ref) {
  return DriftVisitReportRepository(ref.watch(databaseProvider));
});

final cameraScreeningProcessorProvider =
    Provider<CameraScreeningProcessor>((ref) {
  return ProviderCameraScreeningProcessor(ref);
});

final visitReportProvider =
    FutureProvider.family<VisitReportSnapshot, VisitReportRequest>(
  (ref, request) {
    return ref.watch(visitReportRepositoryProvider).load(
          ownerUserId: request.ownerUserId,
          visitUuid: request.visitUuid,
        );
  },
);
