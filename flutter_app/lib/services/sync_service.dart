import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:drift/drift.dart';
import 'package:http/http.dart' as http;

import '../database/daos/child_dao.dart';
import '../database/daos/sync_queue_dao.dart';
import '../database/daos/visit_dao.dart';
import '../database/database.dart';

class SyncService {
  SyncService({
    required AppDatabase db,
    required VisitDao visitDao,
    required ChildDao childDao,
    required SyncQueueDao syncDao,
    required String baseUrl,
    http.Client? httpClient,
    String? authToken,
    void Function()? onUnauthorized,
  })  : _db = db,
        _visitDao = visitDao,
        _childDao = childDao,
        _syncDao = syncDao,
        _baseUrl = baseUrl,
        _authToken = authToken,
        _onUnauthorized = onUnauthorized,
        _client = httpClient ?? http.Client();

  final AppDatabase _db;
  final VisitDao _visitDao;
  final ChildDao _childDao;
  final SyncQueueDao _syncDao;
  final String _baseUrl;
  final http.Client _client;
  final String? _authToken;
  final void Function()? _onUnauthorized;

  static const _maxRetries = 5;

  Future<void> runOnce() async {
    await _recoverStuck();

    final entries = await (_db.select(_db.syncQueue)
          ..where((s) =>
              (s.status.equals('pending') | s.status.equals('failed')) &
              s.retryCount.isSmallerThanValue(_maxRetries))
          ..orderBy([(s) => OrderingTerm.asc(s.createdAt)]))
        .get();

    for (final entry in entries) {
      await _syncOne(entry);
    }
  }

  /// Resets any entries stuck in 'syncing' state back to 'pending'.
  /// Defends against app crashes (SIGKILL, OOM, force-stop) that left
  /// entries mid-flight.
  Future<void> _recoverStuck() async {
    await (_db.update(_db.syncQueue)..where((s) => s.status.equals('syncing')))
        .write(const SyncQueueCompanion(status: Value('pending')));
  }

  Future<void> _syncOne(SyncQueueData entry) async {
    await _syncDao.markSyncing(entry.id);
    try {
      final pair = await _visitDao.getById(entry.visitId);
      if (pair == null) {
        await _syncDao.markFailed(entry.id, 'Visit not found');
        return;
      }
      final child = await _childDao.getById(pair.visit.childId);
      if (child == null) {
        await _syncDao.markFailed(entry.id, 'Child not found');
        return;
      }
      final m = pair.measurement;

      final uri = Uri.parse('$_baseUrl/api/v1/sync');
      final req = http.MultipartRequest('POST', uri);
      if (_authToken != null) {
        req.headers['Authorization'] = 'Bearer $_authToken';
      }
      req.fields.addAll({
        'local_uuid': pair.visit.localUuid,
        'child_name': child.name,
        'date_of_birth': child.dateOfBirth,
        'sex': child.sex,
        'age_months': pair.visit.ageMonths.toString(),
        'visit_date': pair.visit.visitDate.toIso8601String(),
        'entry_method': pair.visit.entryMethod,
        'is_archived': child.isArchived.toString(),
        if (m?.predictedHeightCm != null)
          'predicted_height_cm': m!.predictedHeightCm.toString(),
        if (m?.predictedWeightKg != null)
          'predicted_weight_kg': m!.predictedWeightKg.toString(),
        if (m?.manualHeightCm != null)
          'manual_height_cm': m!.manualHeightCm.toString(),
        if (m?.manualWeightKg != null)
          'manual_weight_kg': m!.manualWeightKg.toString(),
        if (m?.hazZscore != null) 'haz_zscore': m!.hazZscore.toString(),
        if (m?.whzZscore != null) 'whz_zscore': m!.whzZscore.toString(),
        if (m?.hazStatus != null) 'haz_status': m!.hazStatus!,
        if (m?.whzStatus != null) 'whz_status': m!.whzStatus!,
        if (m?.confidenceScore != null)
          'confidence_score': m!.confidenceScore.toString(),
        if (m?.effectiveHeightCm != null)
          'effective_height_cm': m!.effectiveHeightCm.toString(),
        if (m?.effectiveWeightKg != null)
          'effective_weight_kg': m!.effectiveWeightKg.toString(),
        if (m?.heightMethod != null) 'height_method': m!.heightMethod!,
        if (m?.weightMethod != null) 'weight_method': m!.weightMethod!,
        if (m?.estimationMethod != null)
          'estimation_method': m!.estimationMethod!,
        if (m?.bmi != null) 'bmi': m!.bmi.toString(),
        if (m?.bmiStatus != null) 'bmi_status': m!.bmiStatus!,
        if (m?.heightConfidence != null)
          'height_confidence': m!.heightConfidence.toString(),
        if (m?.weightConfidence != null)
          'weight_confidence': m!.weightConfidence.toString(),
        if (m?.classificationConfidence != null)
          'classification_confidence': m!.classificationConfidence.toString(),
        if (m?.bodyBuild != null) 'body_build': m!.bodyBuild!,
        'side_view_used': (m?.sideViewUsed ?? false).toString(),
        if (m?.chestDepthCm != null)
          'chest_depth_cm': m!.chestDepthCm.toString(),
        if (m?.abdDepthCm != null) 'abd_depth_cm': m!.abdDepthCm.toString(),
        if (m?.mlEstimatedWeightKg != null)
          'ml_estimated_weight_kg': m!.mlEstimatedWeightKg.toString(),
        if (m?.wastingStatus != null) 'ml_wasting_status': m!.wastingStatus!,
        if (m?.wastingMethod != null) 'ml_wasting_method': m!.wastingMethod!,
        if (m?.samProbability != null)
          'sam_probability': m!.samProbability.toString(),
        if (m?.mamProbability != null)
          'mam_probability': m!.mamProbability.toString(),
        if (m?.normalProbability != null)
          'normal_probability': m!.normalProbability.toString(),
        // Cross-system naming bridge: Flutter calls this riskOverweightProbability,
        // backend calls it risk_probability.
        if (m?.riskOverweightProbability != null)
          'risk_probability': m!.riskOverweightProbability.toString(),
        if (m?.overweightProbability != null)
          'overweight_probability': m!.overweightProbability.toString(),
        if (m?.muacCm != null) 'muac_cm': m!.muacCm.toString(),
        if (m?.muacStatus != null) 'muac_status': m!.muacStatus!,
        if (m?.muacMethod != null) 'muac_method': m!.muacMethod!,
        if (m?.muacAgeInRange != null)
          'muac_age_in_range': m!.muacAgeInRange.toString(),
        if (m?.muacConfidence != null)
          'muac_confidence': m!.muacConfidence.toString(),
        if (m?.muacUncertaintyLowerCm != null)
          'muac_uncertainty_lower_cm': m!.muacUncertaintyLowerCm.toString(),
        if (m?.muacUncertaintyUpperCm != null)
          'muac_uncertainty_upper_cm': m!.muacUncertaintyUpperCm.toString(),
        if (m?.muacModelVersion != null)
          'muac_model_version': m!.muacModelVersion!,
        if (m?.muacCalibrationVersion != null)
          'muac_calibration_version': m!.muacCalibrationVersion!,
        if (m?.muacIsDirectMeasurement != null)
          'muac_is_direct_measurement': m!.muacIsDirectMeasurement.toString(),
        if (m?.muacRequiresConfirmation != null)
          'muac_requires_confirmation': m!.muacRequiresConfirmation.toString(),
        if (m?.muacReferralGuidance != null)
          'muac_referral_guidance': m!.muacReferralGuidance!,
        if (m?.combinedStatus != null) 'combined_status': m!.combinedStatus!,
        if (m?.combinedTriggeredBy != null)
          'combined_triggered_by': m!.combinedTriggeredBy!,
        if (m?.combinedRationale != null)
          'combined_rationale': m!.combinedRationale!,
        if (m?.combinedMethod != null) 'combined_method': m!.combinedMethod!,
        if (m?.combinedConfidenceScore != null)
          'combined_confidence_score': m!.combinedConfidenceScore.toString(),
        if (m?.combinedProtocolVersion != null)
          'combined_protocol_version': m!.combinedProtocolVersion!,
        if (child.guardianName != null) 'guardian_name': child.guardianName!,
        if (child.location != null) 'location': child.location!,
      });

      final imagePath = pair.visit.imagePath;
      if (imagePath != null && await File(imagePath).exists()) {
        req.files.add(await http.MultipartFile.fromPath('image', imagePath));
      }
      if (pair.visit.sideImagePath != null &&
          await File(pair.visit.sideImagePath!).exists()) {
        req.files.add(await http.MultipartFile.fromPath(
            'image_side', pair.visit.sideImagePath!));
      }
      if (pair.visit.backImagePath != null &&
          await File(pair.visit.backImagePath!).exists()) {
        req.files.add(await http.MultipartFile.fromPath(
            'image_back', pair.visit.backImagePath!));
      }
      if (child.photoPath != null && await File(child.photoPath!).exists()) {
        req.files
            .add(await http.MultipartFile.fromPath('photo', child.photoPath!));
      }

      final streamed =
          await _client.send(req).timeout(const Duration(seconds: 60));
      final response = await http.Response.fromStream(streamed);

      if (response.statusCode == 200) {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        await _syncDao.markSynced(entry.id,
            serverVisitId: body['server_visit_id'] as int?);
      } else if (response.statusCode == 401) {
        _onUnauthorized?.call();
        await _syncDao.markFailed(
            entry.id, 'Unauthorized (401) — re-login required');
      } else {
        await _syncDao.markFailed(
            entry.id, 'HTTP ${response.statusCode}: ${response.body}');
      }
    } catch (e) {
      await _syncDao.markFailed(entry.id, e.toString());
    }
  }
}
