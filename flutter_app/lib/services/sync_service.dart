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
    int? ownerUserId,
    void Function()? onUnauthorized,
  })  : _db = db,
        _visitDao = visitDao,
        _childDao = childDao,
        _syncDao = syncDao,
        _baseUrl = baseUrl,
        _authToken = authToken,
        _ownerUserId = ownerUserId,
        _onUnauthorized = onUnauthorized,
        _client = httpClient ?? http.Client();

  final AppDatabase _db;
  final VisitDao _visitDao;
  final ChildDao _childDao;
  final SyncQueueDao _syncDao;
  final String _baseUrl;
  final http.Client _client;
  final String? _authToken;
  final int? _ownerUserId;
  final void Function()? _onUnauthorized;

  static const _maxRetries = 5;

  Future<void> runOnce() async {
    // Never upload or mutate retry state without an authenticated owner.
    // Offline assessments remain queued and are drained after login/restore.
    final ownerUserId = _ownerUserId;
    if (_authToken == null || ownerUserId == null) return;
    await _recoverStuck(ownerUserId);

    final queueQuery = _db.select(_db.syncQueue).join([
      innerJoin(
        _db.visits,
        _db.visits.id.equalsExp(_db.syncQueue.visitId),
      ),
    ])
      ..where(
        (_db.syncQueue.status.equals('pending') |
                _db.syncQueue.status.equals('failed')) &
            _db.syncQueue.retryCount.isSmallerThanValue(_maxRetries),
      )
      ..orderBy([OrderingTerm.asc(_db.syncQueue.createdAt)]);
    queueQuery.where(_db.visits.ownerUserId.equals(ownerUserId));
    final entries = (await queueQuery.get())
        .map((row) => row.readTable(_db.syncQueue))
        .toList(growable: false);

    for (final entry in entries) {
      await _syncOne(entry, ownerUserId);
    }
  }

  /// Resets any entries stuck in 'syncing' state back to 'pending'.
  /// Defends against app crashes (SIGKILL, OOM, force-stop) that left
  /// entries mid-flight.
  Future<void> _recoverStuck(int ownerUserId) async {
    final ownedVisitIds = await (_db.selectOnly(_db.visits)
          ..addColumns([_db.visits.id])
          ..where(_db.visits.ownerUserId.equals(ownerUserId)))
        .map((row) => row.read(_db.visits.id)!)
        .get();
    if (ownedVisitIds.isEmpty) return;
    await (_db.update(_db.syncQueue)
          ..where((s) =>
              s.status.equals('syncing') & s.visitId.isIn(ownedVisitIds)))
        .write(const SyncQueueCompanion(status: Value('pending')));
  }

  Future<void> _syncOne(SyncQueueData entry, int ownerUserId) async {
    await _syncDao.markSyncing(entry.id);
    try {
      final pair = await _visitDao.getById(
        entry.visitId,
        ownerUserId: ownerUserId,
      );
      if (pair == null) {
        await _syncDao.markFailed(
          entry.id,
          'Visit not found for the signed-in user',
        );
        return;
      }
      final child = await _childDao.getById(
        pair.visit.childId,
        ownerUserId: ownerUserId,
      );
      if (child == null) {
        await _syncDao.markFailed(
          entry.id,
          'Child not found for the signed-in user',
        );
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
        if (m?.bodyBuild != null) 'body_build': m!.bodyBuild!,
        'side_view_used': (m?.sideViewUsed ?? false).toString(),
        if (m?.chestDepthCm != null)
          'chest_depth_cm': m!.chestDepthCm.toString(),
        if (m?.abdDepthCm != null) 'abd_depth_cm': m!.abdDepthCm.toString(),
        if (m?.mlEstimatedWeightKg != null)
          'ml_estimated_weight_kg': m!.mlEstimatedWeightKg.toString(),
        if (m?.wastingStatus != null) 'ml_wasting_status': m!.wastingStatus!,
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
        if (m?.effectiveHeightCm != null)
          'effective_height_cm': m!.effectiveHeightCm.toString(),
        if (m?.effectiveWeightKg != null)
          'effective_weight_kg': m!.effectiveWeightKg.toString(),
        if (m?.heightSource != null) 'height_source': m!.heightSource!,
        if (m?.weightSource != null) 'weight_source': m!.weightSource!,
        if (m?.bmi != null) 'bmi': m!.bmi.toString(),
        if (m?.bmiStatus != null) 'bmi_status': m!.bmiStatus!,
        if (m?.poshanStatus != null) 'poshan_status': m!.poshanStatus!,
        if (m?.poshanTriggeredBy != null)
          'poshan_triggered_by': m!.poshanTriggeredBy!,
        if (m?.classificationMethod != null)
          'classification_method': m!.classificationMethod!,
        if (m?.classificationRationale != null)
          'classification_rationale': m!.classificationRationale!,
        if (m?.mlModelVersion != null) 'ml_model_version': m!.mlModelVersion!,
        if (m?.mlNonClinical != null)
          'ml_non_clinical': m!.mlNonClinical.toString(),
        if (m?.mlTrainingData != null) 'ml_training_data': m!.mlTrainingData!,
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
