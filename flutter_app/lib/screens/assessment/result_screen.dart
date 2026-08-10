import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../constants/config.dart' show wastingStatusLabel;
import '../../features/ar_scan/widgets/ar_scan_card.dart';
import '../../l10n/l10n_provider.dart';
import '../../models/assessment_result.dart';
import '../../models/who_reference_targets.dart';
import '../../providers/assessment_provider.dart';
import '../shared/app_scaffold.dart';
import '../shared/status_badge.dart';

class ResultScreen extends ConsumerWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final result = ref.watch(assessmentResultProvider);

    if (result == null) {
      return AppScaffold(
        currentIndex: 0,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('No assessment result available.'),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () => context.go('/'),
                child: Text(t('run_assessment', ref)),
              ),
            ],
          ),
        ),
      );
    }

    return AppScaffold(
      currentIndex: 0,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _statusBanner(context, ref, result),
            if (_usesEstimatedEvidence(result)) ...[
              const SizedBox(height: 12),
              _estimateDisclosure(context, ref),
            ],
            if (result.visitUuid != null && result.ownerUserId != null) ...[
              const SizedBox(height: 12),
              ArScanCard(
                ownerUserId: result.ownerUserId!,
                visitUuid: result.visitUuid!,
              ),
            ],
            const SizedBox(height: 16),
            _photoSection(context, ref, result),
            if (_hasPhotoEstimates(result)) ...[
              const SizedBox(height: 16),
              _photoEstimateSection(context, ref, result),
              const SizedBox(height: 12),
              _screeningClassificationSection(context, ref, result),
            ],
            const SizedBox(height: 16),
            _metricCards(context, ref, result),
            if (!result.whoReferenceTargets.isEmpty) ...[
              const SizedBox(height: 16),
              _whoReferenceCard(context, ref, result),
            ],
            if (result.mlPrediction != null) ...[
              const SizedBox(height: 16),
              _mlSection(context, ref, result.mlPrediction!),
            ],
            if (result.muac?.requiresConfirmation == true) ...[
              const SizedBox(height: 12),
              _muacNote(context, ref, result.muac!),
            ],
            const SizedBox(height: 24),
            _actionButtons(context, ref),
          ],
        ),
      ),
    );
  }

  Widget _statusBanner(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final heightIsDirect = {
      'manual',
      'reference_object',
    }.contains(result.measurement.heightMethod);
    final haz = heightIsDirect ? result.nutrition.hazStatus : null;

    final poshan = result.poshan.finalStatus;

    String title;
    String message;
    Color color;

    if (poshan == 'SAM') {
      title = t('banner_sam_title', ref);
      message = t('banner_sam_msg', ref);
      color = Colors.red;
    } else if (poshan == 'MAM') {
      title = t('banner_mam_title', ref);
      message = t('banner_mam_msg', ref);
      color = Colors.orange;
    } else if (haz != null && haz.toLowerCase().contains('stunted')) {
      title = haz;
      message = t('banner_stunted_msg', ref);
      color = Colors.amber.shade700;
    } else if (poshan == 'Normal') {
      title = t('banner_normal_title', ref);
      message = t('banner_normal_msg', ref);
      color = Colors.green;
    } else if (_hasPhotoEstimates(result)) {
      title = t('banner_estimates_ready_title', ref);
      message = t('banner_estimates_ready_msg', ref);
      color = Colors.teal;
    } else {
      title = t('banner_unknown_title', ref);
      message = t('banner_unknown_msg', ref);
      color = Colors.grey;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        border: Border(left: BorderSide(color: color, width: 4)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${result.childName} — ${result.ageMonths.toStringAsFixed(1)} ${t('months_unit', ref)}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 4),
          Text(
            title,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
              color: color,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(message),
          const SizedBox(height: 6),
          Text(
            'Poshan Setu status: $poshan',
            style: const TextStyle(fontWeight: FontWeight.bold),
          ),
          if (result.poshan.rationale.isNotEmpty) Text(result.poshan.rationale),
          if (result.mlPrediction == null) ...[
            const SizedBox(height: 6),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
              decoration: BoxDecoration(
                color: Colors.amber.shade100,
                borderRadius: BorderRadius.circular(4),
              ),
              child: Text(
                t('fallback_used', ref),
                style: const TextStyle(fontSize: 11),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _photoSection(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final estimationMethod = result.measurement.estimationMethod;
    final confidence = result.measurement.confidenceScore;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (confidence != null) ...[
              Row(
                children: [
                  Expanded(child: Text('${t('pose_confidence', ref)}:')),
                  const SizedBox(width: 8),
                  Text('${(confidence * 100).toStringAsFixed(0)}%'),
                ],
              ),
              const SizedBox(height: 6),
              LinearProgressIndicator(
                value: confidence,
                backgroundColor: Colors.grey.shade200,
              ),
              const SizedBox(height: 4),
            ],
            if (estimationMethod != null)
              Text(
                estimationMethod == 'who_statistical'
                    ? t('analysis_pose_who', ref)
                    : 'Method: $estimationMethod',
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
        ),
      ),
    );
  }

  Widget _metricCards(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final heightIsDirect = {
      'manual',
      'reference_object',
    }.contains(result.measurement.heightMethod);
    final weightIsDirect = result.measurement.weightMethod == 'manual';
    final muacIsDirect = result.muac?.isDirectMeasurement == true;
    final measuredBmi = heightIsDirect && weightIsDirect
        ? result.poshan.bmi?.toStringAsFixed(2)
        : null;
    final hasAnyDirectMeasurement =
        heightIsDirect || weightIsDirect || muacIsDirect;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          t('child_measurements_title', ref),
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        Card(
          child: ListTile(
            title: const Text('Poshan Setu v1 classification'),
            subtitle: Text(
              '${t('measured_bmi_label', ref)}: '
              '${measuredBmi ?? t('not_assessed', ref)}'
              '${measuredBmi != null ? ' (${result.poshan.bmiStatus})' : ''}'
              ' · ${t('tape_muac_label', ref)}: '
              '${muacIsDirect ? result.poshan.muacStatus : t('not_assessed', ref)}'
              '\n${t('stunting_label', ref)}: '
              '${heightIsDirect ? result.nutrition.hazStatus ?? t('not_assessed', ref) : t('not_assessed', ref)}',
            ),
            trailing: StatusBadge(status: result.poshan.finalStatus),
          ),
        ),
        const SizedBox(height: 8),
        if (!hasAnyDirectMeasurement)
          Card(
            child: ListTile(
              leading: const Icon(Icons.straighten),
              title: Text(t('no_direct_measurements', ref)),
              subtitle: Text(t('direct_measurements_optional', ref)),
            ),
          ),
        if (heightIsDirect) ...[
          _metricCard(
            context,
            ref,
            title: t('metric_height', ref),
            value: result.measurement.effectiveHeightCm,
            unit: 'cm',
            source: _heightSource(ref, result.measurement),
            zscore: result.nutrition.hazZscore,
            status: result.nutrition.hazStatus,
          ),
        ],
        if (weightIsDirect) ...[
          if (heightIsDirect) const SizedBox(height: 8),
          _metricCard(
            context,
            ref,
            title: t('metric_weight', ref),
            value: result.measurement.manualWeightKg,
            unit: 'kg',
            source: _weightSource(ref, result.measurement),
            zscore: heightIsDirect ? result.nutrition.whzZscore : null,
            status: heightIsDirect ? result.nutrition.whzStatus : null,
            extras: _weightExtras(context, ref, result.measurement),
          ),
        ],
        if (muacIsDirect) ...[
          if (heightIsDirect || weightIsDirect) const SizedBox(height: 8),
          _muacCard(context, ref, result.muac),
        ],
      ],
    );
  }

  bool _usesEstimatedEvidence(AssessmentResult result) {
    final measurement = result.measurement;
    return measurement.heightMethod != 'manual' ||
        measurement.weightMethod != 'manual' ||
        result.muac?.isDirectMeasurement != true;
  }

  bool _hasPhotoEstimates(AssessmentResult result) {
    final measurement = result.measurement;
    final hasHeight =
        !{'manual', 'reference_object'}.contains(measurement.heightMethod) &&
        measurement.effectiveHeightCm != null;
    final hasWeight =
        measurement.weightMethod != 'manual' &&
        (measurement.effectiveWeightKg ?? measurement.predictedWeightKg) !=
            null;
    final hasMuac =
        result.muac?.isDirectMeasurement != true && result.muac?.muacCm != null;
    return hasHeight || hasWeight || hasMuac;
  }

  Widget _photoEstimateSection(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final measurement = result.measurement;
    final heightIsEstimated =
        !{'manual', 'reference_object'}.contains(measurement.heightMethod) &&
        measurement.effectiveHeightCm != null;
    final estimatedWeight =
        measurement.effectiveWeightKg ?? measurement.predictedWeightKg;
    final weightIsEstimated =
        measurement.weightMethod != 'manual' && estimatedWeight != null;
    final muac = result.muac;
    final muacIsEstimated =
        muac?.isDirectMeasurement != true && muac?.muacCm != null;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          t('app_estimates_title', ref),
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 2),
        Text(
          t('app_estimates_subtitle', ref),
          style: Theme.of(context).textTheme.bodySmall,
        ),
        const SizedBox(height: 8),
        if (heightIsEstimated) ...[
          _metricCard(
            context,
            ref,
            title: t('metric_height', ref),
            value: measurement.effectiveHeightCm,
            unit: 'cm',
            source: t('badge_pose_age_estimate', ref),
            zscore: null,
            status: null,
            extras: _confidenceText(
              context,
              ref,
              measurement.heightConfidence ?? measurement.confidenceScore,
            ),
          ),
          const SizedBox(height: 8),
        ],
        if (weightIsEstimated) ...[
          _metricCard(
            context,
            ref,
            title: t('metric_weight', ref),
            value: estimatedWeight,
            unit: 'kg',
            source: _weightSource(ref, measurement),
            zscore: null,
            status: null,
            extras: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (_confidenceText(context, ref, measurement.weightConfidence)
                    case final confidence?)
                  confidence,
                if (_weightExtras(context, ref, measurement)
                    case final dimensions?)
                  dimensions,
              ],
            ),
          ),
          const SizedBox(height: 8),
        ],
        if (muacIsEstimated)
          _metricCard(
            context,
            ref,
            title: t('metric_muac', ref),
            value: muac!.muacCm,
            unit: 'cm',
            source: _muacEstimateSource(ref, muac),
            zscore: null,
            status: null,
            extras: _muacEstimateExtras(context, ref, muac),
          ),
      ],
    );
  }

  Widget _estimateDisclosure(BuildContext context, WidgetRef ref) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        border: Border.all(color: Colors.amber.shade700),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            t('screening_estimates_title', ref),
            style: Theme.of(context).textTheme.titleSmall?.copyWith(
              color: Colors.amber.shade900,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(t('screening_estimates_body', ref)),
        ],
      ),
    );
  }

  Widget _screeningClassificationSection(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final mlStatus = _displayWastingStatus(
      ref,
      result.mlPrediction?.wastingStatus,
    );
    final whzStatus = _displayWastingStatus(ref, result.nutrition.whzStatus);
    final hazStatus = result.nutrition.hazStatus ?? t('not_assessed', ref);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.teal.withValues(alpha: 0.06),
        border: Border.all(color: Colors.teal.shade200),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            t('screening_classifications_title', ref),
            style: Theme.of(
              context,
            ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 2),
          Text(
            t('screening_classifications_subtitle', ref),
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 12),
          _classificationRow(
            context,
            label: t('sam_mam_screening_label', ref),
            status: mlStatus,
            evidence: _mlScreeningEvidence(ref, result.mlPrediction),
          ),
          const Divider(height: 20),
          _classificationRow(
            context,
            label: t('who_wasting_label', ref),
            status: whzStatus,
            evidence: _zScoreEvidence(ref, result.nutrition.whzZscore, 'WHZ'),
          ),
          const Divider(height: 20),
          _classificationRow(
            context,
            label: t('who_stunting_label', ref),
            status: hazStatus,
            evidence: _zScoreEvidence(ref, result.nutrition.hazZscore, 'HAZ'),
          ),
          const SizedBox(height: 12),
          Text(
            t('screening_classifications_disclaimer', ref),
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: Colors.teal.shade900,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _classificationRow(
    BuildContext context, {
    required String label,
    required String status,
    required String evidence,
  }) {
    return Semantics(
      label: '$label: $status',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 2),
          Text(evidence, style: Theme.of(context).textTheme.bodySmall),
          const SizedBox(height: 6),
          StatusBadge(status: status),
        ],
      ),
    );
  }

  String _displayWastingStatus(WidgetRef ref, String? status) {
    if (status == null ||
        status.trim().isEmpty ||
        status.toLowerCase() == 'who_fallback') {
      return t('not_assessed', ref);
    }
    return wastingStatusLabel(status);
  }

  String _mlScreeningEvidence(WidgetRef ref, MlPrediction? prediction) {
    if (prediction == null || prediction.wastingStatus == null) {
      return t('camera_screening_unavailable', ref);
    }
    final probability = switch (prediction.wastingStatus) {
      'SAM' => prediction.samProbability,
      'MAM' => prediction.mamProbability,
      'NORMAL' || 'Normal' => prediction.normalProbability,
      'RISK_OVERWEIGHT' || 'Risk_Overweight' => prediction.riskProbability,
      'OVERWEIGHT' || 'Overweight' => prediction.overweightProbability,
      _ => null,
    };
    if (probability == null) return t('camera_ml_screening', ref);
    return '${t('camera_ml_screening', ref)} · '
        '${(probability * 100).toStringAsFixed(0)}% '
        '${t('predicted_probability', ref)}';
  }

  String _zScoreEvidence(WidgetRef ref, double? zScore, String index) {
    if (zScore == null) return t('who_screening_unavailable', ref);
    return '${t('who_estimate_screening', ref)} · '
        '$index ${zScore.toStringAsFixed(2)}';
  }

  String _heightSource(WidgetRef ref, Measurement measurement) {
    switch (measurement.heightMethod) {
      case 'manual':
        return t('badge_manual', ref);
      case 'who_statistical':
      case 'who_median_estimated':
      case 'image_estimated':
        return t('badge_who_age_estimate', ref);
      case 'reference_object':
        return t('badge_image', ref);
      default:
        return t('badge_undetected', ref);
    }
  }

  String _weightSource(WidgetRef ref, Measurement measurement) {
    switch (measurement.weightMethod) {
      case 'manual':
        return t('badge_manual', ref);
      case 'ml_estimated':
        return t('badge_ml_estimate', ref);
      case 'who_statistical':
      case 'who_median_estimated':
        return t('badge_who_weight_estimate', ref);
      default:
        return t('badge_undetected', ref);
    }
  }

  String _muacEstimateSource(WidgetRef ref, MuacDetail muac) {
    return muac.muacMethod == 'landmark_estimated'
        ? t('badge_landmark_estimate', ref)
        : t('badge_whz_estimate', ref);
  }

  Widget? _confidenceText(
    BuildContext context,
    WidgetRef ref,
    double? confidence,
  ) {
    if (confidence == null) return null;
    return Text(
      '${t('estimate_confidence', ref)} '
      '${(confidence * 100).toStringAsFixed(0)}%',
      style: Theme.of(context).textTheme.bodySmall,
    );
  }

  Widget? _muacEstimateExtras(
    BuildContext context,
    WidgetRef ref,
    MuacDetail muac,
  ) {
    final lower = muac.uncertaintyLowerCm;
    final upper = muac.uncertaintyUpperCm;
    if (lower == null && upper == null && muac.confidence == null) return null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (lower != null && upper != null)
          Text(
            '${t('estimate_range', ref)} '
            '${lower.toStringAsFixed(1)}–${upper.toStringAsFixed(1)} cm',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        if (_confidenceText(context, ref, muac.confidence)
            case final confidence?)
          confidence,
      ],
    );
  }

  Widget? _weightExtras(BuildContext context, WidgetRef ref, Measurement m) {
    if (!m.sideViewUsed) return null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          t('badge_side_view_ok', ref),
          style: const TextStyle(fontSize: 12, color: Colors.teal),
        ),
        if (m.chestDepthCm != null)
          Text(
            '${t('chest_depth', ref)} ${m.chestDepthCm!.toStringAsFixed(1)} cm',
            style: const TextStyle(fontSize: 11),
          ),
        if (m.abdDepthCm != null)
          Text(
            '${t('abd_depth', ref)} ${m.abdDepthCm!.toStringAsFixed(1)} cm',
            style: const TextStyle(fontSize: 11),
          ),
      ],
    );
  }

  Widget _metricCard(
    BuildContext context,
    WidgetRef ref, {
    required String title,
    required double? value,
    required String unit,
    required String source,
    required double? zscore,
    required String? status,
    Widget? extras,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(
                        title,
                        style: Theme.of(context).textTheme.titleSmall,
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 6,
                          vertical: 1,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade200,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(
                          source,
                          style: const TextStyle(fontSize: 11),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value != null
                        ? '${value.toStringAsFixed(1)} $unit'
                        : t('not_measured', ref),
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  if (zscore != null)
                    Text(
                      'Z-score: ${zscore.toStringAsFixed(2)}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  if (extras != null) ...[const SizedBox(height: 4), extras],
                ],
              ),
            ),
            if (status != null) StatusBadge(status: status),
          ],
        ),
      ),
    );
  }

  Widget _muacCard(BuildContext context, WidgetRef ref, MuacDetail? muac) {
    if (muac == null) {
      return _metricCard(
        context,
        ref,
        title: t('metric_muac', ref),
        value: null,
        unit: 'cm',
        source: t('badge_na', ref),
        zscore: null,
        status: null,
      );
    }
    if (!muac.isDirectMeasurement) {
      return _metricCard(
        context,
        ref,
        title: t('metric_muac', ref),
        value: null,
        unit: 'cm',
        source: t('not_measured', ref),
        zscore: null,
        status: null,
      );
    }
    return _metricCard(
      context,
      ref,
      title: t('metric_muac', ref),
      value: muac.muacCm,
      unit: 'cm',
      source: t('badge_tape', ref),
      zscore: null,
      status: muac.muacStatus,
    );
  }

  Widget _mlSection(BuildContext context, WidgetRef ref, MlPrediction ml) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('ml_wasting_title', ref),
              style: Theme.of(context).textTheme.titleSmall,
            ),
            Text(
              t('ml_wasting_sub', ref),
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            _probabilityBar(
              t('sam_probability', ref),
              ml.samProbability,
              Colors.red,
            ),
            const SizedBox(height: 4),
            _probabilityBar(
              t('mam_probability', ref),
              ml.mamProbability,
              Colors.orange,
            ),
            const SizedBox(height: 4),
            _probabilityBar(
              t('normal_probability', ref),
              ml.normalProbability,
              Colors.green,
            ),
          ],
        ),
      ),
    );
  }

  Widget _whoReferenceCard(
    BuildContext context,
    WidgetRef ref,
    AssessmentResult result,
  ) {
    final targets = result.whoReferenceTargets;
    final sexLabel = result.sex.toUpperCase() == 'M'
        ? t('boy', ref)
        : t('girl', ref);

    return Card(
      color: Theme.of(
        context,
      ).colorScheme.primaryContainer.withValues(alpha: 0.35),
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('who_reference_title', ref),
              style: Theme.of(
                context,
              ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 2),
            Text(
              '$sexLabel · ${result.ageMonths.toStringAsFixed(1)} '
              '${t('months_unit', ref)}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 12),
            if (targets.heightForAge != null)
              _referenceRow(
                context,
                ref,
                label: result.ageMonths < 24
                    ? t('metric_length', ref)
                    : t('metric_height', ref),
                value: targets.heightForAge!,
                unit: 'cm',
              ),
            if (targets.weightForAge != null)
              _referenceRow(
                context,
                ref,
                label: t('metric_weight', ref),
                value: targets.weightForAge!,
                unit: 'kg',
              ),
            if (targets.muacForAge != null)
              _referenceRow(
                context,
                ref,
                label: t('metric_muac', ref),
                value: targets.muacForAge!,
                unit: 'cm',
              ),
            const Divider(height: 20),
            Text(
              t('who_reference_disclaimer', ref),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }

  Widget _referenceRow(
    BuildContext context,
    WidgetRef ref, {
    required String label,
    required WhoReferenceValue value,
    required String unit,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(label, style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 2),
          Text(
            '${value.target.toStringAsFixed(1)} $unit',
            style: Theme.of(
              context,
            ).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold),
          ),
          Text(
            '${t('who_reference_range', ref)} '
            '${value.lower2Sd.toStringAsFixed(1)}–'
            '${value.upper2Sd.toStringAsFixed(1)} $unit',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }

  Widget _probabilityBar(String label, double? value, Color color) {
    final pct = value ?? 0;
    return Row(
      children: [
        SizedBox(
          width: 120,
          child: Text(label, style: const TextStyle(fontSize: 12)),
        ),
        Expanded(
          child: LinearProgressIndicator(
            value: pct,
            backgroundColor: Colors.grey.shade200,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
        const SizedBox(width: 8),
        Text(
          '${(pct * 100).toStringAsFixed(0)}%',
          style: const TextStyle(fontSize: 12),
        ),
      ],
    );
  }

  Widget _muacNote(BuildContext context, WidgetRef ref, MuacDetail muac) {
    final noteKey = muac.muacMethod == 'landmark_estimated'
        ? 'muac_note_landmark_text'
        : 'muac_note_text';
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.amber.shade200),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, color: Colors.amber.shade800, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: DefaultTextStyle.of(
                  context,
                ).style.copyWith(fontSize: 13),
                children: [
                  TextSpan(
                    text: '${t('muac_note_strong', ref)} ',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  TextSpan(text: t(noteKey, ref)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionButtons(BuildContext context, WidgetRef ref) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        FilledButton.icon(
          onPressed: () {
            ref.read(assessmentResultProvider.notifier).state = null;
            context.go('/');
          },
          icon: const Icon(Icons.refresh),
          label: Text(t('assess_another', ref)),
        ),
        OutlinedButton.icon(
          onPressed: () => context.go('/children'),
          icon: const Icon(Icons.people),
          label: Text(t('view_all_children', ref)),
        ),
      ],
    );
  }
}
