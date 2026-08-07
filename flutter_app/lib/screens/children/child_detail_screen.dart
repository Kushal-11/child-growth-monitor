import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../constants/feature_flags.dart';
import '../../l10n/l10n_provider.dart';
import '../../models/child_detail.dart';
import '../../providers/children_provider.dart';
import '../shared/app_scaffold.dart';
import '../shared/status_badge.dart';

class ChildDetailScreen extends ConsumerWidget {
  const ChildDetailScreen({super.key, required this.childId});

  final int childId;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final detailAsync = ref.watch(childDetailProvider(childId));

    return AppScaffold(
      currentIndex: 1,
      child: detailAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(error.toString(),
                    style: const TextStyle(color: Colors.red)),
                const SizedBox(height: 8),
                OutlinedButton(
                  onPressed: () => ref.invalidate(childDetailProvider(childId)),
                  child: const Text('Retry'),
                ),
              ],
            ),
          ),
        ),
        data: (child) => SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _profileCard(context, ref, child),
              const SizedBox(height: 16),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: () => context.push('/children/$childId/edit'),
                      icon: const Icon(Icons.edit),
                      label: Text(t('edit_profile', ref)),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: FilledButton.icon(
                      onPressed: () =>
                          context.push('/children/$childId/measure'),
                      icon: const Icon(Icons.add_chart),
                      label: Text(t('add_measurement', ref)),
                    ),
                  ),
                ],
              ),
              if (FeatureFlags.liveCaptureEnabled) ...[
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.tonalIcon(
                    onPressed: () => context.push(
                      '/children/$childId/photo-assessment/consent',
                    ),
                    icon: const Icon(Icons.camera_alt_outlined),
                    label: const Text('New photo assessment'),
                  ),
                ),
              ],
              const SizedBox(height: 16),
              if (_hasChartData(child)) ...[
                _growthChart(context, ref, child),
                const SizedBox(height: 16),
              ],
              _visitHistory(context, ref, child),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: () => context.go('/children'),
                icon: const Icon(Icons.arrow_back),
                label: Text(t('back_to_children', ref)),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _profileCard(BuildContext context, WidgetRef ref, ChildDetail child) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              child.name,
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            _profileRow(t('label_dob', ref), child.dateOfBirth),
            _profileRow(
                t('label_sex', ref), child.sex == 'M' ? 'Male' : 'Female'),
            _profileRow(t('label_guardian', ref), child.guardianName ?? '—'),
            _profileRow(t('label_location', ref), child.location ?? '—'),
            _profileRow(t('total_visits', ref), child.visits.length.toString()),
          ],
        ),
      ),
    );
  }

  Widget _profileRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          SizedBox(
            width: 100,
            child: Text(label,
                style: const TextStyle(fontWeight: FontWeight.w500)),
          ),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }

  double? _eligibleHeight(ChildVisitMeasurement? measurement) {
    final method = measurement?.heightMethod;
    if (measurement == null ||
        (method != 'manual' && method != 'reference_object')) {
      return null;
    }
    return measurement.predictedHeightCm;
  }

  double? _eligibleWeight(ChildVisitMeasurement? measurement) {
    final method = measurement?.weightMethod;
    if (measurement == null ||
        (method != 'manual' && method != 'calibrated_scale')) {
      return null;
    }
    return measurement.predictedWeightKg;
  }

  bool _hasChartData(ChildDetail child) {
    int withData = 0;
    for (final v in child.visits) {
      if (_eligibleHeight(v.measurement) != null ||
          _eligibleWeight(v.measurement) != null) {
        withData++;
      }
    }
    return withData >= 2;
  }

  Widget _growthChart(BuildContext context, WidgetRef ref, ChildDetail child) {
    final visitsWithData = child.visits
        .where((v) =>
            _eligibleHeight(v.measurement) != null ||
            _eligibleWeight(v.measurement) != null)
        .toList()
      ..sort((a, b) => (a.ageMonths ?? 0).compareTo(b.ageMonths ?? 0));

    final heightSpots = <FlSpot>[];
    final weightSpots = <FlSpot>[];

    for (final v in visitsWithData) {
      final x = v.ageMonths ?? 0;
      final h = _eligibleHeight(v.measurement);
      final w = _eligibleWeight(v.measurement);
      if (h != null) heightSpots.add(FlSpot(x, h));
      if (w != null) weightSpots.add(FlSpot(x, w));
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('growth_chart_title', ref),
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 250,
              child: LineChart(
                LineChartData(
                  lineBarsData: [
                    if (heightSpots.isNotEmpty)
                      LineChartBarData(
                        spots: heightSpots,
                        isCurved: true,
                        color: Colors.blue,
                        barWidth: 2,
                        dotData: const FlDotData(show: true),
                        belowBarData: BarAreaData(show: false),
                      ),
                    if (weightSpots.isNotEmpty)
                      LineChartBarData(
                        spots: weightSpots,
                        isCurved: true,
                        color: Colors.orange,
                        barWidth: 2,
                        dotData: const FlDotData(show: true),
                        belowBarData: BarAreaData(show: false),
                      ),
                  ],
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      axisNameWidget: Text(t('age_months', ref),
                          style: const TextStyle(fontSize: 12)),
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 30,
                        getTitlesWidget: (value, meta) => Text(
                          value.toStringAsFixed(0),
                          style: const TextStyle(fontSize: 10),
                        ),
                      ),
                    ),
                    leftTitles: AxisTitles(
                      axisNameWidget: Text(
                        t('chart_height_cm', ref),
                        style:
                            const TextStyle(fontSize: 12, color: Colors.blue),
                      ),
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        getTitlesWidget: (value, meta) => Text(
                          value.toStringAsFixed(0),
                          style:
                              const TextStyle(fontSize: 10, color: Colors.blue),
                        ),
                      ),
                    ),
                    rightTitles: AxisTitles(
                      axisNameWidget: Text(
                        t('chart_weight_kg', ref),
                        style:
                            const TextStyle(fontSize: 12, color: Colors.orange),
                      ),
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        getTitlesWidget: (value, meta) => Text(
                          value.toStringAsFixed(0),
                          style: const TextStyle(
                              fontSize: 10, color: Colors.orange),
                        ),
                      ),
                    ),
                    topTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false)),
                  ),
                  gridData: const FlGridData(
                    show: true,
                    drawHorizontalLine: true,
                    drawVerticalLine: false,
                  ),
                  borderData: FlBorderData(show: true),
                  lineTouchData: LineTouchData(
                    touchTooltipData: LineTouchTooltipData(
                      getTooltipItems: (touchedSpots) {
                        return touchedSpots.map((spot) {
                          final isHeight =
                              spot.barIndex == 0 && heightSpots.isNotEmpty;
                          return LineTooltipItem(
                            '${spot.y.toStringAsFixed(1)} ${isHeight ? 'cm' : 'kg'}',
                            TextStyle(
                              color: isHeight ? Colors.blue : Colors.orange,
                              fontSize: 12,
                            ),
                          );
                        }).toList();
                      },
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _legendDot(Colors.blue, t('chart_height_cm', ref)),
                const SizedBox(width: 16),
                _legendDot(Colors.orange, t('chart_weight_kg', ref)),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 12)),
      ],
    );
  }

  Widget _visitHistory(BuildContext context, WidgetRef ref, ChildDetail child) {
    final visits = child.visits.reversed.toList();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('visit_history', ref),
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            if (visits.isEmpty)
              Text(t('no_visits_yet', ref))
            else
              ...visits.map((v) => _visitRow(context, ref, v)),
          ],
        ),
      ),
    );
  }

  Widget _visitRow(BuildContext context, WidgetRef ref, ChildVisit visit) {
    final m = visit.measurement;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                visit.visitDate ?? '—',
                style: const TextStyle(fontWeight: FontWeight.w500),
              ),
              const Spacer(),
              Text(
                '${visit.ageMonths?.toStringAsFixed(1) ?? '—'} ${t('months_unit', ref)}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
          const SizedBox(height: 2),
          if (visit.entryMethod == 'guided_capture' &&
              visit.captureState != null) ...[
            _guidedVisitStatus(context, visit),
            const SizedBox(height: 6),
          ],
          if (m != null)
            Wrap(
              spacing: 12,
              runSpacing: 4,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: [
                Text(
                  '${t('th_height_cm', ref)}'
                  '${_estimateSuffix(ref, m.heightMethod)}: '
                  '${m.predictedHeightCm?.toStringAsFixed(1) ?? '—'}',
                  style: const TextStyle(fontSize: 13),
                ),
                Text(
                  '${t('th_weight_kg', ref)}'
                  '${_estimateSuffix(ref, m.weightMethod)}: '
                  '${m.predictedWeightKg?.toStringAsFixed(1) ?? '—'}',
                  style: const TextStyle(fontSize: 13),
                ),
                if (m.muacCm != null)
                  Text(
                    '${t('metric_muac', ref)}'
                    '${_estimateSuffix(ref, m.muacMethod)}: '
                    '${m.muacCm!.toStringAsFixed(1)} cm',
                    style: const TextStyle(fontSize: 13),
                  ),
                if (m.hazStatus != null) ...[
                  StatusBadge(status: m.hazStatus),
                ],
                if (m.whzStatus != null) StatusBadge(status: m.whzStatus),
              ],
            )
          else
            Text(
              t('no_measurement_data', ref),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          if (visit.entryMethod == 'guided_capture' &&
              visit.localUuid != null) ...[
            const SizedBox(height: 6),
            _guidedVisitActions(context, visit),
          ],
          const Divider(height: 12),
        ],
      ),
    );
  }

  String _estimateSuffix(WidgetRef ref, String? method) {
    const directMethods = {'manual', 'reference_object'};
    if (method == null || directMethods.contains(method)) return '';
    return ' (${t('badge_est', ref)})';
  }

  Widget _guidedVisitStatus(BuildContext context, ChildVisit visit) {
    final label = switch (visit.captureState) {
      'draft_capture' || 'incomplete_capture' => 'Incomplete capture',
      'processing' => 'Processing estimate',
      'estimated_report' => 'Estimated report',
      'processing_failed' => 'Estimate failed — retry',
      'measured_report' => 'Measured report added',
      _ => null,
    };
    if (label == null) return const SizedBox.shrink();
    final camera = visit.cameraResultSummary;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.labelLarge,
        ),
        if (camera != null)
          Text(
            '${camera.modelVersion} · result v${camera.version}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        if (visit.mediaDeletedAt != null)
          Text(
            'Visit media deleted',
            style: Theme.of(context).textTheme.bodySmall,
          )
        else if (visit.requiredAssetAcknowledgement.isNotEmpty)
          Text(
            visit.requiredAssetsAcknowledged
                ? 'Required media acknowledged'
                : 'Required media pending acknowledgement',
            style: Theme.of(context).textTheme.bodySmall,
          ),
      ],
    );
  }

  Widget _guidedVisitActions(BuildContext context, ChildVisit visit) {
    final visitUuid = visit.localUuid!;
    final date = visit.visitDate?.split('T').first;
    final actions = <Widget>[
      if (visit.captureState == 'draft_capture' ||
          visit.captureState == 'incomplete_capture')
        TextButton(
          onPressed: () => context.push('/visits/$visitUuid/capture'),
          child: const Text('Resume capture'),
        ),
      if (visit.captureState == 'processing' ||
          visit.captureState == 'estimated_report' ||
          visit.captureState == 'measured_report')
        TextButton(
          onPressed: () => context.push('/visits/$visitUuid/report'),
          child: Text(
            visit.captureState == 'measured_report'
                ? 'View measured report'
                : 'View report',
          ),
        ),
      if (visit.captureState == 'processing_failed')
        TextButton(
          onPressed: () => context.push('/visits/$visitUuid/report'),
          child: const Text('Retry estimate'),
        ),
      if (visit.captureState == 'estimated_report' &&
          !visit.hasMeasuredReport &&
          date != null)
        TextButton(
          onPressed: () => context.push(
            '/visits/$visitUuid/measured-details?visitDate=$date',
          ),
          child: const Text('Add Measured Details'),
        ),
    ];
    return Wrap(spacing: 8, runSpacing: 4, children: actions);
  }
}
