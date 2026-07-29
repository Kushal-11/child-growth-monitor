import 'package:flutter/material.dart';
import 'package:intl/intl.dart';

import '../providers/visit_report_provider.dart';
import 'report_metric_card.dart';

class MeasuredReportView extends StatelessWidget {
  const MeasuredReportView({
    super.key,
    required this.report,
    required this.visitDate,
    required this.onEditMeasuredDetails,
  });

  final MeasuredReportSnapshot report;
  final DateTime visitDate;
  final VoidCallback onEditMeasuredDetails;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Measurement-based Growth Report',
            style: Theme.of(context).textTheme.headlineSmall,
          ),
          const SizedBox(height: 4),
          Text('Visit date: ${DateFormat('dd MMM yyyy').format(visitDate)}'),
          const SizedBox(height: 12),
          ReportMetricCard(
            label: 'Measured height or length',
            value: _measurement(report.heightCm, 'cm'),
            detail: report.heightCm == null ? null : 'Manual measurement',
            icon: Icons.height,
          ),
          ReportMetricCard(
            label: 'Measured weight',
            value: _measurement(report.weightKg, 'kg'),
            detail: report.weightKg == null ? null : 'Manual measurement',
            icon: Icons.monitor_weight_outlined,
          ),
          ReportMetricCard(
            label: 'Tape MUAC',
            value: _measurement(report.muacCm, 'cm'),
            detail: report.muacCm == null
                ? null
                : report.muacEligible == false
                    ? 'Stored; not classification-eligible for this age'
                    : 'Direct tape measurement',
            icon: Icons.straighten,
          ),
          const SizedBox(height: 8),
          _ReportSection(
            title: 'WHO HAZ stunting',
            value: _displayStatus(report.hazStatus),
            detail: report.hazZscore == null
                ? 'Height or length has not been measured.'
                : 'HAZ ${report.hazZscore!.toStringAsFixed(2)}',
          ),
          _ReportSection(
            title: 'WHO acute malnutrition',
            value: _displayStatus(report.whoAcuteStatus),
            detail: _acuteDetail(report),
          ),
          _ReportSection(
            title: 'Poshan Setu v1',
            value: _displayStatus(
              report.poshanComplete == true ? report.poshanStatus : null,
            ),
            detail: report.poshanComplete == true
                ? _triggerDetail(report.poshanTriggeredBy)
                : 'Not enough eligible measured inputs.',
          ),
          const SizedBox(height: 12),
          OutlinedButton.icon(
            onPressed: onEditMeasuredDetails,
            icon: const Icon(Icons.edit_outlined),
            label: const Text('Update Measured Details'),
          ),
        ],
      ),
    );
  }

  static String _measurement(double? value, String unit) =>
      value == null ? 'Not measured' : '${value.toStringAsFixed(1)} $unit';

  static String _displayStatus(String? value) {
    if (value == null ||
        value.trim().isEmpty ||
        value.toUpperCase() == 'UNKNOWN' ||
        value.toLowerCase() == 'indeterminate') {
      return 'Not measured';
    }
    return value;
  }

  static String _acuteDetail(MeasuredReportSnapshot report) {
    if (_displayStatus(report.whoAcuteStatus) == 'Not measured') {
      return 'Eligible measured WHZ, tape MUAC, or oedema is unavailable.';
    }
    return _triggerDetail(report.whoAcuteTriggeredBy);
  }

  static String _triggerDetail(List<String> triggers) => triggers.isEmpty
      ? 'Calculated from eligible measured components.'
      : 'Triggered by ${triggers.join(', ')}.';
}

class _ReportSection extends StatelessWidget {
  const _ReportSection({
    required this.title,
    required this.value,
    required this.detail,
  });

  final String title;
  final String value;
  final String detail;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(14),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(title, style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 6),
            Text(value, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(height: 4),
            Text(detail),
          ],
        ),
      ),
    );
  }
}
