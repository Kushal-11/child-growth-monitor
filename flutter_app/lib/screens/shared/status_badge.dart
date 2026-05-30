import 'package:flutter/material.dart';

class StatusBadge extends StatelessWidget {
  const StatusBadge({super.key, required this.status});

  final String? status;

  @override
  Widget build(BuildContext context) {
    final label = status ?? 'Unknown';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: _color(label),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: const TextStyle(color: Colors.white, fontSize: 12),
      ),
    );
  }

  /// Maps a status label to a triage colour using SUBSTRING matching, ordered
  /// most-severe-first. The classifiers in config.dart emit full human-readable
  /// strings ('Severe Acute Malnutrition (SAM)', 'Moderate Acute Malnutrition
  /// (MAM)', 'Severely Stunted', 'Possible Risk of Overweight'), while MUAC uses
  /// the short forms ('SAM', 'At Risk (MAM)'). Exact-match switching silently
  /// dropped the full danger strings to grey — a safety-critical defect on the
  /// result weight card and visit-history rows. Substring matching keeps both
  /// forms red/orange. Order is load-bearing: check SAM before MAM, 'severely
  /// stunted' before 'stunted', and the malnutrition cases before 'overweight'
  /// (so 'Possible Risk of Overweight' never reads as a wasting alert).
  static Color _color(String status) {
    final s = status.toLowerCase();

    // Severe acute malnutrition — most urgent.
    if (s.contains('sam') || s.contains('severe acute')) return Colors.red;
    // Moderate acute malnutrition / at-risk.
    if (s.contains('mam') ||
        s.contains('moderate acute') ||
        s.contains('at risk')) {
      return Colors.orange;
    }
    // Stunting — check the severe form first.
    if (s.contains('severely stunted') || s.contains('sev. stunted')) {
      return Colors.orange;
    }
    if (s.contains('stunted')) return Colors.amber.shade700;
    // Over-nutrition spectrum (incl. 'Possible Risk of Overweight').
    if (s.contains('overweight') || s.contains('obese')) return Colors.purple;
    // Healthy.
    if (s.contains('normal')) return Colors.green;

    return Colors.grey;
  }
}
