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

  static Color _color(String status) {
    switch (status.toLowerCase()) {
      case 'normal':
        return Colors.green;
      case 'stunted':
        return Colors.amber.shade700;
      case 'severely stunted':
      case 'sev. stunted':
        return Colors.orange;
      case 'mam':
      case 'at risk':
      case 'at risk (mam)':
        return Colors.orange;
      case 'sam':
      case 'severe':
        return Colors.red;
      case 'overweight':
      case 'obese':
        return Colors.purple;
      default:
        return Colors.grey;
    }
  }
}
