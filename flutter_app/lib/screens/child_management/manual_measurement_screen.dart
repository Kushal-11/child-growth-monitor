import 'package:flutter/material.dart';

class ManualMeasurementScreen extends StatelessWidget {
  const ManualMeasurementScreen({super.key, required this.childId});
  final int childId;

  @override
  Widget build(BuildContext context) =>
      const Scaffold(body: Center(child: Text('Manual measurement (stub)')));
}
