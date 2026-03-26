import 'package:flutter/material.dart';

import 'screens/assessment_screen.dart';

void main() {
  runApp(const ChildGrowthApp());
}

class ChildGrowthApp extends StatelessWidget {
  const ChildGrowthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Child Growth Monitor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      home: const AssessmentScreen(),
    );
  }
}
