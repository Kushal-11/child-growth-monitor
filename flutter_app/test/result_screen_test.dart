import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';

import 'package:child_growth_monitor_app/database/database.dart' show AppDatabase;
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/providers/assessment_provider.dart';
import 'package:child_growth_monitor_app/models/assessment_result.dart';
import 'package:child_growth_monitor_app/screens/assessment/result_screen.dart';

/// A tape-measured SAM child (MUAC < 11.5) whose WHZ reads Normal — the exact
/// false-negative the WHO OR-rule must catch on the headline banner.
AssessmentResult _samViaMuacOnly() => AssessmentResult(
      childName: 'Fatima',
      sex: 'F',
      ageMonths: 29,
      summary: 'SAM',
      measurement: Measurement(
        predictedHeightCm: 87.0,
        predictedWeightKg: 12.0,
        confidenceScore: 0.9,
        estimationMethod: 'image',
      ),
      nutrition: Nutrition(
        hazZscore: -0.5,
        whzZscore: -0.3,
        hazStatus: 'Normal',
        whzStatus: 'Normal',
        ageMonths: 29,
      ),
      muac: MuacDetail(
        muacCm: 10.0,
        muacStatus: 'SAM',
        muacMethod: 'manual',
        ageInRange: true,
      ),
    );

void main() {
  testWidgets(
      'banner shows SAM when MUAC is SAM even though WHZ is Normal',
      (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      assessmentResultProvider.overrideWith((ref) => _samViaMuacOnly()),
    ]);
    addTearDown(() {
      container.dispose();
      db.close();
    });

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: const MaterialApp(home: ResultScreen()),
    ));
    await tester.pumpAndSettle();

    // The red SAM banner title must be shown (banner_sam_title, en)...
    expect(find.text('Severe Acute Malnutrition (SAM)'), findsOneWidget);
    // ...and the green "Normal" banner must NOT be shown.
    expect(find.text('Normal Nutritional Status'), findsNothing);
  });
}
