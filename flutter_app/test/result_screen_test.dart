import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';

import 'package:child_growth_monitor_app/database/database.dart'
    show AppDatabase;
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
        estimationMethod: 'manual',
        heightSource: 'manual',
        weightSource: 'manual',
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
      poshan: const PoshanDetail(
        bmi: 13.4,
        bmiStatus: 'MAM',
        muacStatus: 'SAM',
        finalStatus: 'SAM',
        triggeredBy: ['muac'],
        classificationMethod: 'poshan_setu_v1',
        rationale: 'Tape MUAC classified as SAM.',
        complete: true,
      ),
    );

void main() {
  testWidgets('banner shows SAM when MUAC is SAM even though WHZ is Normal',
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

  testWidgets('Indeterminate result visibly requires direct measurements',
      (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final result = AssessmentResult(
      childName: 'A',
      sex: 'F',
      ageMonths: 24,
      summary: 'Indeterminate',
      measurement: Measurement(
        effectiveHeightCm: 87,
        effectiveWeightKg: 11,
        heightSource: 'who_statistical',
        weightSource: 'ml_estimated',
      ),
      nutrition: Nutrition(ageMonths: 24),
      muac: MuacDetail(
        muacCm: 14,
        muacStatus: 'Indeterminate',
        muacMethod: 'whz_derived',
        ageInRange: true,
      ),
      poshan: const PoshanDetail(
        bmiStatus: 'Indeterminate',
        muacStatus: 'Indeterminate',
        finalStatus: 'Indeterminate',
        triggeredBy: [],
        classificationMethod: 'poshan_setu_v1',
        rationale: 'Direct measurements are unavailable.',
        complete: false,
      ),
    );
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      assessmentResultProvider.overrideWith((ref) => result),
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

    expect(find.text('Measurement Required — Indeterminate'), findsOneWidget);
    expect(find.text('WHO statistical'), findsWidgets);
    expect(find.text('ML estimate'), findsWidgets);
    expect(find.text('Image'), findsNothing);
    expect(find.text('Direct measurements required'), findsOneWidget);
    expect(
      find.text('WHO median fallback used (on-device ML unavailable)'),
      findsNothing,
    );
  });

  testWidgets('missing Poshan never promotes a legacy Normal summary',
      (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final result = AssessmentResult(
      childName: 'Legacy row',
      sex: 'M',
      ageMonths: 24,
      summary: 'Normal',
      measurement: Measurement(
        effectiveHeightCm: 87,
        effectiveWeightKg: 11,
        heightSource: 'who_statistical',
        weightSource: 'who_statistical',
      ),
      nutrition: Nutrition(
        hazStatus: 'Normal',
        whzStatus: 'Normal',
        ageMonths: 24,
      ),
    );
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      assessmentResultProvider.overrideWith((ref) => result),
    ]);
    addTearDown(() {
      container.dispose();
      db.close();
    });
    await tester.pumpWidget(
      UncontrolledProviderScope(
        container: container,
        child: const MaterialApp(home: ResultScreen()),
      ),
    );
    await tester.pumpAndSettle();

    expect(find.text('Measurement Required — Indeterminate'), findsOneWidget);
    expect(find.text('Normal Nutritional Status'), findsNothing);
    expect(
      find.text('WHO median fallback used (on-device ML unavailable)'),
      findsOneWidget,
    );
  });
}
