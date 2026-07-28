import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';

import 'package:child_growth_monitor_app/database/database.dart'
    show AppDatabase;
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/providers/assessment_provider.dart';
import 'package:child_growth_monitor_app/models/assessment_result.dart';
import 'package:child_growth_monitor_app/models/who_reference_targets.dart';
import 'package:child_growth_monitor_app/screens/assessment/result_screen.dart';

/// A tape-measured SAM child (MUAC < 11.5) whose WHZ reads Normal — the exact
/// false-negative the WHO OR-rule must catch on the headline banner.
AssessmentResult _samViaMuacOnly() => AssessmentResult(
      childName: 'Fatima',
      sex: 'F',
      ageMonths: 29,
      summary: 'SAM',
      combinedNutrition: const CombinedNutritionDetail(
          status: 'SAM',
          triggeredBy: ['muac'],
          rationale: 'SAM flagged by muac'),
      poshan: const PoshanDetail(
        bmiStatus: 'Indeterminate',
        muacStatus: 'SAM',
        finalStatus: 'SAM',
        triggeredBy: ['muac'],
        rationale: 'Eligible measured MUAC classified as SAM.',
      ),
      measurement: Measurement(
        effectiveHeightCm: 87.0,
        heightMethod: 'image_estimated',
        predictedHeightCm: 87.0,
        predictedWeightKg: 12.0,
        confidenceScore: 0.9,
        estimationMethod: 'image',
      ),
      nutrition: Nutrition(
        hazZscore: -0.5,
        whzZscore: -0.3,
        hazStatus: 'Normal',
        whzStatus: 'NORMAL',
        ageMonths: 29,
      ),
      muac: MuacDetail(
        muacCm: 10.0,
        muacStatus: 'SAM',
        muacMethod: 'manual',
        ageInRange: true,
      ),
    );

AssessmentResult _manualHeightWins() => AssessmentResult(
      childName: 'Asha',
      sex: 'F',
      ageMonths: 30,
      summary: 'Normal',
      combinedNutrition: const CombinedNutritionDetail(
        status: 'NORMAL',
        rationale: 'No wasting indicator triggered',
      ),
      measurement: Measurement(
        effectiveHeightCm: 80.0,
        heightMethod: 'manual',
        predictedHeightCm: 95.0,
        manualHeightCm: 80.0,
      ),
      nutrition: Nutrition(ageMonths: 30),
    );

AssessmentResult _estimatedOnly() => AssessmentResult(
      childName: 'Asha',
      sex: 'F',
      ageMonths: 46.8,
      summary: 'Indeterminate',
      combinedNutrition: const CombinedNutritionDetail(
        status: 'NORMAL',
        rationale: 'No direct MUAC or WHZ flag triggered',
      ),
      poshan: const PoshanDetail(
        bmiStatus: 'Indeterminate',
        muacStatus: 'Indeterminate',
        finalStatus: 'Indeterminate',
        rationale: 'Measured BMI and tape MUAC evidence is incomplete.',
      ),
      whoReferenceTargets: const WhoReferenceTargets(
        heightForAge: WhoReferenceValue(
          target: 102.1,
          lower2Sd: 94.5,
          upper2Sd: 109.7,
        ),
        weightForAge: WhoReferenceValue(
          target: 15.5,
          lower2Sd: 12.0,
          upper2Sd: 20.0,
        ),
        muacForAge: WhoReferenceValue(
          target: 15.8,
          lower2Sd: 13.5,
          upper2Sd: 18.6,
        ),
      ),
      measurement: Measurement(
        effectiveHeightCm: 102.1,
        heightMethod: 'who_statistical',
        predictedHeightCm: 102.1,
        predictedWeightKg: 15.15,
        effectiveWeightKg: 15.15,
        weightMethod: 'ml_estimated',
        confidenceScore: 0.995,
        estimationMethod: 'who_statistical',
      ),
      nutrition: Nutrition(
        hazZscore: 0,
        whzZscore: -0.5,
        hazStatus: 'Normal',
        whzStatus: 'NORMAL',
        ageMonths: 46.8,
      ),
      muac: MuacDetail(
        muacCm: 15,
        muacMethod: 'estimated_from_whz',
        ageInRange: true,
        requiresConfirmation: true,
      ),
    );

void main() {
  testWidgets('banner shows SAM when eligible Poshan MUAC is SAM',
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

  testWidgets('height card displays authoritative manual height and method',
      (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      assessmentResultProvider.overrideWith((ref) => _manualHeightWins()),
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

    expect(find.text('80.0 cm'), findsOneWidget);
    expect(find.text('95.0 cm'), findsNothing);
    expect(find.text('Manual'), findsWidgets);
  });

  testWidgets('estimated results disclose their actual provenance',
      (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
      assessmentResultProvider.overrideWith((ref) => _estimatedOnly()),
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

    expect(find.text('Screening estimates — not direct measurements'),
        findsOneWidget);
    expect(find.text('Not measured'), findsWidgets);
    expect(find.text('WHO reference targets'), findsOneWidget);
    expect(find.text('102.1 cm'), findsOneWidget);
    expect(find.text('15.5 kg'), findsOneWidget);
    expect(find.text('15.8 cm'), findsOneWidget);
    expect(find.text('15.2 kg'), findsNothing);
    expect(find.text('15.0 cm'), findsNothing);
    expect(find.text('WHO age estimate'), findsNothing);
    expect(find.text('ML estimate'), findsNothing);
    expect(find.text('Image'), findsNothing);
  });
}
