import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:drift/native.dart';
import 'package:child_growth_monitor_app/database/database.dart';
import 'package:child_growth_monitor_app/providers/database_provider.dart';
import 'package:child_growth_monitor_app/screens/child_management/manual_measurement_screen.dart';

void main() {
  testWidgets('manual measurement requires height and weight', (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
    ]);
    addTearDown(() {
      container.dispose();
      db.close();
    });

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: const MaterialApp(home: ManualMeasurementScreen(childId: 1)),
    ));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const Key('measure_save')));
    await tester.tap(find.byKey(const Key('measure_save')));
    await tester.pumpAndSettle();
    expect(find.text('Required'), findsNWidgets(2)); // height + weight
  });

  testWidgets('save with missing child surfaces an inline error',
      (tester) async {
    final db = AppDatabase.forTesting(NativeDatabase.memory());
    final container = ProviderContainer(overrides: [
      databaseProvider.overrideWithValue(db),
    ]);
    addTearDown(() {
      container.dispose();
      db.close();
    });

    await tester.pumpWidget(UncontrolledProviderScope(
      container: container,
      child: const MaterialApp(home: ManualMeasurementScreen(childId: 999)),
    ));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(const Key('measure_height')), '80');
    await tester.enterText(find.byKey(const Key('measure_weight')), '10');
    await tester.ensureVisible(find.byKey(const Key('measure_save')));
    await tester.tap(find.byKey(const Key('measure_save')));
    await tester.pumpAndSettle();
    expect(find.textContaining('Could not save'), findsOneWidget);
  });
}
