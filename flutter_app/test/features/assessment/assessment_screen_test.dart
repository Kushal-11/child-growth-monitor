import 'package:child_growth_monitor_app/features/assessment/providers/assessment_form_provider.dart';
import 'package:child_growth_monitor_app/providers/sync_provider.dart';
import 'package:child_growth_monitor_app/screens/assessment/assessment_screen.dart';
import 'package:child_growth_monitor_app/theme/app_theme.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    SharedPreferences.setMockInitialValues(<String, Object>{});
  });

  testWidgets('shows the Figma-based child details step', (tester) async {
    await tester.binding.setSurfaceSize(const Size(390, 844));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    final container = _container();
    addTearDown(container.dispose);
    await _pumpScreen(tester, container);

    expect(find.text('SNEH Growth Monitor'), findsOneWidget);
    expect(find.byKey(const Key('assessment_step_1')), findsOneWidget);
    expect(find.text('Who are we assessing?'), findsOneWidget);
    expect(find.byKey(const Key('child_selector')), findsOneWidget);
    expect(find.byKey(const Key('assessment_next')), findsOneWidget);
  });

  testWidgets('validates child details and required front photo', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(390, 844));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    final container = _container();
    addTearDown(container.dispose);
    await _pumpScreen(tester, container);

    await tester.tap(find.byKey(const Key('assessment_next')));
    await tester.pump();
    expect(find.text('Required'), findsWidgets);
    expect(find.byKey(const Key('assessment_step_1')), findsOneWidget);

    await tester.enterText(
      find.byKey(const Key('assessment_child_name_0')),
      'Aarav',
    );
    _selectRequiredDemographics(container);
    await tester.tap(find.byKey(const Key('assessment_next')));
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('assessment_step_2')), findsOneWidget);
    expect(find.text('Capture photos'), findsOneWidget);

    await tester.tap(find.byKey(const Key('assessment_next')));
    await tester.pump();
    expect(find.text('Please select a front image.'), findsOneWidget);
    expect(find.byKey(const Key('assessment_step_2')), findsOneWidget);

    container
        .read(assessmentFormProvider.notifier)
        .setImage('front', 'assets/images/body_positioning_guide.png');
    await tester.enterText(find.byKey(const Key('assessment_weight')), '-1');
    await tester.tap(find.byKey(const Key('assessment_next')));
    await tester.pump();
    expect(find.textContaining('Enter a plausible value'), findsOneWidget);
    expect(find.byKey(const Key('assessment_step_2')), findsOneWidget);
  });

  testWidgets('moves from photos to review and supports back navigation', (
    tester,
  ) async {
    await tester.binding.setSurfaceSize(const Size(390, 844));
    addTearDown(() => tester.binding.setSurfaceSize(null));

    final container = _container();
    addTearDown(container.dispose);
    await _pumpScreen(tester, container);

    await tester.enterText(
      find.byKey(const Key('assessment_child_name_0')),
      'Meera',
    );
    _selectRequiredDemographics(container);
    await tester.tap(find.byKey(const Key('assessment_next')));
    await tester.pumpAndSettle();

    container
        .read(assessmentFormProvider.notifier)
        .setImage('front', 'assets/images/body_positioning_guide.png');
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('assessment_next')));
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('assessment_step_3')), findsOneWidget);
    expect(find.text('Review assessment'), findsOneWidget);
    expect(find.text('Meera'), findsOneWidget);
    expect(find.byKey(const Key('assessment_submit')), findsOneWidget);

    await tester.tap(find.byKey(const Key('assessment_back')));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('assessment_step_2')), findsOneWidget);
  });

  testWidgets('new assessment starts without fabricated DOB or sex',
      (tester) async {
    await tester.binding.setSurfaceSize(const Size(390, 844));
    addTearDown(() => tester.binding.setSurfaceSize(null));
    final container = _container();
    addTearDown(container.dispose);
    await _pumpScreen(tester, container);

    expect(container.read(assessmentFormProvider).dateOfBirth, isEmpty);
    expect(container.read(assessmentFormProvider).sex, isEmpty);
    expect(find.text('Select date of birth'), findsOneWidget);

    await tester.tap(find.text('Male'));
    await tester.pump();
    expect(container.read(assessmentFormProvider).sex, 'M');
    await tester.tap(find.text('Male'));
    await tester.pump();
    expect(container.read(assessmentFormProvider).sex, isEmpty);
    expect(tester.takeException(), isNull);
  });
}

void _selectRequiredDemographics(ProviderContainer container) {
  final date = DateTime.now().subtract(const Duration(days: 365));
  final dob = date.toIso8601String().substring(0, 10);
  final notifier = container.read(assessmentFormProvider.notifier);
  notifier.updateSex('M');
  notifier.updateDateOfBirth(dob);
}

ProviderContainer _container() {
  return ProviderContainer(
    overrides: [
      pendingSyncCountProvider.overrideWith((ref) => Stream.value(0)),
      assessmentChildrenProvider.overrideWith((ref) => Stream.value(const [])),
    ],
  );
}

Future<void> _pumpScreen(
  WidgetTester tester,
  ProviderContainer container,
) async {
  await tester.pumpWidget(
    UncontrolledProviderScope(
      container: container,
      child: MaterialApp(
        theme: AppTheme.light(),
        home: const AssessmentScreen(),
      ),
    ),
  );
  await tester.pumpAndSettle();
}
