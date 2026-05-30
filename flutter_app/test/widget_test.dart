import 'package:child_growth_monitor_app/screens/shared/status_badge.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

/// Pumps [StatusBadge] inside a minimal MaterialApp and returns the
/// background colour of its decorated container.
Future<Color?> _badgeColor(WidgetTester tester, String? status) async {
  await tester.pumpWidget(
    MaterialApp(home: Scaffold(body: StatusBadge(status: status))),
  );
  final container = tester.widget<Container>(
    find.descendant(
      of: find.byType(StatusBadge),
      matching: find.byType(Container),
    ),
  );
  return (container.decoration as BoxDecoration).color;
}

void main() {
  group('StatusBadge', () {
    testWidgets('renders the status label', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(home: Scaffold(body: StatusBadge(status: 'SAM'))),
      );
      expect(find.text('SAM'), findsOneWidget);
    });

    testWidgets('falls back to "Unknown" when status is null', (tester) async {
      await tester.pumpWidget(
        const MaterialApp(home: Scaffold(body: StatusBadge(status: null))),
      );
      expect(find.text('Unknown'), findsOneWidget);
    });

    // Colour mapping is safety-relevant: SAM/MAM must never read as Normal.
    testWidgets('SAM is red', (tester) async {
      expect(await _badgeColor(tester, 'SAM'), Colors.red);
    });

    testWidgets('MAM is orange', (tester) async {
      expect(await _badgeColor(tester, 'MAM'), Colors.orange);
    });

    testWidgets('Normal is green', (tester) async {
      expect(await _badgeColor(tester, 'Normal'), Colors.green);
    });

    testWidgets('unknown status is grey', (tester) async {
      expect(await _badgeColor(tester, 'something-else'), Colors.grey);
    });

    // Regression guard: the WHZ/HAZ classifiers emit FULL human-readable
    // strings, not the short 'SAM'/'MAM' tokens. Exact-match colouring silently
    // rendered these as grey (looked identical to 'Unknown') on the result
    // weight card and visit-history rows. These assertions feed the literal
    // classifyWhz/classifyHaz outputs to lock the danger colours in place.
    testWidgets('full SAM string is red', (tester) async {
      expect(
        await _badgeColor(tester, 'Severe Acute Malnutrition (SAM)'),
        Colors.red,
      );
    });

    testWidgets('full MAM string is orange', (tester) async {
      expect(
        await _badgeColor(tester, 'Moderate Acute Malnutrition (MAM)'),
        Colors.orange,
      );
    });

    testWidgets('Severely Stunted is orange (not amber/grey)', (tester) async {
      expect(await _badgeColor(tester, 'Severely Stunted'), Colors.orange);
    });

    testWidgets('Stunted is amber', (tester) async {
      expect(await _badgeColor(tester, 'Stunted'), Colors.amber.shade700);
    });

    // 'Possible Risk of Overweight' contains 'overweight' and must NOT be
    // mistaken for a wasting alert — it maps to the over-nutrition colour.
    testWidgets('Possible Risk of Overweight is purple', (tester) async {
      expect(
        await _badgeColor(tester, 'Possible Risk of Overweight'),
        Colors.purple,
      );
    });
  });
}
