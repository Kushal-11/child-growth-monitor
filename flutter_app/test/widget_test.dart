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
  });
}
