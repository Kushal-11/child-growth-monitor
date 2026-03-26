import 'package:child_growth_monitor_app/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('app loads title', (tester) async {
    await tester.pumpWidget(const ChildGrowthApp());
    await tester.pumpAndSettle();

    expect(find.text('Child Growth Monitor'), findsOneWidget);
  });
}
