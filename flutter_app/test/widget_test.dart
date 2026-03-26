import 'package:child_growth_monitor_app/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('app loads title', (tester) async {
    await tester.pumpWidget(const ChildGrowthApp());

    expect(find.text('Child Growth Monitor'), findsOneWidget);
  });
}
