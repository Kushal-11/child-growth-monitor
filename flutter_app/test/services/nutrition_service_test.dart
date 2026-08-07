import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

import '../fixtures/who_test_data.dart';

void main() {
  late WhoDataService who;
  late NutritionService svc;

  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    who = WhoDataService();
    await loadWhoForTests(who);
    svc = NutritionService(who);
  });

  test('computeHaz returns z near 0 for median height', () {
    final median = who.getMedianHeightForAge('M', 24);
    final z = svc.computeHaz('M', 24, median!);
    expect(z, isNotNull);
    expect(z!, closeTo(0.0, 0.1));
  });

  test('computeHaz returns -2 for z=-2 boundary height', () {
    final boundaries = who.getHazBoundaries('M', 24);
    final z = svc.computeHaz('M', 24, boundaries![-2]!);
    expect(z!, closeTo(-2.0, 0.1));
  });

  test('computeWhz returns z for known weight/height', () {
    final z = svc.computeWhz('M', 24.0, 87.0, 12.0);
    expect(z, isNotNull);
    expect(z!, closeTo(0.0, 1.0));
  });

  test('computeWaz returns z near 0 for WHO median weight', () {
    final median = who.getReferenceTargets('F', 24).weightForAge!.target;
    final z = svc.computeWaz('F', 24, median);
    expect(z, isNotNull);
    expect(z!, closeTo(0, 0.01));
  });

  test('computeBaz returns z near 0 for WHO median BMI', () {
    final lms = who.getBfaLms('M', 24);
    expect(lms, isNotNull);
    final z = svc.computeBaz('M', 24, lms!.$2);
    expect(z, isNotNull);
    expect(z!, closeTo(0, 0.01));
  });
}
