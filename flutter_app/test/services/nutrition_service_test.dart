import 'package:flutter_test/flutter_test.dart';
import 'package:child_growth_monitor_app/services/nutrition_service.dart';
import 'package:child_growth_monitor_app/services/who_data_service.dart';

void main() {
  late WhoDataService who;
  late NutritionService svc;

  setUpAll(() async {
    TestWidgetsFlutterBinding.ensureInitialized();
    who = WhoDataService();
    await who.loadFromFiles(
      hazCsvPath: 'test/fixtures/who_haz_0_59m.csv',
      wflBoysPath: 'test/fixtures/who_wfl_boys_0_2.xlsx',
      wflGirlsPath: 'test/fixtures/who_wfl_girls_0_2.xlsx',
      wfhBoysPath: 'test/fixtures/who_wfh_boys_2_5.xlsx',
      wfhGirlsPath: 'test/fixtures/who_wfh_girls_2_5.xlsx',
    );
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
}
