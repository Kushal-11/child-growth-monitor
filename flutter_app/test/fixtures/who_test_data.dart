import 'dart:io';

import 'package:child_growth_monitor_app/services/who_data_service.dart';

/// Loads bundled WHO fixtures for tests (rootBundle is unavailable in unit tests).
Future<void> loadWhoForTests(WhoDataService who) async {
  final base = Directory.current.path.endsWith('flutter_app')
      ? 'test/fixtures'
      : 'flutter_app/test/fixtures';
  await who.loadFromFiles(
    hazCsvPath: '$base/who_haz_0_59m.csv',
    wflBoysPath: '$base/who_wfl_boys_0_2.xlsx',
    wflGirlsPath: '$base/who_wfl_girls_0_2.xlsx',
    wfhBoysPath: '$base/who_wfh_boys_2_5.xlsx',
    wfhGirlsPath: '$base/who_wfh_girls_2_5.xlsx',
    lfaBoysPath: 'assets/who_data/who_lhfa_boys_0_2.xlsx',
    lfaGirlsPath: 'assets/who_data/who_lhfa_girls_0_2.xlsx',
    hfaBoysPath: 'assets/who_data/who_lhfa_boys_2_5.xlsx',
    hfaGirlsPath: 'assets/who_data/who_lhfa_girls_2_5.xlsx',
    wfaBoysPath: 'assets/who_data/who_wfa_boys_0_5.xlsx',
    wfaGirlsPath: 'assets/who_data/who_wfa_girls_0_5.xlsx',
    acfaBoysPath: 'assets/who_data/who_acfa_boys_3_5.xlsx',
    acfaGirlsPath: 'assets/who_data/who_acfa_girls_3_5.xlsx',
  );
}
