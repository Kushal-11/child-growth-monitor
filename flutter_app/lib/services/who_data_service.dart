/// Unified WHO reference data loader for offline growth assessment.
///
/// Loads official WHO Excel growth-standard files into in-memory tables.
/// Provides lookup methods for Z-score computation that match the Python
/// backend exactly.
///
/// Data source priority:
///   - Excel files for length/height-for-age, weight-for-age and
///     arm-circumference-for-age reference targets
///   - Excel files (L-M-S parameters) for WFH/WFL (0-2y and 2-5y)
///   - Existing CSV file for the legacy HAZ calculation path
library;

import 'dart:io';
import 'dart:math';

import 'package:csv/csv.dart';
import 'package:excel/excel.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../models/who_reference_targets.dart';

/// A single HAZ row from the CSV file.
class _HazRow {
  final String sex;
  final String measure;
  final int ageMonths;
  final Map<int, double> zBoundaries; // keys: -3..-1, 0, 1..3

  const _HazRow({
    required this.sex,
    required this.measure,
    required this.ageMonths,
    required this.zBoundaries,
  });
}

/// A single WFL/WFH LMS row from an Excel file.
class _LmsRow {
  final double indexValue; // height or length in cm
  final double l;
  final double m;
  final double s;

  const _LmsRow({
    required this.indexValue,
    required this.l,
    required this.m,
    required this.s,
  });
}

class WhoDataService {
  List<_HazRow> _hazData = [];

  /// WFL LMS rows keyed by sex ('M' or 'F'), sorted by indexValue.
  Map<String, List<_LmsRow>> _wflLms = {};

  /// WFH LMS rows keyed by sex ('M' or 'F'), sorted by indexValue.
  Map<String, List<_LmsRow>> _wfhLms = {};

  /// Length-for-age LMS rows keyed by sex, indexed by age in months.
  Map<String, List<_LmsRow>> _lfaLms = {};

  /// Height-for-age LMS rows keyed by sex, indexed by age in months.
  Map<String, List<_LmsRow>> _hfaLms = {};

  /// Weight-for-age LMS rows keyed by sex, indexed by age in months.
  Map<String, List<_LmsRow>> _wfaLms = {};

  /// Arm-circumference-for-age LMS rows keyed by sex, indexed by age in months.
  Map<String, List<_LmsRow>> _acfaLms = {};

  bool _loaded = false;

  bool get isLoaded => _loaded;

  // ---------------------------------------------------------------------------
  // Loading: from Flutter assets (production)
  // ---------------------------------------------------------------------------

  /// Load all WHO data from bundled Flutter assets.
  Future<void> loadFromAssets() async {
    final hazCsv =
        await rootBundle.loadString('assets/who_data/who_haz_0_59m.csv');
    _parseHazCsv(hazCsv);

    final wflBoys =
        await rootBundle.load('assets/who_data/who_wfl_boys_0_2.xlsx');
    final wflGirls =
        await rootBundle.load('assets/who_data/who_wfl_girls_0_2.xlsx');
    final wfhBoys =
        await rootBundle.load('assets/who_data/who_wfh_boys_2_5.xlsx');
    final wfhGirls =
        await rootBundle.load('assets/who_data/who_wfh_girls_2_5.xlsx');
    final lfaBoys =
        await rootBundle.load('assets/who_data/who_lhfa_boys_0_2.xlsx');
    final lfaGirls =
        await rootBundle.load('assets/who_data/who_lhfa_girls_0_2.xlsx');
    final hfaBoys =
        await rootBundle.load('assets/who_data/who_lhfa_boys_2_5.xlsx');
    final hfaGirls =
        await rootBundle.load('assets/who_data/who_lhfa_girls_2_5.xlsx');
    final wfaBoys =
        await rootBundle.load('assets/who_data/who_wfa_boys_0_5.xlsx');
    final wfaGirls =
        await rootBundle.load('assets/who_data/who_wfa_girls_0_5.xlsx');
    final acfaBoys =
        await rootBundle.load('assets/who_data/who_acfa_boys_3_5.xlsx');
    final acfaGirls =
        await rootBundle.load('assets/who_data/who_acfa_girls_3_5.xlsx');

    _wflLms = {
      'M': _parseExcelLms(wflBoys.buffer.asUint8List()),
      'F': _parseExcelLms(wflGirls.buffer.asUint8List()),
    };
    _wfhLms = {
      'M': _parseExcelLms(wfhBoys.buffer.asUint8List()),
      'F': _parseExcelLms(wfhGirls.buffer.asUint8List()),
    };
    _lfaLms = {
      'M': _parseExcelLms(lfaBoys.buffer.asUint8List()),
      'F': _parseExcelLms(lfaGirls.buffer.asUint8List()),
    };
    _hfaLms = {
      'M': _parseExcelLms(hfaBoys.buffer.asUint8List()),
      'F': _parseExcelLms(hfaGirls.buffer.asUint8List()),
    };
    _wfaLms = {
      'M': _parseExcelLms(wfaBoys.buffer.asUint8List()),
      'F': _parseExcelLms(wfaGirls.buffer.asUint8List()),
    };
    _acfaLms = {
      'M': _parseExcelLms(acfaBoys.buffer.asUint8List()),
      'F': _parseExcelLms(acfaGirls.buffer.asUint8List()),
    };

    _loaded = true;
  }

  // ---------------------------------------------------------------------------
  // Loading: from file system (tests)
  // ---------------------------------------------------------------------------

  /// Load all WHO data from file paths — used in tests where rootBundle is
  /// unavailable.
  Future<void> loadFromFiles({
    required String hazCsvPath,
    required String wflBoysPath,
    required String wflGirlsPath,
    required String wfhBoysPath,
    required String wfhGirlsPath,
    String? lfaBoysPath,
    String? lfaGirlsPath,
    String? hfaBoysPath,
    String? hfaGirlsPath,
    String? wfaBoysPath,
    String? wfaGirlsPath,
    String? acfaBoysPath,
    String? acfaGirlsPath,
  }) async {
    final hazCsv = await File(hazCsvPath).readAsString();
    _parseHazCsv(hazCsv);

    _wflLms = {
      'M': _parseExcelLms(await File(wflBoysPath).readAsBytes()),
      'F': _parseExcelLms(await File(wflGirlsPath).readAsBytes()),
    };
    _wfhLms = {
      'M': _parseExcelLms(await File(wfhBoysPath).readAsBytes()),
      'F': _parseExcelLms(await File(wfhGirlsPath).readAsBytes()),
    };
    if (lfaBoysPath != null && lfaGirlsPath != null) {
      _lfaLms = {
        'M': _parseExcelLms(await File(lfaBoysPath).readAsBytes()),
        'F': _parseExcelLms(await File(lfaGirlsPath).readAsBytes()),
      };
    }
    if (hfaBoysPath != null && hfaGirlsPath != null) {
      _hfaLms = {
        'M': _parseExcelLms(await File(hfaBoysPath).readAsBytes()),
        'F': _parseExcelLms(await File(hfaGirlsPath).readAsBytes()),
      };
    }
    if (wfaBoysPath != null && wfaGirlsPath != null) {
      _wfaLms = {
        'M': _parseExcelLms(await File(wfaBoysPath).readAsBytes()),
        'F': _parseExcelLms(await File(wfaGirlsPath).readAsBytes()),
      };
    }
    if (acfaBoysPath != null && acfaGirlsPath != null) {
      _acfaLms = {
        'M': _parseExcelLms(await File(acfaBoysPath).readAsBytes()),
        'F': _parseExcelLms(await File(acfaGirlsPath).readAsBytes()),
      };
    }

    _loaded = true;
  }

  // ---------------------------------------------------------------------------
  // CSV parsing: HAZ
  // ---------------------------------------------------------------------------

  /// Parse the HAZ CSV string.
  ///
  /// Expected columns:
  /// sex, measure, age_months, z_minus_3, z_minus_2, z_minus_1, z_0,
  /// z_plus_1, z_plus_2, z_plus_3
  void _parseHazCsv(String csvString) {
    final rows = const CsvToListConverter(eol: '\n').convert(csvString);
    if (rows.isEmpty) return;

    // Skip header row.
    _hazData = [];
    for (var i = 1; i < rows.length; i++) {
      final row = rows[i];
      if (row.length < 10) continue;

      final sex = row[0].toString().trim();
      final measure = row[1].toString().trim();
      final ageMonths = _toInt(row[2]);
      if (ageMonths == null) continue;

      final zMinus3 = _toDouble(row[3]);
      final zMinus2 = _toDouble(row[4]);
      final zMinus1 = _toDouble(row[5]);
      final z0 = _toDouble(row[6]);
      final zPlus1 = _toDouble(row[7]);
      final zPlus2 = _toDouble(row[8]);
      final zPlus3 = _toDouble(row[9]);

      if (zMinus3 == null ||
          zMinus2 == null ||
          zMinus1 == null ||
          z0 == null ||
          zPlus1 == null ||
          zPlus2 == null ||
          zPlus3 == null) {
        continue;
      }

      _hazData.add(_HazRow(
        sex: sex,
        measure: measure,
        ageMonths: ageMonths,
        zBoundaries: {
          -3: zMinus3,
          -2: zMinus2,
          -1: zMinus1,
          0: z0,
          1: zPlus1,
          2: zPlus2,
          3: zPlus3,
        },
      ));
    }
  }

  // ---------------------------------------------------------------------------
  // Excel parsing: WFL / WFH LMS
  // ---------------------------------------------------------------------------

  /// Parse an Excel file containing LMS data.
  ///
  /// Expected columns (by position):
  ///   0: Length or Height (index value in cm)
  ///   1: L
  ///   2: M
  ///   3: S
  ///   4-10: SD3neg .. SD3 (ignored here — we use L, M, S directly)
  List<_LmsRow> _parseExcelLms(List<int> bytes) {
    final excel = Excel.decodeBytes(bytes);
    final sheetName = excel.tables.keys.first;
    final sheet = excel.tables[sheetName]!;

    final result = <_LmsRow>[];

    for (var i = 0; i < sheet.rows.length; i++) {
      final row = sheet.rows[i];
      if (i == 0) continue; // skip header

      final indexVal = _cellToDouble(row.isNotEmpty ? row[0] : null);
      final l = _cellToDouble(row.length > 1 ? row[1] : null);
      final m = _cellToDouble(row.length > 2 ? row[2] : null);
      final s = _cellToDouble(row.length > 3 ? row[3] : null);

      if (indexVal == null || l == null || m == null || s == null) continue;

      result.add(_LmsRow(indexValue: indexVal, l: l, m: m, s: s));
    }

    // Sort by index value for binary search / interpolation.
    result.sort((a, b) => a.indexValue.compareTo(b.indexValue));
    return result;
  }

  // ---------------------------------------------------------------------------
  // Public API: HAZ lookups
  // ---------------------------------------------------------------------------

  /// Get HAZ Z-score boundary values for given sex and age.
  ///
  /// Returns a map with keys -3..3 (z-score levels) and values in cm,
  /// or null if the combination is not found.
  Map<int, double>? getHazBoundaries(String sex, int ageMonths) {
    final measure = ageMonths < 24 ? 'length' : 'height';
    for (final row in _hazData) {
      if (row.sex == sex &&
          row.measure == measure &&
          row.ageMonths == ageMonths) {
        return Map<int, double>.from(row.zBoundaries);
      }
    }
    return null;
  }

  /// Get median height (z=0) for a given sex and age.
  ///
  /// Returns the WHO median height in cm, or null if age is out of range.
  double? getMedianHeightForAge(String sex, int ageMonths) {
    final boundaries = getHazBoundaries(sex, ageMonths);
    return boundaries?[0];
  }

  /// Return official WHO reference values for a child of [sex] and [ageMonths].
  ///
  /// The target is the WHO median (z=0), with the -2 to +2 z-score interval
  /// supplied as context. These values are population references and must not
  /// be treated as measurements of the child.
  WhoReferenceTargets getReferenceTargets(String sex, double ageMonths) {
    if (ageMonths < 0) return const WhoReferenceTargets();

    final normalizedSex = _normalizeSex(sex);
    final heightDataset = ageMonths < 24 ? _lfaLms : _hfaLms;

    return WhoReferenceTargets(
      heightForAge: _referenceAt(heightDataset[normalizedSex], ageMonths),
      weightForAge: _referenceAt(_wfaLms[normalizedSex], ageMonths),
      muacForAge: _referenceAt(_acfaLms[normalizedSex], ageMonths),
    );
  }

  /// Get approximate standard deviation of height for a given age.
  ///
  /// Computed as (z_0 - z_minus_1). Returns null if age is out of range.
  double? getHeightSdForAge(String sex, int ageMonths) {
    final boundaries = getHazBoundaries(sex, ageMonths);
    if (boundaries == null) return null;
    return boundaries[0]! - boundaries[-1]!;
  }

  /// Get valid height range for a given age (±numSd standard deviations).
  ///
  /// Returns (min, max) in cm, or null if age is out of range.
  (double, double)? getHeightRangeForAge(
    String sex,
    int ageMonths, {
    double numSd = 3.0,
  }) {
    final boundaries = getHazBoundaries(sex, ageMonths);
    if (boundaries == null) return null;

    // For exactly ±3 SD, use the pre-computed boundaries.
    if (numSd == 3.0) {
      return (boundaries[-3]!, boundaries[3]!);
    }

    // For other SD values, interpolate from median and SD.
    final z0 = boundaries[0]!;
    final sd = z0 - boundaries[-1]!;
    return (z0 - numSd * sd, z0 + numSd * sd);
  }

  // ---------------------------------------------------------------------------
  // Public API: WFL / WFH LMS lookups
  // ---------------------------------------------------------------------------

  /// Get (L, M, S) parameters for weight-for-height/length.
  ///
  /// Dataset selection:
  ///   - ageMonths < 24 → WFL (Weight-for-Length)
  ///   - ageMonths >= 24 → WFH (Weight-for-Height)
  ///
  /// Performs exact match (within 0.05 cm tolerance) or linear interpolation
  /// between the two nearest entries.
  ///
  /// Returns (L, M, S) or null if out of range.
  (double, double, double)? getWfhLms(
    String sex,
    double heightCm,
    double ageMonths,
  ) {
    final dataset = ageMonths < 24 ? _wflLms : _wfhLms;
    final rows = dataset[sex];
    if (rows == null || rows.isEmpty) return null;

    // Try exact match first (tolerance 0.05 cm).
    for (final row in rows) {
      if ((row.indexValue - heightCm).abs() <= 0.05) {
        return (row.l, row.m, row.s);
      }
    }

    // Interpolate between two nearest entries.
    _LmsRow? below;
    _LmsRow? above;

    for (final row in rows) {
      if (row.indexValue <= heightCm) {
        below = row;
      }
      if (row.indexValue >= heightCm && above == null) {
        above = row;
      }
    }

    if (below == null || above == null) return null;

    final denom = above.indexValue - below.indexValue;
    if (denom == 0) return null;

    final fraction = (heightCm - below.indexValue) / denom;
    final l = below.l + fraction * (above.l - below.l);
    final m = below.m + fraction * (above.m - below.m);
    final s = below.s + fraction * (above.s - below.s);

    return (l, m, s);
  }

  /// Get median weight for a given height using LMS parameters.
  ///
  /// The M parameter in LMS is the median. [ageMonths] determines whether to
  /// use WFL (< 24) or WFH (>= 24) data.
  double? getMedianWeightForHeight(
    String sex,
    double heightCm, {
    double ageMonths = 36.0,
  }) {
    final lms = getWfhLms(sex, heightCm, ageMonths);
    if (lms == null) return null;
    return lms.$2; // M = median
  }

  // ---------------------------------------------------------------------------
  // Static: LMS z-score formula
  // ---------------------------------------------------------------------------

  /// Compute a z-score from a measurement using LMS parameters.
  ///
  /// Formula (matches Python backend exactly):
  /// ```
  /// if |L| < 1e-6:
  ///     z = ln(measurement / M) / S
  /// else:
  ///     z = (((measurement / M) ^ L) - 1) / (L * S)
  /// ```
  static double lmsZscore(double measurement, double l, double m, double s) {
    if (l.abs() < 1e-6) {
      return log(measurement / m) / s;
    }
    return (pow(measurement / m, l) - 1) / (l * s);
  }

  /// Convert an LMS z-score back to its measurement value.
  static double lmsMeasurement(double z, double l, double m, double s) {
    if (l.abs() < 1e-6) {
      return m * exp(s * z);
    }
    return m * pow(1 + l * s * z, 1 / l);
  }

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  static double? _toDouble(dynamic v) {
    if (v is double) return v;
    if (v is int) return v.toDouble();
    if (v is num) return v.toDouble();
    if (v is String) {
      // Handle Unicode minus sign (U+2212) that some Excel exports produce.
      final cleaned = v.replaceAll('\u2212', '-').trim();
      return double.tryParse(cleaned);
    }
    return null;
  }

  static WhoReferenceValue? _referenceAt(
    List<_LmsRow>? rows,
    double indexValue,
  ) {
    final lms = _interpolateLms(rows, indexValue);
    if (lms == null) return null;
    final (l, m, s) = lms;
    return WhoReferenceValue(
      target: m,
      lower2Sd: lmsMeasurement(-2, l, m, s),
      upper2Sd: lmsMeasurement(2, l, m, s),
    );
  }

  static (double, double, double)? _interpolateLms(
    List<_LmsRow>? rows,
    double indexValue,
  ) {
    if (rows == null || rows.isEmpty) return null;
    if (indexValue < rows.first.indexValue ||
        indexValue > rows.last.indexValue) {
      return null;
    }

    _LmsRow? below;
    _LmsRow? above;
    for (final row in rows) {
      if ((row.indexValue - indexValue).abs() < 1e-9) {
        return (row.l, row.m, row.s);
      }
      if (row.indexValue < indexValue) below = row;
      if (row.indexValue > indexValue) {
        above = row;
        break;
      }
    }
    if (below == null || above == null) return null;

    final fraction =
        (indexValue - below.indexValue) / (above.indexValue - below.indexValue);
    return (
      below.l + fraction * (above.l - below.l),
      below.m + fraction * (above.m - below.m),
      below.s + fraction * (above.s - below.s),
    );
  }

  static String _normalizeSex(String sex) {
    final normalized = sex.trim().toUpperCase();
    if (normalized == 'M' || normalized == 'MALE' || normalized == 'BOY') {
      return 'M';
    }
    if (normalized == 'F' || normalized == 'FEMALE' || normalized == 'GIRL') {
      return 'F';
    }
    return normalized;
  }

  static int? _toInt(dynamic v) {
    if (v is int) return v;
    if (v is double) return v.round();
    if (v is num) return v.round();
    if (v is String) return int.tryParse(v.trim());
    return null;
  }

  /// Extract a double value from an Excel [Data] cell.
  static double? _cellToDouble(Data? cell) {
    if (cell == null) return null;
    final value = cell.value;
    if (value == null) return null;

    return switch (value) {
      IntCellValue(:final value) => value.toDouble(),
      DoubleCellValue(:final value) => value,
      TextCellValue() => _toDouble(value.toString()),
      _ => _toDouble(value.toString()),
    };
  }
}
