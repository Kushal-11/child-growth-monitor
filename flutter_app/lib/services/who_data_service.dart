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
library;

import 'dart:convert';
import 'dart:io';
import 'dart:math';

import 'package:crypto/crypto.dart';
import 'package:excel/excel.dart';
import 'package:flutter/services.dart' show rootBundle;

import '../models/who_reference_targets.dart';

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
  static const _manifestAsset = 'assets/who_data/who_reference_manifest.json';
  static const _lfaBoysAsset = 'assets/who_data/who_lhfa_boys_0_2.xlsx';
  static const _lfaGirlsAsset = 'assets/who_data/who_lhfa_girls_0_2.xlsx';
  static const _hfaBoysAsset = 'assets/who_data/who_lhfa_boys_2_5.xlsx';
  static const _hfaGirlsAsset = 'assets/who_data/who_lhfa_girls_2_5.xlsx';
  static const _wfaBoysAsset = 'assets/who_data/who_wfa_boys_0_5.xlsx';
  static const _wfaGirlsAsset = 'assets/who_data/who_wfa_girls_0_5.xlsx';
  static const _bfaBoys0To2Asset = 'assets/who_data/who_bfa_boys_0_2.xlsx';
  static const _bfaBoys2To5Asset = 'assets/who_data/who_bfa_boys_2_5.xlsx';
  static const _bfaGirls0To2Asset = 'assets/who_data/who_bfa_girls_0_2.xlsx';
  static const _bfaGirls2To5Asset = 'assets/who_data/who_bfa_girls_2_5.xlsx';

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

  /// BMI-for-age LMS rows keyed by sex and measurement basis.
  Map<String, List<_LmsRow>> _bfa0To2Lms = {};
  Map<String, List<_LmsRow>> _bfa2To5Lms = {};

  /// Arm-circumference-for-age LMS rows keyed by sex, indexed by age in months.
  Map<String, List<_LmsRow>> _acfaLms = {};

  bool _loaded = false;

  bool get isLoaded => _loaded;

  // ---------------------------------------------------------------------------
  // Loading: from Flutter assets (production)
  // ---------------------------------------------------------------------------

  /// Load all WHO data from bundled Flutter assets.
  Future<void> loadFromAssets() async {
    final manifest = _parseManifest(
      await rootBundle.loadString(_manifestAsset),
      _manifestAsset,
    );

    final wflBoys =
        await rootBundle.load('assets/who_data/who_wfl_boys_0_2.xlsx');
    final wflGirls =
        await rootBundle.load('assets/who_data/who_wfl_girls_0_2.xlsx');
    final wfhBoys =
        await rootBundle.load('assets/who_data/who_wfh_boys_2_5.xlsx');
    final wfhGirls =
        await rootBundle.load('assets/who_data/who_wfh_girls_2_5.xlsx');
    final lfaBoys = await rootBundle.load(_lfaBoysAsset);
    final lfaGirls = await rootBundle.load(_lfaGirlsAsset);
    final hfaBoys = await rootBundle.load(_hfaBoysAsset);
    final hfaGirls = await rootBundle.load(_hfaGirlsAsset);
    final wfaBoys = await rootBundle.load(_wfaBoysAsset);
    final wfaGirls = await rootBundle.load(_wfaGirlsAsset);
    final bfaBoys0To2 = await rootBundle.load(_bfaBoys0To2Asset);
    final bfaBoys2To5 = await rootBundle.load(_bfaBoys2To5Asset);
    final bfaGirls0To2 = await rootBundle.load(_bfaGirls0To2Asset);
    final bfaGirls2To5 = await rootBundle.load(_bfaGirls2To5Asset);
    final acfaBoys =
        await rootBundle.load('assets/who_data/who_acfa_boys_3_5.xlsx');
    final acfaGirls =
        await rootBundle.load('assets/who_data/who_acfa_girls_3_5.xlsx');

    final lfaBoysBytes = lfaBoys.buffer.asUint8List();
    final lfaGirlsBytes = lfaGirls.buffer.asUint8List();
    final hfaBoysBytes = hfaBoys.buffer.asUint8List();
    final hfaGirlsBytes = hfaGirls.buffer.asUint8List();
    _verifyReferenceBytes(_lfaBoysAsset, lfaBoysBytes, manifest);
    _verifyReferenceBytes(_lfaGirlsAsset, lfaGirlsBytes, manifest);
    _verifyReferenceBytes(_hfaBoysAsset, hfaBoysBytes, manifest);
    _verifyReferenceBytes(_hfaGirlsAsset, hfaGirlsBytes, manifest);
    _verifyReferenceBytes(
      _wfaBoysAsset,
      wfaBoys.buffer.asUint8List(),
      manifest,
    );
    _verifyReferenceBytes(
      _wfaGirlsAsset,
      wfaGirls.buffer.asUint8List(),
      manifest,
    );
    _verifyReferenceBytes(
      _bfaBoys0To2Asset,
      bfaBoys0To2.buffer.asUint8List(),
      manifest,
    );
    _verifyReferenceBytes(
      _bfaBoys2To5Asset,
      bfaBoys2To5.buffer.asUint8List(),
      manifest,
    );
    _verifyReferenceBytes(
      _bfaGirls0To2Asset,
      bfaGirls0To2.buffer.asUint8List(),
      manifest,
    );
    _verifyReferenceBytes(
      _bfaGirls2To5Asset,
      bfaGirls2To5.buffer.asUint8List(),
      manifest,
    );

    _wflLms = {
      'M': _parseExcelLms(wflBoys.buffer.asUint8List()),
      'F': _parseExcelLms(wflGirls.buffer.asUint8List()),
    };
    _wfhLms = {
      'M': _parseExcelLms(wfhBoys.buffer.asUint8List()),
      'F': _parseExcelLms(wfhGirls.buffer.asUint8List()),
    };
    _lfaLms = {
      'M': _parseExcelLms(lfaBoysBytes),
      'F': _parseExcelLms(lfaGirlsBytes),
    };
    _hfaLms = {
      'M': _parseExcelLms(hfaBoysBytes),
      'F': _parseExcelLms(hfaGirlsBytes),
    };
    _wfaLms = {
      'M': _parseExcelLms(wfaBoys.buffer.asUint8List()),
      'F': _parseExcelLms(wfaGirls.buffer.asUint8List()),
    };
    _bfa0To2Lms = {
      'M': _parseExcelLms(bfaBoys0To2.buffer.asUint8List()),
      'F': _parseExcelLms(bfaGirls0To2.buffer.asUint8List()),
    };
    _bfa2To5Lms = {
      'M': _parseExcelLms(bfaBoys2To5.buffer.asUint8List()),
      'F': _parseExcelLms(bfaGirls2To5.buffer.asUint8List()),
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
    required String manifestPath,
    required String wflBoysPath,
    required String wflGirlsPath,
    required String wfhBoysPath,
    required String wfhGirlsPath,
    required String lfaBoysPath,
    required String lfaGirlsPath,
    required String hfaBoysPath,
    required String hfaGirlsPath,
    String? wfaBoysPath,
    String? wfaGirlsPath,
    String? bfaBoys0To2Path,
    String? bfaBoys2To5Path,
    String? bfaGirls0To2Path,
    String? bfaGirls2To5Path,
    String? acfaBoysPath,
    String? acfaGirlsPath,
  }) async {
    final manifest = _parseManifest(
      await File(manifestPath).readAsString(),
      manifestPath,
    );
    final lfaBoysBytes = await File(lfaBoysPath).readAsBytes();
    final lfaGirlsBytes = await File(lfaGirlsPath).readAsBytes();
    final hfaBoysBytes = await File(hfaBoysPath).readAsBytes();
    final hfaGirlsBytes = await File(hfaGirlsPath).readAsBytes();
    _verifyReferenceBytes(_lfaBoysAsset, lfaBoysBytes, manifest);
    _verifyReferenceBytes(_lfaGirlsAsset, lfaGirlsBytes, manifest);
    _verifyReferenceBytes(_hfaBoysAsset, hfaBoysBytes, manifest);
    _verifyReferenceBytes(_hfaGirlsAsset, hfaGirlsBytes, manifest);

    _wflLms = {
      'M': _parseExcelLms(await File(wflBoysPath).readAsBytes()),
      'F': _parseExcelLms(await File(wflGirlsPath).readAsBytes()),
    };
    _wfhLms = {
      'M': _parseExcelLms(await File(wfhBoysPath).readAsBytes()),
      'F': _parseExcelLms(await File(wfhGirlsPath).readAsBytes()),
    };
    _lfaLms = {
      'M': _parseExcelLms(lfaBoysBytes),
      'F': _parseExcelLms(lfaGirlsBytes),
    };
    _hfaLms = {
      'M': _parseExcelLms(hfaBoysBytes),
      'F': _parseExcelLms(hfaGirlsBytes),
    };
    if (wfaBoysPath != null && wfaGirlsPath != null) {
      final wfaBoysBytes = await File(wfaBoysPath).readAsBytes();
      final wfaGirlsBytes = await File(wfaGirlsPath).readAsBytes();
      _verifyReferenceBytes(_wfaBoysAsset, wfaBoysBytes, manifest);
      _verifyReferenceBytes(_wfaGirlsAsset, wfaGirlsBytes, manifest);
      _wfaLms = {
        'M': _parseExcelLms(wfaBoysBytes),
        'F': _parseExcelLms(wfaGirlsBytes),
      };
    }
    if (bfaBoys0To2Path != null &&
        bfaBoys2To5Path != null &&
        bfaGirls0To2Path != null &&
        bfaGirls2To5Path != null) {
      final bfaBoys0To2Bytes = await File(bfaBoys0To2Path).readAsBytes();
      final bfaBoys2To5Bytes = await File(bfaBoys2To5Path).readAsBytes();
      final bfaGirls0To2Bytes = await File(bfaGirls0To2Path).readAsBytes();
      final bfaGirls2To5Bytes = await File(bfaGirls2To5Path).readAsBytes();
      _verifyReferenceBytes(
        _bfaBoys0To2Asset,
        bfaBoys0To2Bytes,
        manifest,
      );
      _verifyReferenceBytes(
        _bfaBoys2To5Asset,
        bfaBoys2To5Bytes,
        manifest,
      );
      _verifyReferenceBytes(
        _bfaGirls0To2Asset,
        bfaGirls0To2Bytes,
        manifest,
      );
      _verifyReferenceBytes(
        _bfaGirls2To5Asset,
        bfaGirls2To5Bytes,
        manifest,
      );
      _bfa0To2Lms = {
        'M': _parseExcelLms(bfaBoys0To2Bytes),
        'F': _parseExcelLms(bfaGirls0To2Bytes),
      };
      _bfa2To5Lms = {
        'M': _parseExcelLms(bfaBoys2To5Bytes),
        'F': _parseExcelLms(bfaGirls2To5Bytes),
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
    final lms = getHazLms(sex, ageMonths);
    if (lms == null) return null;
    return {
      for (var z = -3; z <= 3; z++)
        z: lmsMeasurement(z.toDouble(), lms.$1, lms.$2, lms.$3),
    };
  }

  /// Get official WHO length/height-for-age (L, M, S) parameters.
  ///
  /// Recumbent length is used through month 23 and standing height from
  /// month 24 through month 60.
  (double, double, double)? getHazLms(String sex, int ageMonths) {
    if (ageMonths < 0 || ageMonths > 60) return null;
    final normalizedSex = _normalizeSex(sex);
    final rows = (ageMonths < 24 ? _lfaLms : _hfaLms)[normalizedSex];
    return _interpolateLms(rows, ageMonths.toDouble());
  }

  /// Get HAZ LMS parameters using exact decimal age rather than truncating to
  /// completed months.
  (double, double, double)? getHazLmsForAge(
    String sex,
    double ageMonths,
  ) {
    if (!ageMonths.isFinite || ageMonths < 0 || ageMonths >= 60) return null;
    final normalizedSex = _normalizeSex(sex);
    final rows = (ageMonths < 24 ? _lfaLms : _hfaLms)[normalizedSex];
    return _interpolateLms(rows, ageMonths);
  }

  /// Return official WHO weight-for-age LMS parameters.
  (double, double, double)? getWfaLms(String sex, double ageMonths) {
    if (!ageMonths.isFinite || ageMonths < 0 || ageMonths >= 60) return null;
    return _interpolateLms(_wfaLms[_normalizeSex(sex)], ageMonths);
  }

  /// Return official WHO BMI-for-age LMS parameters.
  (double, double, double)? getBfaLms(String sex, double ageMonths) {
    if (!ageMonths.isFinite || ageMonths < 0 || ageMonths >= 60) return null;
    final normalizedSex = _normalizeSex(sex);
    final rows = (ageMonths < 24 ? _bfa0To2Lms : _bfa2To5Lms)[normalizedSex];
    return _interpolateLms(rows, ageMonths);
  }

  /// Get median height (z=0) for a given sex and age.
  ///
  /// Returns the WHO median height in cm, or null if age is out of range.
  double? getMedianHeightForAge(String sex, int ageMonths) {
    return getHazLms(sex, ageMonths)?.$2;
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
    final lms = getHazLms(sex, ageMonths);
    if (lms == null) return null;
    return lms.$2 - lmsMeasurement(-1, lms.$1, lms.$2, lms.$3);
  }

  /// Get valid height range for a given age (±numSd standard deviations).
  ///
  /// Returns (min, max) in cm, or null if age is out of range.
  (double, double)? getHeightRangeForAge(
    String sex,
    int ageMonths, {
    double numSd = 3.0,
  }) {
    final lms = getHazLms(sex, ageMonths);
    if (lms == null) return null;
    return (
      lmsMeasurement(-numSd, lms.$1, lms.$2, lms.$3),
      lmsMeasurement(numSd, lms.$1, lms.$2, lms.$3),
    );
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
    final rows = dataset[_normalizeSex(sex)];
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

  static Map<String, dynamic> _parseManifest(
    String manifestJson,
    String source,
  ) {
    try {
      final decoded = jsonDecode(manifestJson);
      if (decoded is! Map<String, dynamic> ||
          decoded['schema_version'] != 1 ||
          decoded['files'] is! Map<String, dynamic>) {
        throw const FormatException('unsupported schema');
      }
      return decoded;
    } on FormatException catch (error) {
      throw StateError(
        'Authoritative WHO reference manifest is malformed: $source: $error',
      );
    }
  }

  static void _verifyReferenceBytes(
    String assetPath,
    List<int> bytes,
    Map<String, dynamic> manifest,
  ) {
    final fileName = assetPath.split('/').last;
    final files = manifest['files'] as Map<String, dynamic>;
    final record = files[fileName];
    if (record is! Map<String, dynamic>) {
      throw StateError(
        'WHO reference manifest has no entry for $fileName',
      );
    }
    final expectedSize = record['size_bytes'];
    final expectedChecksum = record['sha256'];
    if (expectedSize is! int || expectedChecksum is! String) {
      throw StateError(
        'WHO reference manifest entry for $fileName is malformed',
      );
    }
    if (bytes.length != expectedSize) {
      throw StateError(
        'WHO reference size mismatch for $fileName: '
        'expected $expectedSize, got ${bytes.length}',
      );
    }
    final actualChecksum = sha256.convert(bytes).toString();
    if (actualChecksum != expectedChecksum) {
      throw StateError(
        'WHO reference checksum mismatch for $fileName: '
        'expected $expectedChecksum, got $actualChecksum',
      );
    }
  }

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
