/// Pure, measurement-provenance-aware implementation of the Poshan Setu v1
/// contract in `docs/POSHAN_SETU_V1.md`.
///
/// WHO statistical values and ML estimates remain useful secondary screening
/// signals, but they are deliberately ineligible to certify a non-SAM Poshan
/// result.
class PoshanSetuResult {
  const PoshanSetuResult({
    required this.bmi,
    required this.bmiStatus,
    required this.muacStatus,
    required this.finalStatus,
    required this.triggeredBy,
    required this.classificationMethod,
    required this.rationale,
    required this.complete,
  });

  final double? bmi;
  final String bmiStatus;
  final String muacStatus;
  final String finalStatus;
  final List<String> triggeredBy;
  final String classificationMethod;
  final String rationale;

  /// Whether both BMI and MUAC were eligible. A SAM result may still be final
  /// when false because any eligible SAM component takes precedence.
  final bool complete;
}

class PoshanSetuService {
  const PoshanSetuService();

  static const String method = 'poshan_setu_v1';
  static const String sam = 'SAM';
  static const String mam = 'MAM';
  static const String normal = 'Normal';
  static const String indeterminate = 'Indeterminate';

  static const Set<String> _eligibleBodyMeasurementSources = {
    'manual',
    'reference_object',
  };
  static const Set<String> _eligibleMuacSources = {'manual'};
  static const Set<String> _canonicalSources = {
    'manual',
    'reference_object',
    'ml_estimated',
    'who_statistical',
    'whz_derived',
    'landmark_estimated',
    'unavailable',
  };

  PoshanSetuResult classify({
    required String sex,
    required double ageMonths,
    required double? heightCm,
    required String heightSource,
    required double? weightKg,
    required String weightSource,
    required double? muacCm,
    required String muacSource,
  }) {
    final normalizedHeightSource = normalizeSource(heightSource);
    final normalizedWeightSource = normalizeSource(weightSource);
    final normalizedMuacSource = normalizeMuacSource(muacSource);
    final normalizedSex = sex.trim().toUpperCase();
    final bmiEligible =
        _eligibleBodyMeasurementSources.contains(normalizedHeightSource) &&
            _eligibleBodyMeasurementSources.contains(normalizedWeightSource) &&
            heightCm != null &&
            heightCm.isFinite &&
            heightCm > 0 &&
            weightKg != null &&
            weightKg.isFinite &&
            weightKg > 0 &&
            (normalizedSex == 'M' || normalizedSex == 'F');
    final bmi =
        bmiEligible ? weightKg / ((heightCm / 100) * (heightCm / 100)) : null;
    final bmiStatus = bmi == null || !bmi.isFinite
        ? indeterminate
        : _classifyBmi(normalizedSex, bmi);

    final muacEligible = _eligibleMuacSources.contains(normalizedMuacSource) &&
        ageMonths.isFinite &&
        ageMonths >= 6.0 &&
        ageMonths < 60.0 &&
        muacCm != null &&
        muacCm.isFinite &&
        muacCm > 0;
    final muacStatus = muacEligible ? _classifyMuac(muacCm) : indeterminate;
    final complete = bmiStatus != indeterminate && muacStatus != indeterminate;

    late final String finalStatus;
    if (bmiStatus == sam || muacStatus == sam) {
      finalStatus = sam;
    } else if (complete) {
      finalStatus = bmiStatus == mam || muacStatus == mam ? mam : normal;
    } else {
      finalStatus = indeterminate;
    }

    final triggeredBy = _triggeredBy(
      bmiStatus: bmiStatus,
      muacStatus: muacStatus,
      finalStatus: finalStatus,
    );
    return PoshanSetuResult(
      bmi: bmi,
      bmiStatus: bmiStatus,
      muacStatus: muacStatus,
      finalStatus: finalStatus,
      triggeredBy: triggeredBy,
      classificationMethod: method,
      rationale: _rationale(
        bmiStatus: bmiStatus,
        muacStatus: muacStatus,
        finalStatus: finalStatus,
        triggeredBy: triggeredBy,
      ),
      complete: complete,
    );
  }

  /// Canonicalises legacy values still found in pre-v4 rows/API responses.
  static String normalizeSource(String source) {
    final normalized = switch (source.trim().toLowerCase()) {
      'who_median_estimated' => 'who_statistical',
      'estimated_from_whz' => 'whz_derived',
      'anthropometric' => 'landmark_estimated',
      '' || 'none' || 'unknown' => 'unavailable',
      _ => source.trim().toLowerCase(),
    };
    return _canonicalSources.contains(normalized) ? normalized : 'unavailable';
  }

  /// MUAC accepts the historical `tape` label as a manual tape measurement.
  static String normalizeMuacSource(String source) {
    final normalized = source.trim().toLowerCase();
    return normalized == 'tape' ? 'manual' : normalizeSource(normalized);
  }

  static bool isEligibleBodyMeasurementSource(String source) {
    return _eligibleBodyMeasurementSources.contains(normalizeSource(source));
  }

  String _classifyBmi(String sex, double bmi) {
    if (sex.toUpperCase() == 'M') {
      if (bmi < 13.0) return sam;
      if (bmi < 13.7) return mam;
      return normal;
    }
    if (bmi < 12.8) return sam;
    if (bmi < 13.5) return mam;
    return normal;
  }

  String _classifyMuac(double muacCm) {
    if (muacCm < 11.5) return sam;
    if (muacCm < 12.5) return mam;
    return normal;
  }

  List<String> _triggeredBy({
    required String bmiStatus,
    required String muacStatus,
    required String finalStatus,
  }) {
    final target = finalStatus == sam
        ? sam
        : (bmiStatus == mam || muacStatus == mam)
            ? mam
            : finalStatus == normal
                ? normal
                : null;
    if (target == null) return const [];
    return [
      if (bmiStatus == target) 'bmi',
      if (muacStatus == target) 'muac',
    ];
  }

  String _rationale({
    required String bmiStatus,
    required String muacStatus,
    required String finalStatus,
    required List<String> triggeredBy,
  }) {
    final named = triggeredBy.join(' and ');
    if (finalStatus == sam) {
      return 'Eligible $named measurement classified as SAM; any SAM '
          'component determines the final result.';
    }
    if (finalStatus == mam) {
      return 'Both eligible components were available; $named produced the '
          'more severe MAM result.';
    }
    if (finalStatus == normal) {
      return 'Both eligible BMI and MUAC measurements classified as Normal.';
    }
    if (bmiStatus == mam || muacStatus == mam) {
      final known = bmiStatus == mam ? 'BMI' : 'MUAC';
      final missing = bmiStatus == indeterminate ? 'BMI' : 'MUAC';
      return '$known classified as MAM, but eligible $missing is unavailable; '
          'both components are required for a non-SAM final result.';
    }
    if (bmiStatus != indeterminate || muacStatus != indeterminate) {
      final known = bmiStatus != indeterminate ? 'BMI' : 'MUAC';
      final missing = bmiStatus == indeterminate ? 'BMI' : 'MUAC';
      return '$known was available, but eligible $missing is unavailable; '
          'both components are required to certify a non-SAM result.';
    }
    return 'Eligible measured height, weight, and tape MUAC are unavailable; '
        'the result cannot be determined.';
  }
}
