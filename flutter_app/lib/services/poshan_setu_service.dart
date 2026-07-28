import '../constants/config.dart';

/// Pure, provenance-aware implementation of the Poshan Setu v1 contract.
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
  final bool complete;
}

class PoshanSetuService {
  const PoshanSetuService();

  static const method = 'poshan_setu_v1';
  static const sam = 'SAM';
  static const mam = 'MAM';
  static const normal = 'Normal';
  static const indeterminate = 'Indeterminate';
  static const _eligibleBodySources = {'manual', 'reference_object'};
  static const _canonicalSources = {
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
    final normalizedSex = sex.trim().toUpperCase();
    final normalizedHeightSource = normalizeSource(heightSource);
    final normalizedWeightSource = normalizeSource(weightSource);
    final normalizedMuacSource = normalizeMuacSource(muacSource);
    final bmiEligible = _eligibleBodySources.contains(normalizedHeightSource) &&
        _eligibleBodySources.contains(normalizedWeightSource) &&
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

    final muacEligible = normalizedMuacSource == 'manual' &&
        ageMonths.isFinite &&
        ageMonths >= poshanMuacMinAgeMonths &&
        ageMonths < poshanMuacMaxAgeMonths &&
        muacCm != null &&
        muacCm.isFinite &&
        muacCm > 0;
    final muacStatus = muacEligible ? _classifyMuac(muacCm) : indeterminate;
    final complete = bmiStatus != indeterminate && muacStatus != indeterminate;
    final finalStatus = bmiStatus == sam || muacStatus == sam
        ? sam
        : complete
            ? (bmiStatus == mam || muacStatus == mam ? mam : normal)
            : indeterminate;
    final target = finalStatus == sam
        ? sam
        : (bmiStatus == mam || muacStatus == mam)
            ? mam
            : finalStatus == normal
                ? normal
                : null;
    final triggeredBy = <String>[
      if (target != null && bmiStatus == target) 'bmi',
      if (target != null && muacStatus == target) 'muac',
    ];
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

  static String normalizeMuacSource(String source) {
    return source.trim().toLowerCase() == 'tape'
        ? 'manual'
        : normalizeSource(source);
  }

  static bool isEligibleBodySource(String source) {
    return _eligibleBodySources.contains(normalizeSource(source));
  }

  String _classifyBmi(String sex, double bmi) {
    final thresholds = poshanBmiThresholds[sex]!;
    if (bmi < thresholds.$1) return sam;
    if (bmi < thresholds.$2) return mam;
    return normal;
  }

  String _classifyMuac(double muacCm) {
    if (muacCm < poshanMuacSamMaxCm) return sam;
    if (muacCm < poshanMuacNormalMinCm) return mam;
    return normal;
  }

  String _rationale({
    required String bmiStatus,
    required String muacStatus,
    required String finalStatus,
    required List<String> triggeredBy,
  }) {
    if (finalStatus == sam) {
      return 'At least one eligible component is SAM; any eligible SAM '
          'component determines the final result.';
    }
    if (finalStatus == mam) {
      return 'Both eligible components were available; '
          '${triggeredBy.join(' and ')} produced the more severe MAM result.';
    }
    if (finalStatus == normal) {
      return 'Both eligible BMI and MUAC measurements classified as Normal.';
    }
    if (bmiStatus == mam || muacStatus == mam) {
      return 'An eligible component classified as MAM, but the other '
          'component is unavailable; both are required for a non-SAM result.';
    }
    return 'Eligible measured BMI and tape MUAC evidence is incomplete.';
  }
}
