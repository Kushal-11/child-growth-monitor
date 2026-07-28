import '../constants/config.dart';

class MuacResult {
  final double? muacCm;
  final String? muacStatus;
  final String muacMethod;
  final bool ageInRange;
  final double? confidence;
  final double? uncertaintyLowerCm;
  final double? uncertaintyUpperCm;
  final String? modelVersion;
  final String? calibrationVersion;
  final bool isDirectMeasurement;
  final bool requiresConfirmation;
  final String? referralGuidance;

  const MuacResult({
    this.muacCm,
    this.muacStatus,
    required this.muacMethod,
    required this.ageInRange,
    this.confidence,
    this.uncertaintyLowerCm,
    this.uncertaintyUpperCm,
    this.modelVersion,
    this.calibrationVersion,
    this.isDirectMeasurement = false,
    this.requiresConfirmation = false,
    this.referralGuidance,
  });
}

class MuacService {
  static MuacResult estimate({
    required double ageMonths,
    required String sex,
    required double? whz,
    double? manualMuacCm,
  }) {
    final ageInRange = ageMonths >= 6.0 && ageMonths <= 59.9;

    if (manualMuacCm != null && manualMuacCm > 0) {
      return MuacResult(
        muacCm: double.parse(manualMuacCm.toStringAsFixed(1)),
        muacStatus: classifyMuac(manualMuacCm, ageInRange),
        muacMethod: 'manual',
        ageInRange: ageInRange,
        confidence: 1,
        uncertaintyLowerCm: double.parse(manualMuacCm.toStringAsFixed(1)),
        uncertaintyUpperCm: double.parse(manualMuacCm.toStringAsFixed(1)),
        calibrationVersion: 'direct-tape',
        isDirectMeasurement: true,
      );
    }

    if (whz == null) {
      return MuacResult(
        muacCm: null,
        muacStatus: null,
        muacMethod: 'estimated_from_whz',
        ageInRange: ageInRange,
        modelVersion: 'whz-explanatory-v1',
        calibrationVersion: 'who-median-formula-v1',
        requiresConfirmation: true,
        referralGuidance: 'Obtain a direct tape MUAC measurement.',
      );
    }

    final median = medianForAge(ageMonths, sex);
    final whzClamped = whz.clamp(-3.0, 3.0);
    final muacCm = double.parse(
      (median * (1.0 + 0.087 * whzClamped)).toStringAsFixed(1),
    );
    return MuacResult(
      muacCm: muacCm,
      // This value is derived from WHZ and must not be treated as an
      // independent clinical MUAC classification.
      muacStatus: null,
      muacMethod: 'estimated_from_whz',
      ageInRange: ageInRange,
      confidence: 0.4,
      uncertaintyLowerCm: double.parse((muacCm - 1).toStringAsFixed(1)),
      uncertaintyUpperCm: double.parse((muacCm + 1).toStringAsFixed(1)),
      modelVersion: 'whz-explanatory-v1',
      calibrationVersion: 'who-median-formula-v1',
      requiresConfirmation: true,
      referralGuidance:
          'WHZ-derived MUAC is explanatory only; obtain a direct tape measurement.',
    );
  }

  static double medianForAge(double ageMonths, String sex) {
    final table = sex.toUpperCase() == 'M' ? muacBoys : muacGirls;
    if (ageMonths <= table.first.$1) return table.first.$2;
    if (ageMonths >= table.last.$1) return table.last.$2;
    for (int i = 0; i < table.length - 1; i++) {
      final (a0, m0) = table[i];
      final (a1, m1) = table[i + 1];
      if (a0 <= ageMonths && ageMonths <= a1) {
        final t = (ageMonths - a0) / (a1 - a0);
        return m0 + t * (m1 - m0);
      }
    }
    return table.last.$2;
  }
}
