import '../constants/config.dart';

class MuacResult {
  final double? muacCm;
  final String? muacStatus;
  final String muacMethod;
  final bool ageInRange;

  const MuacResult({
    this.muacCm,
    this.muacStatus,
    required this.muacMethod,
    required this.ageInRange,
  });
}

class MuacService {
  static MuacResult estimate({
    required double ageMonths,
    required String sex,
    required double? whz,
    double? manualMuacCm,
  }) {
    final ageInRange = ageMonths >= 6.0 && ageMonths < 60.0;

    if (manualMuacCm != null && manualMuacCm > 0) {
      return MuacResult(
        // Preserve the tape reading exactly for clinical thresholding and
        // persistence. Any display rounding belongs in the UI; rounding here
        // can move 11.49 from SAM to MAM or 12.49 from MAM to Normal.
        muacCm: manualMuacCm,
        muacStatus: classifyMuac(manualMuacCm, ageInRange),
        muacMethod: 'manual',
        ageInRange: ageInRange,
      );
    }

    if (whz == null || !ageInRange) {
      return MuacResult(
        muacCm: null,
        muacStatus: null,
        muacMethod: 'unavailable',
        ageInRange: ageInRange,
      );
    }

    final median = medianForAge(ageMonths, sex);
    final whzClamped = whz.clamp(-3.0, 3.0);
    final muacCm = double.parse(
      (median * (1.0 + 0.087 * whzClamped)).toStringAsFixed(1),
    );
    return MuacResult(
      muacCm: muacCm,
      muacStatus: classifyMuac(muacCm, ageInRange),
      muacMethod: 'whz_derived',
      ageInRange: ageInRange,
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
