class ProtocolResult {
  const ProtocolResult(this.bmiValue, this.bmiStatus, this.finalStatus, this.triggeredIndicators);
  final double? bmiValue;
  final String bmiStatus;
  final String finalStatus;
  final List<String> triggeredIndicators;
}

class ProtocolClassificationService {
  static const samBoundary = {'M': 13.0, 'F': 12.8};
  static const mamBoundary = {'M': 13.7, 'F': 13.5};

  static ProtocolResult classify(double? weightKg, double? heightCm, String sex, String? muacStatus) {
    double? bmi;
    var bmiStatus = 'Insufficient data';
    if (weightKg != null && heightCm != null && weightKg > 0 && heightCm > 0) {
      bmi = weightKg / ((heightCm / 100) * (heightCm / 100));
      final sam = samBoundary[sex.toUpperCase()];
      final mam = mamBoundary[sex.toUpperCase()];
      bmiStatus = sam == null || mam == null ? 'Unknown' : (bmi < sam ? 'SAM' : (bmi < mam ? 'MAM' : 'Normal'));
    }
    final muac = muacStatus == 'At Risk (MAM)' ? 'MAM' : muacStatus;
    for (final severity in ['SAM', 'MAM']) {
      final triggers = <String>[if (bmiStatus == severity) 'bmi', if (muac == severity) 'muac'];
      if (triggers.isNotEmpty) return ProtocolResult(bmi, bmiStatus, severity, triggers);
    }
    if (bmiStatus == 'Normal' || muac == 'Normal') return ProtocolResult(bmi, bmiStatus, 'Normal', const []);
    return ProtocolResult(bmi, bmiStatus, bmiStatus == 'Unknown' ? 'Unknown' : 'Insufficient data', const []);
  }
}
