/// Official WHO Child Growth Standards reference values for a child of the
/// same sex and age.
///
/// These are population reference values, not measurements of the child.
class WhoReferenceValue {
  const WhoReferenceValue({
    required this.target,
    required this.lower2Sd,
    required this.upper2Sd,
  });

  /// WHO median (z-score 0).
  final double target;

  /// Lower bound of the WHO reference interval (z-score -2).
  final double lower2Sd;

  /// Upper bound of the WHO reference interval (z-score +2).
  final double upper2Sd;

  factory WhoReferenceValue.fromJson(Map<String, dynamic> json) {
    return WhoReferenceValue(
      target: (json['target'] as num).toDouble(),
      lower2Sd: (json['lower_2sd'] as num).toDouble(),
      upper2Sd: (json['upper_2sd'] as num).toDouble(),
    );
  }
}

class WhoReferenceTargets {
  const WhoReferenceTargets({
    this.heightForAge,
    this.weightForAge,
    this.muacForAge,
  });

  final WhoReferenceValue? heightForAge;
  final WhoReferenceValue? weightForAge;
  final WhoReferenceValue? muacForAge;

  bool get isEmpty =>
      heightForAge == null && weightForAge == null && muacForAge == null;

  factory WhoReferenceTargets.fromJson(Map<String, dynamic> json) {
    WhoReferenceValue? parse(String key) {
      final value = json[key];
      return value is Map<String, dynamic>
          ? WhoReferenceValue.fromJson(value)
          : null;
    }

    return WhoReferenceTargets(
      heightForAge: parse('height_for_age'),
      weightForAge: parse('weight_for_age'),
      muacForAge: parse('muac_for_age'),
    );
  }
}
