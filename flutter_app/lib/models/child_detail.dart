class ChildDetail {
  ChildDetail({
    required this.id,
    required this.name,
    required this.dateOfBirth,
    required this.sex,
    this.guardianName,
    this.location,
    required this.visits,
  });

  final int id;
  final String name;
  final String dateOfBirth;
  final String sex;
  final String? guardianName;
  final String? location;
  final List<ChildVisit> visits;

  factory ChildDetail.fromJson(Map<String, dynamic> json) {
    return ChildDetail(
      id: json['id'] as int,
      name: json['name'] as String,
      dateOfBirth: json['date_of_birth'] as String,
      sex: json['sex'] as String,
      guardianName: json['guardian_name'] as String?,
      location: json['location'] as String?,
      visits: (json['visits'] as List<dynamic>? ?? const [])
          .map((v) => ChildVisit.fromJson(v as Map<String, dynamic>))
          .toList(),
    );
  }
}

class ChildVisit {
  ChildVisit({
    required this.visitId,
    this.visitDate,
    this.ageMonths,
    this.entryMethod,
    this.measurement,
  });

  final int visitId;
  final String? visitDate;
  final double? ageMonths;
  final String? entryMethod;
  final ChildVisitMeasurement? measurement;

  factory ChildVisit.fromJson(Map<String, dynamic> json) {
    return ChildVisit(
      visitId: json['visit_id'] as int,
      visitDate: json['visit_date'] as String?,
      ageMonths: (json['age_months'] as num?)?.toDouble(),
      entryMethod: json['entry_method'] as String?,
      measurement: json['measurement'] == null
          ? null
          : ChildVisitMeasurement.fromJson(
              json['measurement'] as Map<String, dynamic>,
            ),
    );
  }
}

class ChildVisitMeasurement {
  ChildVisitMeasurement({
    this.predictedHeightCm,
    this.predictedWeightKg,
    this.manualHeightCm,
    this.manualWeightKg,
    this.effectiveHeightCm,
    this.effectiveWeightKg,
    this.hazZscore,
    this.whzZscore,
    this.hazStatus,
    this.whzStatus,
    this.confidenceScore,
    this.heightSource,
    this.weightSource,
    this.bmi,
    this.bmiStatus,
    this.muacCm,
    this.muacStatus,
    this.muacMethod,
    this.poshanStatus,
    this.poshanTriggeredBy = const [],
    this.classificationMethod,
    this.classificationRationale,
  });

  final double? predictedHeightCm;
  final double? predictedWeightKg;
  final double? manualHeightCm;
  final double? manualWeightKg;
  final double? effectiveHeightCm;
  final double? effectiveWeightKg;
  final double? hazZscore;
  final double? whzZscore;
  final String? hazStatus;
  final String? whzStatus;
  final double? confidenceScore;
  final String? heightSource;
  final String? weightSource;
  final double? bmi;
  final String? bmiStatus;
  final double? muacCm;
  final String? muacStatus;
  final String? muacMethod;
  final String? poshanStatus;
  final List<String> poshanTriggeredBy;
  final String? classificationMethod;
  final String? classificationRationale;

  double? get displayHeightCm =>
      effectiveHeightCm ?? manualHeightCm ?? predictedHeightCm;
  double? get displayWeightKg =>
      effectiveWeightKg ?? manualWeightKg ?? predictedWeightKg;

  factory ChildVisitMeasurement.fromJson(Map<String, dynamic> json) {
    return ChildVisitMeasurement(
      predictedHeightCm: (json['predicted_height_cm'] as num?)?.toDouble(),
      predictedWeightKg: (json['predicted_weight_kg'] as num?)?.toDouble(),
      manualHeightCm: (json['manual_height_cm'] as num?)?.toDouble(),
      manualWeightKg: (json['manual_weight_kg'] as num?)?.toDouble(),
      effectiveHeightCm: (json['effective_height_cm'] as num?)?.toDouble(),
      effectiveWeightKg: (json['effective_weight_kg'] as num?)?.toDouble(),
      hazZscore: (json['haz_zscore'] as num?)?.toDouble(),
      whzZscore: (json['whz_zscore'] as num?)?.toDouble(),
      hazStatus: json['haz_status'] as String?,
      whzStatus: json['whz_status'] as String?,
      confidenceScore: (json['confidence_score'] as num?)?.toDouble(),
      heightSource: json['height_source'] as String?,
      weightSource: json['weight_source'] as String?,
      bmi: (json['bmi'] as num?)?.toDouble(),
      bmiStatus: json['bmi_status'] as String?,
      muacCm: (json['muac_cm'] as num?)?.toDouble(),
      muacStatus: json['muac_status'] as String?,
      muacMethod: json['muac_method'] as String?,
      poshanStatus: json['poshan_status'] as String?,
      poshanTriggeredBy:
          (json['poshan_triggered_by'] as List<dynamic>? ?? const <dynamic>[])
              .whereType<String>()
              .toList(growable: false),
      classificationMethod: json['classification_method'] as String?,
      classificationRationale: json['classification_rationale'] as String?,
    );
  }
}
