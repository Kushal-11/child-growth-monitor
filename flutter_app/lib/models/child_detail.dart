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
    this.measurement,
  });

  final int visitId;
  final String? visitDate;
  final double? ageMonths;
  final ChildVisitMeasurement? measurement;

  factory ChildVisit.fromJson(Map<String, dynamic> json) {
    return ChildVisit(
      visitId: json['visit_id'] as int,
      visitDate: json['visit_date'] as String?,
      ageMonths: (json['age_months'] as num?)?.toDouble(),
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
    this.hazZscore,
    this.whzZscore,
    this.hazStatus,
    this.whzStatus,
    this.confidenceScore,
  });

  final double? predictedHeightCm;
  final double? predictedWeightKg;
  final double? hazZscore;
  final double? whzZscore;
  final String? hazStatus;
  final String? whzStatus;
  final double? confidenceScore;

  factory ChildVisitMeasurement.fromJson(Map<String, dynamic> json) {
    return ChildVisitMeasurement(
      predictedHeightCm: (json['predicted_height_cm'] as num?)?.toDouble(),
      predictedWeightKg: (json['predicted_weight_kg'] as num?)?.toDouble(),
      hazZscore: (json['haz_zscore'] as num?)?.toDouble(),
      whzZscore: (json['whz_zscore'] as num?)?.toDouble(),
      hazStatus: json['haz_status'] as String?,
      whzStatus: json['whz_status'] as String?,
      confidenceScore: (json['confidence_score'] as num?)?.toDouble(),
    );
  }
}
