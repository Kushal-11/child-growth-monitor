/// One persisted assessment/report projected onto the clinical predictions
/// CSV contract used by the field-data pipeline.
class ClinicalCsvRecord {
  const ClinicalCsvRecord({
    required this.childId,
    required this.area,
    required this.sex,
    required this.dateOfBirth,
    required this.measurementDate,
    required this.actualHeightCm,
    required this.calculatedHeightCm,
    required this.actualWeightKg,
    required this.calculatedWeightKg,
    required this.muacCm,
    required this.calculatedMuacCm,
    required this.fieldCategory,
    required this.predictedFieldCategory,
    required this.stuntingPrediction,
    required this.wastingPrediction,
    required this.notes,
  });

  static const headers = <String>[
    'child_id',
    'area',
    'sex',
    'date_of_birth',
    'measurement_date',
    'actual_height_cm',
    'calculated_height_cm',
    'actual_weight_kg',
    'calculated_weight_kg',
    'muac_cm',
    'calculated_muac_cm',
    'field_category',
    'predicted_field_category',
    'stunting_prediction',
    'wasting_prediction',
    'notes',
  ];

  final int childId;
  final String? area;
  final String sex;
  final String dateOfBirth;
  final String measurementDate;
  final double? actualHeightCm;
  final double? calculatedHeightCm;
  final double? actualWeightKg;
  final double? calculatedWeightKg;
  final double? muacCm;
  final double? calculatedMuacCm;
  final String? fieldCategory;
  final String? predictedFieldCategory;
  final String? stuntingPrediction;
  final String? wastingPrediction;
  final String? notes;

  List<Object?> toCsvRow() => <Object?>[
        childId,
        area,
        sex,
        dateOfBirth,
        measurementDate,
        _formatNumber(actualHeightCm),
        _formatNumber(calculatedHeightCm),
        _formatNumber(actualWeightKg),
        _formatNumber(calculatedWeightKg),
        _formatNumber(muacCm),
        _formatNumber(calculatedMuacCm),
        fieldCategory,
        predictedFieldCategory,
        stuntingPrediction,
        wastingPrediction,
        notes,
      ];

  static String? _formatNumber(double? value) {
    if (value == null) return null;
    final fixed = value.toStringAsFixed(4);
    return fixed.replaceFirst(RegExp(r'\.?0+$'), '');
  }
}
