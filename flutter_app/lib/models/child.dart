class ChildSummary {
  ChildSummary({
    required this.id,
    required this.name,
    required this.dateOfBirth,
    required this.sex,
    required this.visitCount,
  });

  final int id;
  final String name;
  final String dateOfBirth;
  final String sex;
  final int visitCount;

  factory ChildSummary.fromJson(Map<String, dynamic> json) {
    int parseIntField(String key) {
      final value = json[key];
      if (value is int) return value;
      if (value is String) {
        final parsed = int.tryParse(value);
        if (parsed != null) return parsed;
      }
      throw FormatException('Invalid "$key" in child summary: $value');
    }

    String parseStringField(String key) {
      final value = json[key];
      if (value is String && value.isNotEmpty) return value;
      throw FormatException('Invalid "$key" in child summary: $value');
    }

    return ChildSummary(
      id: parseIntField('id'),
      name: parseStringField('name'),
      dateOfBirth: parseStringField('date_of_birth'),
      sex: parseStringField('sex'),
      visitCount: parseIntField('visit_count'),
    );
  }
}
