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
    return ChildSummary(
      id: json['id'] as int,
      name: json['name'] as String,
      dateOfBirth: json['date_of_birth'] as String,
      sex: json['sex'] as String,
      visitCount: json['visit_count'] as int,
    );
  }
}
