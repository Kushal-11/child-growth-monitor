import 'package:drift/drift.dart';
import 'children_table.dart';

class Visits extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get childId => integer().references(Children, #id)();
  DateTimeColumn get visitDate => dateTime().withDefault(currentDateAndTime)();
  RealColumn get ageMonths => real()();
  TextColumn get imagePath => text()();
  TextColumn get sideImagePath => text().nullable()();
  TextColumn get backImagePath => text().nullable()();
  TextColumn get notes => text().nullable()();
}
