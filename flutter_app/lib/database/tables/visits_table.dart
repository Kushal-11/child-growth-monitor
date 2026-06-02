import 'package:drift/drift.dart';
import 'children_table.dart';

class Visits extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get childId => integer().references(Children, #id)();
  TextColumn get localUuid => text().withLength(min: 36, max: 36).unique()();
  DateTimeColumn get visitDate => dateTime().withDefault(currentDateAndTime)();
  RealColumn get ageMonths => real()();
  TextColumn get imagePath => text().nullable()();
  TextColumn get sideImagePath => text().nullable()();
  TextColumn get backImagePath => text().nullable()();
  TextColumn get notes => text().nullable()();
  IntColumn get ownerUserId => integer().nullable()();
  TextColumn get entryMethod =>
      text().withDefault(const Constant('assessment'))();
}
