import 'package:drift/drift.dart';

class Children extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get name => text()();
  TextColumn get dateOfBirth => text()(); // ISO 8601
  TextColumn get sex => text().withLength(min: 1, max: 1)(); // M or F
  TextColumn get guardianName => text().nullable()();
  TextColumn get location => text().nullable()();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get updatedAt => dateTime().withDefault(currentDateAndTime)();
}
