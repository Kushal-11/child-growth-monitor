import 'package:drift/drift.dart';

class Children extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get name => text()();
  TextColumn get dateOfBirth => text()(); // ISO 8601
  TextColumn get sex => text().withLength(min: 1, max: 1)(); // M or F
  TextColumn get guardianName => text().nullable()();
  TextColumn get location => text().nullable()();
  IntColumn get ownerUserId => integer().nullable()();
  TextColumn get photoPath => text().nullable()();
  BoolColumn get isArchived => boolean().withDefault(const Constant(false))();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get updatedAt => dateTime().withDefault(currentDateAndTime)();
}
