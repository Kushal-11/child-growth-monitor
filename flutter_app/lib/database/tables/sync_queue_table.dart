import 'package:drift/drift.dart';
import 'visits_table.dart';

class SyncQueue extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get visitId => integer().references(Visits, #id)();
  TextColumn get status => text().withDefault(const Constant('pending'))();
  IntColumn get retryCount => integer().withDefault(const Constant(0))();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get lastAttemptAt => dateTime().nullable()();
  IntColumn get serverVisitId => integer().nullable()();
  TextColumn get errorMessage => text().nullable()();
}
