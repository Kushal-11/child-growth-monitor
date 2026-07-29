import 'package:drift/drift.dart';
import 'children_table.dart';

@TableIndex(
  name: 'ix_visits_owner_local_uuid',
  columns: {#ownerUserId, #localUuid},
)
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
  TextColumn get captureState => text().nullable()();
  DateTimeColumn get captureStartedAt => dateTime().nullable()();
  DateTimeColumn get captureCompletedAt => dateTime().nullable()();
  TextColumn get deviceMetadataJson => text().nullable()();
  TextColumn get consentVersion => text().nullable()();
  DateTimeColumn get consentTimestamp => dateTime().nullable()();
  TextColumn get consentOperatorIdentifier => text().nullable()();
  DateTimeColumn get mediaDeletedAt => dateTime().nullable()();
}
