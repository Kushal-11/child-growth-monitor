import 'package:drift/drift.dart';

import 'visits_table.dart';

@TableIndex(
  name: 'ix_measured_revisions_visit_revision',
  columns: {#visitId, #revisionNumber},
)
class MeasuredDetailRevisions extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get revisionUuid => text().unique()();
  IntColumn get serverId => integer().nullable()();
  IntColumn get visitId => integer().references(
        Visits,
        #id,
        onDelete: KeyAction.cascade,
      )();
  IntColumn get revisionNumber => integer()();
  TextColumn get beforeJson => text()();
  TextColumn get afterJson => text()();
  IntColumn get editorUserId => integer().nullable()();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  TextColumn get reason => text().nullable()();

  @override
  List<Set<Column>> get uniqueKeys => [
        {visitId, revisionNumber},
      ];
}
