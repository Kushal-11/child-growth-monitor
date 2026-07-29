import 'package:drift/drift.dart';

@TableIndex(
  name: 'ix_sync_outbox_owner_status_created',
  columns: {#ownerUserId, #status, #createdAt},
)
@TableIndex(
  name: 'ix_sync_outbox_visit_type',
  columns: {#visitUuid, #entityType},
)
class SyncOutbox extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get ownerUserId => integer()();
  TextColumn get visitUuid => text()();
  TextColumn get entityType => text()();
  TextColumn get entityUuid => text()();
  TextColumn get operation => text().withDefault(const Constant('upsert'))();
  TextColumn get dependencyEntityUuid => text().nullable()();
  TextColumn get payloadJson => text()();
  TextColumn get payloadChecksum => text()();
  TextColumn get status => text().withDefault(const Constant('pending'))();
  IntColumn get retryCount => integer().withDefault(const Constant(0))();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();
  DateTimeColumn get lastAttemptAt => dateTime().nullable()();
  DateTimeColumn get acknowledgedAt => dateTime().nullable()();
  TextColumn get acknowledgementPayloadJson => text().nullable()();
  TextColumn get errorMessage => text().nullable()();

  @override
  List<Set<Column>> get uniqueKeys => [
        {entityType, entityUuid},
      ];

  @override
  List<String> get customConstraints => [
        "CHECK (entity_type IN ('visit', 'capture_asset', 'camera_result', "
            "'measured_revision', 'media_deletion'))",
      ];
}
