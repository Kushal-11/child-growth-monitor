import 'package:drift/drift.dart';

import 'visits_table.dart';

@TableIndex(
  name: 'ix_capture_assets_visit_role',
  columns: {#visitId, #role},
)
class CaptureAssets extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get assetUuid => text().unique()();
  IntColumn get visitId => integer().references(
        Visits,
        #id,
        onDelete: KeyAction.cascade,
      )();
  TextColumn get role => text()();
  TextColumn get localPath => text().nullable()();
  IntColumn get serverId => integer().nullable()();
  TextColumn get serverObjectId => text().nullable()();
  DateTimeColumn get capturedAt => dateTime()();
  IntColumn get selectedRank => integer().nullable()();
  RealColumn get poseScore => real().nullable()();
  RealColumn get coverageScore => real().nullable()();
  RealColumn get orientationScore => real().nullable()();
  RealColumn get sharpnessScore => real().nullable()();
  RealColumn get lightingScore => real().nullable()();
  RealColumn get overallScore => real().nullable()();
  TextColumn get qualityVerdict => text().nullable()();
  TextColumn get rejectionReason => text().nullable()();
  TextColumn get qualityThresholdVersion => text().nullable()();
  IntColumn get imageWidth => integer().nullable()();
  IntColumn get imageHeight => integer().nullable()();
  IntColumn get exifOrientation => integer().nullable()();
  IntColumn get displayOrientation => integer().nullable()();
  TextColumn get deviceCameraMetadataJson => text().nullable()();
  TextColumn get syncState => text().withDefault(const Constant('pending'))();
  DateTimeColumn get serverAcknowledgedAt => dateTime().nullable()();

  @override
  List<String> get customConstraints => [
        "CHECK (role IN ('front', 'side', 'back', 'arm_front', 'arm_side'))",
      ];
}
