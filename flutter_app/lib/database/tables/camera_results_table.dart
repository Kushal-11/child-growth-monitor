import 'package:drift/drift.dart';

import 'visits_table.dart';

@TableIndex(
  name: 'ix_camera_results_visit_version',
  columns: {#visitId, #version},
)
class CameraResults extends Table {
  IntColumn get id => integer().autoIncrement()();
  TextColumn get resultUuid => text().unique()();
  IntColumn get visitId => integer().references(
        Visits,
        #id,
        onDelete: KeyAction.cascade,
      )();
  IntColumn get version => integer()();
  TextColumn get supersedesResultUuid => text().nullable()();
  RealColumn get estimatedHeightCm => real().nullable()();
  RealColumn get estimatedWeightKg => real().nullable()();
  TextColumn get heightSource => text().nullable()();
  TextColumn get weightSource => text().nullable()();
  RealColumn get estimatedHaz => real().nullable()();
  RealColumn get estimatedWhz => real().nullable()();
  TextColumn get estimatedStuntingStatus => text().nullable()();
  TextColumn get estimatedWastingStatus => text().nullable()();
  TextColumn get experimentalOverallCategory => text().nullable()();
  TextColumn get componentProbabilitiesJson => text().nullable()();
  TextColumn get bodyProportionFeaturesJson => text().nullable()();
  TextColumn get captureQualitySummaryJson => text().nullable()();
  TextColumn get method => text()();
  TextColumn get modelVersion => text()();
  TextColumn get manifestChecksum => text()();
  TextColumn get trainingDataLabel => text()();
  BoolColumn get nonClinical => boolean().withDefault(const Constant(true))();
  DateTimeColumn get createdAt => dateTime().withDefault(currentDateAndTime)();

  @override
  List<Set<Column>> get uniqueKeys => [
        {visitId, version},
      ];

  @override
  List<String> get customConstraints => [
        'CHECK (non_clinical = 1)',
      ];
}
