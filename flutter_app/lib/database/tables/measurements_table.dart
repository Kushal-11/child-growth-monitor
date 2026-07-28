import 'package:drift/drift.dart';
import 'visits_table.dart';

class Measurements extends Table {
  IntColumn get id => integer().autoIncrement()();
  IntColumn get visitId => integer().unique().references(Visits, #id)();
  RealColumn get predictedHeightCm => real().nullable()();
  RealColumn get predictedWeightKg => real().nullable()();
  RealColumn get manualHeightCm => real().nullable()();
  RealColumn get manualWeightKg => real().nullable()();
  RealColumn get hazZscore => real().nullable()();
  RealColumn get whzZscore => real().nullable()();
  TextColumn get hazStatus => text().nullable()();
  TextColumn get whzStatus => text().nullable()();
  RealColumn get confidenceScore => real().nullable()();
  TextColumn get bodyBuild => text().nullable()();
  TextColumn get estimationMethod => text().nullable()();
  BoolColumn get sideViewUsed => boolean().withDefault(const Constant(false))();
  RealColumn get chestDepthCm => real().nullable()();
  RealColumn get abdDepthCm => real().nullable()();
  RealColumn get mlEstimatedWeightKg => real().nullable()();
  RealColumn get samProbability => real().nullable()();
  RealColumn get mamProbability => real().nullable()();
  RealColumn get normalProbability => real().nullable()();
  RealColumn get riskOverweightProbability => real().nullable()();
  RealColumn get overweightProbability => real().nullable()();
  TextColumn get wastingStatus => text().nullable()();
  RealColumn get muacCm => real().nullable()();
  TextColumn get muacStatus => text().nullable()();
  TextColumn get muacMethod => text().nullable()();
  RealColumn get bmiValue => real().nullable()();
  TextColumn get bmiStatus => text().nullable()();
  TextColumn get protocolStatus => text().nullable()();
  TextColumn get triggeredIndicators => text().nullable()();
  TextColumn get measurementMethods => text().nullable()();
}
