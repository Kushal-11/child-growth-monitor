import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../services/assessment_service.dart';
import '../services/image_storage_service.dart';
import '../services/measurement_service.dart';
import '../services/ml_inference_service.dart';
import '../services/nutrition_service.dart';
import '../services/pose_service.dart';
import '../services/pose_source.dart';
import '../services/who_data_service.dart';
import 'database_provider.dart';

final whoDataServiceProvider = FutureProvider<WhoDataService>((ref) async {
  final who = WhoDataService();
  await who.loadFromAssets();
  return who;
});

final poseServiceProvider = Provider<PoseService>((ref) {
  final svc = PoseService();
  ref.onDispose(svc.dispose);
  return svc;
});

final poseSourceProvider = Provider<PoseSource>(
    (ref) => PoseServiceSource(ref.watch(poseServiceProvider)));

final mlInferenceServiceProvider =
    FutureProvider<MlInferenceService>((ref) async {
  final svc = MlInferenceService();
  await svc.load();
  ref.onDispose(svc.dispose);
  return svc;
});

final imageStorageProvider =
    Provider<ImageStorageService>((ref) => ImageStorageService());

final assessmentServiceProvider =
    FutureProvider<AssessmentService>((ref) async {
  final who = await ref.watch(whoDataServiceProvider.future);
  final ml = await ref.watch(mlInferenceServiceProvider.future);
  final storage = ref.watch(imageStorageProvider);

  return AssessmentService(
    db: ref.watch(databaseProvider),
    childDao: ref.watch(childDaoProvider),
    visitDao: ref.watch(visitDaoProvider),
    syncQueueDao: ref.watch(syncQueueDaoProvider),
    pose: ref.watch(poseSourceProvider),
    measurement: MeasurementService(who),
    nutrition: NutritionService(who),
    who: who,
    ml: ml,
    persistImage: storage.persist,
  );
});
