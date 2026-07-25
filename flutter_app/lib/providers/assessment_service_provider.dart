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
  try {
    await svc.load();
  } catch (error, stackTrace) {
    // The ML models are secondary screening aids. A missing, corrupt, or
    // unsupported TFLite asset must not block manual measurements, WHO
    // calculations, or the deterministic Poshan Setu classifier. Returning
    // the unloaded service lets AssessmentService's guarded predict() call
    // fall back while still surfacing diagnostics for support.
    // ignore: avoid_print
    print(
      'ML runtime unavailable; continuing without ML screening. '
      '$error\n$stackTrace',
    );
  }
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
