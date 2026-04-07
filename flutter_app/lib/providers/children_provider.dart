import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/child.dart';
import '../models/child_detail.dart';
import 'api_provider.dart';

final childrenProvider = FutureProvider<List<ChildSummary>>((ref) async {
  final api = ref.watch(apiProvider);
  return api.getChildren();
});

final childDetailProvider =
    FutureProvider.family<ChildDetail, int>((ref, childId) async {
  final api = ref.watch(apiProvider);
  return api.getChildDetail(childId);
});
