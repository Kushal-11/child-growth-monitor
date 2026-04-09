import 'package:go_router/go_router.dart';

import 'screens/assessment/assessment_screen.dart';
import 'screens/assessment/result_screen.dart';
import 'screens/children/child_detail_screen.dart';
import 'screens/children/children_list_screen.dart';
import 'screens/settings/settings_screen.dart';

final appRouter = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(
      path: '/',
      builder: (context, state) => const AssessmentScreen(),
    ),
    GoRoute(
      path: '/result',
      builder: (context, state) => const ResultScreen(),
    ),
    GoRoute(
      path: '/children',
      builder: (context, state) => const ChildrenListScreen(),
    ),
    GoRoute(
      path: '/children/:id',
      builder: (context, state) {
        final id = int.parse(state.pathParameters['id']!);
        return ChildDetailScreen(childId: id);
      },
    ),
    GoRoute(
      path: '/settings',
      builder: (context, state) => const SettingsScreen(),
    ),
  ],
);
