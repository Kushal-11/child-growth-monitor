import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import 'providers/auth_provider.dart';
import 'screens/assessment/assessment_screen.dart';
import 'screens/assessment/capture_screen.dart';
import 'screens/assessment/result_screen.dart';
import 'screens/auth/login_screen.dart';
import 'screens/children/child_detail_screen.dart';
import 'screens/children/children_list_screen.dart';
import 'screens/child_management/child_form_screen.dart';
import 'screens/child_management/manual_measurement_screen.dart';
import 'screens/settings/settings_screen.dart';

GoRouter buildRouter(Ref ref) {
  return GoRouter(
    initialLocation: '/',
    refreshListenable: _AuthListenable(ref),
    redirect: (context, state) {
      final status = ref.read(authProvider).status;
      final loc = state.matchedLocation;
      if (status == AuthStatus.unknown) {
        return loc == '/splash' ? null : '/splash';
      }
      // status resolved: never stay on splash
      final loggingIn = loc == '/login';
      if (status == AuthStatus.unauthenticated) {
        // Settings must be reachable pre-login so the server URL can be
        // configured before the first successful authentication.
        if (loggingIn || loc == '/settings') return null;
        return '/login';
      }
      // authenticated
      if (loggingIn || loc == '/splash') return '/';
      return null;
    },
    routes: [
      GoRoute(
        path: '/splash',
        builder: (c, s) => const Scaffold(
          body: Center(child: CircularProgressIndicator()),
        ),
      ),
      GoRoute(path: '/login', builder: (c, s) => const LoginScreen()),
      GoRoute(path: '/', builder: (c, s) => const AssessmentScreen()),
      GoRoute(
        path: '/capture/:role',
        builder: (c, s) => CaptureScreen(role: s.pathParameters['role']!),
      ),
      GoRoute(path: '/result', builder: (c, s) => const ResultScreen()),
      GoRoute(path: '/children', builder: (c, s) => const ChildrenListScreen()),
      GoRoute(path: '/children/new', builder: (c, s) => const ChildFormScreen()),
      GoRoute(
        path: '/children/:id',
        builder: (c, s) => ChildDetailScreen(childId: int.parse(s.pathParameters['id']!)),
      ),
      GoRoute(
        path: '/children/:id/edit',
        builder: (c, s) => ChildFormScreen(childId: int.parse(s.pathParameters['id']!)),
      ),
      GoRoute(
        path: '/children/:id/measure',
        builder: (c, s) => ManualMeasurementScreen(childId: int.parse(s.pathParameters['id']!)),
      ),
      GoRoute(path: '/settings', builder: (c, s) => const SettingsScreen()),
    ],
  );
}

/// Bridges Riverpod auth state changes to GoRouter's refresh mechanism.
class _AuthListenable extends ChangeNotifier {
  _AuthListenable(Ref ref) {
    ref.listen(authProvider, (_, __) => notifyListeners());
  }
}

final routerProvider = Provider<GoRouter>((ref) => buildRouter(ref));
