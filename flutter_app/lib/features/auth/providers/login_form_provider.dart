import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../providers/auth_provider.dart';
import '../../../services/auth_service.dart';

class LoginFormState {
  const LoginFormState({
    this.username = '',
    this.password = '',
    this.passwordVisible = false,
    this.showValidationErrors = false,
    this.isSubmitting = false,
    this.error,
  });

  final String username;
  final String password;
  final bool passwordVisible;
  final bool showValidationErrors;
  final bool isSubmitting;
  final String? error;

  bool get isValid => username.trim().isNotEmpty && password.isNotEmpty;

  LoginFormState copyWith({
    String? username,
    String? password,
    bool? passwordVisible,
    bool? showValidationErrors,
    bool? isSubmitting,
    String? error,
    bool clearError = false,
  }) {
    return LoginFormState(
      username: username ?? this.username,
      password: password ?? this.password,
      passwordVisible: passwordVisible ?? this.passwordVisible,
      showValidationErrors: showValidationErrors ?? this.showValidationErrors,
      isSubmitting: isSubmitting ?? this.isSubmitting,
      error: clearError ? null : (error ?? this.error),
    );
  }
}

class LoginFormNotifier extends StateNotifier<LoginFormState> {
  LoginFormNotifier(this._ref) : super(const LoginFormState());

  final Ref _ref;

  void updateUsername(String value) {
    state = state.copyWith(username: value, clearError: true);
  }

  void updatePassword(String value) {
    state = state.copyWith(password: value, clearError: true);
  }

  void togglePasswordVisibility() {
    state = state.copyWith(passwordVisible: !state.passwordVisible);
  }

  Future<bool> submit() async {
    if (!state.isValid) {
      state = state.copyWith(showValidationErrors: true);
      return false;
    }

    state = state.copyWith(
      isSubmitting: true,
      showValidationErrors: false,
      clearError: true,
    );
    try {
      await _ref
          .read(authProvider.notifier)
          .login(state.username.trim(), state.password);
      if (mounted) state = state.copyWith(isSubmitting: false);
      return true;
    } on AuthException catch (error) {
      if (mounted) {
        state = state.copyWith(isSubmitting: false, error: error.message);
      }
      return false;
    } catch (error) {
      if (mounted) {
        state = state.copyWith(
          isSubmitting: false,
          error: 'Login failed: $error',
        );
      }
      return false;
    }
  }
}

final loginFormProvider =
    StateNotifierProvider.autoDispose<LoginFormNotifier, LoginFormState>((ref) {
  return LoginFormNotifier(ref);
});
