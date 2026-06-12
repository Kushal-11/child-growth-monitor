import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;

import 'local_auth.dart';

class AuthUser {
  AuthUser({required this.id, required this.username, required this.fullName, required this.role});

  final int id;
  final String username;
  final String fullName;
  final String role;

  factory AuthUser.fromJson(Map<String, dynamic> json) => AuthUser(
        id: json['id'] as int,
        username: json['username'] as String,
        fullName: json['full_name'] as String,
        role: json['role'] as String,
      );

  Map<String, dynamic> toJson() =>
      {'id': id, 'username': username, 'full_name': fullName, 'role': role};
}

class AuthLoginResult {
  AuthLoginResult({required this.token, required this.user});
  final String token;
  final AuthUser user;

  factory AuthLoginResult.fromJson(Map<String, dynamic> json) => AuthLoginResult(
        token: json['access_token'] as String,
        user: AuthUser.fromJson(json['user'] as Map<String, dynamic>),
      );
}

class AuthException implements Exception {
  AuthException(this.message, {this.statusCode});
  final String message;
  final int? statusCode;
  @override
  String toString() => message;
}

/// Handles login HTTP + secure persistence of the token & user.
class AuthService {
  AuthService({
    required this.baseUrl,
    FlutterSecureStorage? storage,
    http.Client? httpClient,
    Duration storageTimeout = const Duration(seconds: 5),
  })  : _storage = storage ?? const FlutterSecureStorage(),
        _client = httpClient ?? http.Client(),
        _storageTimeout = storageTimeout;

  final String baseUrl;
  final FlutterSecureStorage _storage;
  final http.Client _client;

  /// Upper bound on any single secure-storage operation. A platform keyring
  /// (e.g. the Android keystore) can block indefinitely; bounding every call
  /// guarantees login and session-restore can never hang the UI.
  final Duration _storageTimeout;

  static const _kToken = 'auth_token';
  static const _kUser = 'auth_user';
  static const Duration _timeout = Duration(seconds: 30);

  Future<AuthLoginResult> login(String username, String password) async {
    // Offline-first: a hardcoded field-test credential resolves locally with
    // no network call. Any other credential falls through to the backend.
    final local = LocalAuth.tryLogin(username, password);
    if (local != null) {
      await _persist(local);
      return local;
    }

    final uri = Uri.parse('$baseUrl/api/v1/auth/login');
    late final http.Response resp;
    try {
      resp = await _client
          .post(uri,
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({'username': username, 'password': password}))
          .timeout(_timeout);
    } on TimeoutException {
      throw AuthException('Login timed out. Check your connection.');
    } on http.ClientException catch (e) {
      throw AuthException('Network error during login: $e');
    }
    if (resp.statusCode == 200) {
      final result = AuthLoginResult.fromJson(jsonDecode(resp.body) as Map<String, dynamic>);
      await _persist(result);
      return result;
    }
    if (resp.statusCode == 401) {
      throw AuthException('Invalid username or password', statusCode: 401);
    }
    throw AuthException('Login failed (${resp.statusCode})', statusCode: resp.statusCode);
  }

  Future<void> _persist(AuthLoginResult result) async {
    // Persistence must never hang or fail the login. Secure storage can block
    // indefinitely on a misbehaving platform keyring, so each write is bounded.
    // On failure the session remains valid in memory for this app run; it is
    // simply not restored after a restart. This is surfaced (not silent) via a
    // logged warning rather than blocking the user on the login screen.
    try {
      await _storage
          .write(key: _kToken, value: result.token)
          .timeout(_storageTimeout);
      await _storage
          .write(key: _kUser, value: jsonEncode(result.user.toJson()))
          .timeout(_storageTimeout);
    } catch (e) {
      debugPrint('AuthService: secure-storage persist failed ($e); '
          'continuing with in-memory session only.');
    }
  }

  /// Reads a value from secure storage, bounded by [_storageTimeout]. Returns
  /// null on timeout or any storage error so a blocked keyring can never strand
  /// the app on the splash gate during restore.
  Future<String?> _readBounded(String key) async {
    try {
      return await _storage.read(key: key).timeout(_storageTimeout);
    } catch (e) {
      debugPrint('AuthService: secure-storage read failed ($e); '
          'treating as no cached session.');
      return null;
    }
  }

  Future<String?> readToken() => _readBounded(_kToken);

  Future<AuthUser?> readUser() async {
    final raw = await _readBounded(_kUser);
    if (raw == null) return null;
    return AuthUser.fromJson(jsonDecode(raw) as Map<String, dynamic>);
  }

  Future<void> logout() async {
    await _storage.delete(key: _kToken);
    await _storage.delete(key: _kUser);
  }
}
