import 'dart:async';
import 'dart:convert';

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
  })  : _storage = storage ?? const FlutterSecureStorage(),
        _client = httpClient ?? http.Client();

  final String baseUrl;
  final FlutterSecureStorage _storage;
  final http.Client _client;

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
    await _storage.write(key: _kToken, value: result.token);
    await _storage.write(key: _kUser, value: jsonEncode(result.user.toJson()));
  }

  Future<String?> readToken() => _storage.read(key: _kToken);

  Future<AuthUser?> readUser() async {
    final raw = await _storage.read(key: _kUser);
    if (raw == null) return null;
    return AuthUser.fromJson(jsonDecode(raw) as Map<String, dynamic>);
  }

  Future<void> logout() async {
    await _storage.delete(key: _kToken);
    await _storage.delete(key: _kUser);
  }
}
