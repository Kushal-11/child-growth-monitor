import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/api_service.dart';
import 'auth_provider.dart';

const _developmentBaseUrl = 'http://10.0.2.2:8000';
const _productionBaseUrl = 'https://api.child-growth-monitor.org';
const _prefsBaseUrlKey = 'api_base_url';

const _approvedHosts = <String>{
  'localhost',
  'api.child-growth-monitor.org',
};

bool isAllowlistedHost(String baseUrl) {
  if (baseUrl.isEmpty) return false;
  try {
    final uri = Uri.parse(baseUrl);
    if (uri.scheme != 'http' && uri.scheme != 'https') return false;
    final host = uri.host;
    return _approvedHosts.contains(host) ||
        host == '127.0.0.1' ||
        host == '10.0.2.2' ||
        _isPrivateIpv4(host);
  } on FormatException {
    return false;
  }
}

bool _isPrivateIpv4(String host) {
  final octets = host.split('.');
  if (octets.length != 4) return false;
  final values = <int>[];
  for (final o in octets) {
    final v = int.tryParse(o);
    if (v == null || v < 0 || v > 255) return false;
    values.add(v);
  }
  return values[0] == 10 ||
      (values[0] == 172 && values[1] >= 16 && values[1] <= 31) ||
      (values[0] == 192 && values[1] == 168);
}

String effectiveBaseUrl(String inputUrl) {
  if (isAllowlistedHost(inputUrl)) return inputUrl;
  if (kDebugMode) return _developmentBaseUrl;
  return _productionBaseUrl;
}

Future<String> loadSavedBaseUrl() async {
  final prefs = await SharedPreferences.getInstance();
  final saved = prefs.getString(_prefsBaseUrlKey);
  if (saved != null && saved.isNotEmpty) return saved;
  return kDebugMode ? _developmentBaseUrl : _productionBaseUrl;
}

Future<void> saveBaseUrl(String url) async {
  final prefs = await SharedPreferences.getInstance();
  await prefs.setString(_prefsBaseUrlKey, url.trim());
}

/// Provider that holds the current base URL string.
final baseUrlProvider = StateProvider<String>(
  (ref) => kDebugMode ? _developmentBaseUrl : _productionBaseUrl,
);

/// Provider that creates an ApiService from the current base URL.
final apiProvider = Provider<ApiService>((ref) {
  final url = ref.watch(baseUrlProvider);
  final token = ref.watch(authProvider).token;
  return ApiService(baseUrl: effectiveBaseUrl(url), authToken: token);
});
