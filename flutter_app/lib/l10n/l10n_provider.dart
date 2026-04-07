import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'translations.dart';

const _prefsLangKey = 'app_lang';

class LocaleNotifier extends StateNotifier<String> {
  LocaleNotifier() : super('en') {
    _load();
  }

  Future<void> _load() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_prefsLangKey);
    if (saved != null && (saved == 'en' || saved == 'mr')) {
      state = saved;
    }
  }

  Future<void> toggle() async {
    state = state == 'en' ? 'mr' : 'en';
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefsLangKey, state);
  }
}

final localeProvider = StateNotifierProvider<LocaleNotifier, String>(
  (ref) => LocaleNotifier(),
);

/// Translate a key using the current locale.
/// Falls back to English if the key is missing for the current locale.
String t(String key, WidgetRef ref) {
  final lang = ref.watch(localeProvider);
  final entry = translations[key];
  if (entry == null) return key;
  return entry[lang] ?? entry['en'] ?? key;
}
