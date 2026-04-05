# Flutter Web Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the Flutter app's UI layer to match the Jinja2 web version — 4 screens, Riverpod state management, GoRouter navigation, fl_chart growth charts, and EN/MR i18n.

**Architecture:** Keep existing models (`assessment_result.dart`, `child.dart`, `child_detail.dart`) and `api_service.dart` untouched. Replace the monolithic `assessment_screen.dart` with 6 new screen files, 3 providers, 2 i18n files, a router, and a rewritten `main.dart`.

**Tech Stack:** Flutter, Riverpod (`flutter_riverpod`), GoRouter (`go_router`), fl_chart, image_picker, shared_preferences, http, intl.

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `flutter_app/pubspec.yaml` | Add flutter_riverpod, go_router, fl_chart |
| Create | `flutter_app/lib/l10n/translations.dart` | EN/MR string maps |
| Create | `flutter_app/lib/l10n/l10n_provider.dart` | Riverpod locale provider + `t()` helper |
| Create | `flutter_app/lib/providers/api_provider.dart` | ApiService singleton provider |
| Create | `flutter_app/lib/providers/children_provider.dart` | Children list + detail providers |
| Create | `flutter_app/lib/providers/assessment_provider.dart` | Assessment result state provider |
| Create | `flutter_app/lib/screens/shared/status_badge.dart` | Reusable color-coded status badge |
| Create | `flutter_app/lib/screens/shared/app_scaffold.dart` | Shared AppBar + BottomNav scaffold |
| Create | `flutter_app/lib/router.dart` | GoRouter config with 4 routes |
| Create | `flutter_app/lib/screens/assessment/assessment_screen.dart` | Assessment form |
| Create | `flutter_app/lib/screens/assessment/result_screen.dart` | Rich result display |
| Create | `flutter_app/lib/screens/children/children_list_screen.dart` | Searchable children table |
| Create | `flutter_app/lib/screens/children/child_detail_screen.dart` | Profile + chart + history |
| Rewrite | `flutter_app/lib/main.dart` | ProviderScope + GoRouter MaterialApp |
| Delete | `flutter_app/lib/screens/assessment_screen.dart` | Old monolith (replaced) |

---

### Task 1: Add Dependencies

**Files:**
- Modify: `flutter_app/pubspec.yaml`

- [ ] **Step 1: Add new dependencies to pubspec.yaml**

Add these under `dependencies:` (after the existing `shared_preferences` line):

```yaml
  flutter_riverpod: ^2.6.1
  go_router: ^14.8.1
  fl_chart: ^0.70.2
```

The full dependencies section should be:

```yaml
dependencies:
  flutter:
    sdk: flutter
  http: ^1.2.2
  image_picker: ^1.1.2
  intl: ^0.19.0
  shared_preferences: ^2.3.2
  flutter_riverpod: ^2.6.1
  go_router: ^14.8.1
  fl_chart: ^0.70.2
```

Note: `riverpod_annotation`, `riverpod_generator`, and `build_runner` are NOT needed — we'll use plain Riverpod providers without codegen to keep things simple.

- [ ] **Step 2: Run flutter pub get**

```bash
cd flutter_app && flutter pub get
```

Expected: `Resolving dependencies...` followed by `Got dependencies!` — no errors.

- [ ] **Step 3: Commit**

```bash
git add flutter_app/pubspec.yaml flutter_app/pubspec.lock
git commit -m "Add flutter_riverpod, go_router, fl_chart dependencies"
```

---

### Task 2: Translations (i18n)

**Files:**
- Create: `flutter_app/lib/l10n/translations.dart`
- Create: `flutter_app/lib/l10n/l10n_provider.dart`

- [ ] **Step 1: Create translations.dart**

Port the Python `TRANSLATIONS` dict from `app/web/translations.py` to Dart. Create `flutter_app/lib/l10n/translations.dart`:

```dart
const Map<String, Map<String, String>> translations = {
  'app_title': {'en': 'SNEH Growth Monitor', 'mr': 'स्नेह वाढ निरीक्षक'},
  'nav_assess': {'en': 'Assess', 'mr': 'मूल्यांकन'},
  'nav_children': {'en': 'Children', 'mr': 'मुले'},
  'footer_tagline': {
    'en': 'SNEH Growth Monitor — WHO Standard-Based Assessment',
    'mr': 'स्नेह वाढ निरीक्षक — WHO मानक आधारित मूल्यांकन',
  },
  'lang_en': {'en': 'English', 'mr': 'English'},
  'lang_mr': {'en': 'मराठी', 'mr': 'मराठी'},
  // Index / Assessment form
  'assess_heading': {'en': 'Child Growth Assessment', 'mr': 'बाल वाढ मूल्यांकन'},
  'assess_subtitle': {
    'en':
        'Upload a photo to automatically estimate height, weight, and MUAC, '
        'then classify stunting and wasting using WHO standards.',
    'mr':
        'उंची, वजन आणि MUAC स्वयंचलितपणे अंदाज करण्यासाठी फोटो अपलोड करा, '
        'नंतर WHO मानकांनुसार कुपोषण आणि वाढ वर्गीकरण करा.',
  },
  'front_view_photo': {'en': 'Front View Photo', 'mr': 'समोरून फोटो'},
  'tip_front_1': {
    'en': 'Child standing upright, facing the camera',
    'mr': 'मूल सरळ उभे, कॅमेऱ्याकडे तोंड',
  },
  'tip_front_2': {
    'en': 'Full body visible — head to feet',
    'mr': 'संपूर्ण शरीर दिसावे — डोक्यापासून पायांपर्यंत',
  },
  'tip_front_3': {
    'en': '1–2 metres from camera, good lighting',
    'mr': 'कॅमेऱ्यापासून १–२ मीटर, चांगली प्रकाशयोजना',
  },
  'tip_front_4': {
    'en': 'Plain background preferred',
    'mr': 'साधी पार्श्वभूमी पसंत',
  },
  'side_view': {'en': 'Side View', 'mr': 'बाजूचे दृश्य'},
  'side_accuracy_badge': {
    'en': '+30–40% weight accuracy',
    'mr': '+३०–४०% वजन अचूकता',
  },
  'side_view_help': {
    'en': 'Child standing sideways — left or right side toward camera',
    'mr': 'मूल बाजूने उभे — डावी किंवा उजवी बाजू कॅमेऱ्याकडे',
  },
  'back_view': {'en': 'Back View', 'mr': 'मागचे दृश्य'},
  'back_view_help': {
    'en': 'Child facing away, full body visible',
    'mr': 'मूल पाठ दाखवून, संपूर्ण शरीर दिसावे',
  },
  'optional_label': {'en': 'Optional', 'mr': 'ऐच्छिक'},
  'child_information': {'en': 'Child Information', 'mr': 'मुलाची माहिती'},
  'child_name': {'en': 'Child Name', 'mr': 'मुलाचे नाव'},
  'sex': {'en': 'Sex', 'mr': 'लिंग'},
  'male': {'en': 'Male', 'mr': 'मुलगा'},
  'female': {'en': 'Female', 'mr': 'मुलगी'},
  'age_months': {'en': 'Age (months)', 'mr': 'वय (महिने)'},
  'placeholder_age_months': {'en': '0 – 59', 'mr': '० – ५९'},
  'toggle_dob': {
    'en': 'Or enter exact date of birth',
    'mr': 'किंवा जन्मतारीख प्रविष्ट करा',
  },
  'date_of_birth': {'en': 'Date of Birth', 'mr': 'जन्मतारीख'},
  'toggle_age_months': {
    'en': 'Or enter age in months',
    'mr': 'किंवा वय महिन्यांत प्रविष्ट करा',
  },
  'optional_measurements': {'en': 'Optional Measurements', 'mr': 'ऐच्छिक मोजमाप'},
  'optional_measurements_note': {
    'en': '— improve accuracy if available',
    'mr': '— उपलब्ध असल्यास अचूकता वाढते',
  },
  'weight_kg': {'en': 'Weight (kg)', 'mr': 'वजन (किग्रॅ)'},
  'weight_placeholder': {'en': 'e.g. 11.5', 'mr': 'उदा. ११.५'},
  'weight_help': {'en': 'From weighing scale', 'mr': 'वजन मापक यंत्रावरून'},
  'muac_cm': {'en': 'MUAC (cm)', 'mr': 'MUAC (सेमी)'},
  'muac_placeholder': {'en': 'e.g. 13.5', 'mr': 'उदा. १३.५'},
  'muac_help': {'en': 'MUAC tape measurement', 'mr': 'MUAC टेप मोजमाप'},
  'height': {'en': 'Height', 'mr': 'उंची'},
  'height_placeholder': {'en': 'Optional', 'mr': 'ऐच्छिक'},
  'unit_cm': {'en': 'cm', 'mr': 'सेमी'},
  'unit_inch': {'en': 'inch', 'mr': 'इंच'},
  'height_fallback': {
    'en': 'Fallback if image fails',
    'mr': 'प्रतिमा अयशस्वी असल्यास पर्याय',
  },
  'guardian_name': {'en': 'Guardian Name', 'mr': 'पालक / पालकाचे नाव'},
  'location_clinic': {'en': 'Location / Clinic', 'mr': 'ठिकाण / दवाखाना'},
  'placeholder_optional': {'en': 'Optional', 'mr': 'ऐच्छिक'},
  'run_assessment': {'en': 'Run Assessment', 'mr': 'मूल्यांकन चालवा'},
  'processing': {'en': 'Processing…', 'mr': 'प्रक्रिया सुरू…'},
  'err_invalid_date': {
    'en': 'Invalid date format. Please use the date picker.',
    'mr': 'अवैध तारीख स्वरूप. कृपया तारीख निवडक वापरा.',
  },
  'age_required_feedback': {
    'en': "Please enter the child's age (in months) or date of birth.",
    'mr': 'कृपया मुलाचे वय (महिन्यांत) किंवा जन्मतारीख प्रविष्ट करा.',
  },
  // Children list
  'registered_children': {'en': 'Registered Children', 'mr': 'नोंदणीकृत मुले'},
  'search_children_placeholder': {
    'en': 'Search by name…',
    'mr': 'नावाने शोधा…',
  },
  'search_no_results': {
    'en': 'No children match your search.',
    'mr': 'तुमच्या शोधाशी जुळणारी मुले नाहीत.',
  },
  'th_name': {'en': 'Name', 'mr': 'नाव'},
  'th_dob': {'en': 'Date of Birth', 'mr': 'जन्मतारीख'},
  'th_sex': {'en': 'Sex', 'mr': 'लिंग'},
  'th_guardian': {'en': 'Guardian', 'mr': 'पालक'},
  'th_location': {'en': 'Location', 'mr': 'ठिकाण'},
  'th_visits': {'en': 'Visits', 'mr': 'भेटी'},
  'view': {'en': 'View', 'mr': 'पहा'},
  'empty_children': {
    'en': 'No children registered yet.',
    'mr': 'अद्याप कोणतीही मूल नोंदणी नाही.',
  },
  'empty_children_link': {'en': 'Run an assessment', 'mr': 'मूल्यांकन चालवा'},
  'empty_children_suffix': {
    'en': 'to get started.',
    'mr': 'सुरुवात करण्यासाठी.',
  },
  // Child detail
  'child_profile': {'en': 'Child Profile', 'mr': 'मुलाचे प्रोफाइल'},
  'label_dob': {'en': 'DOB:', 'mr': 'जन्मतारीख:'},
  'label_sex': {'en': 'Sex:', 'mr': 'लिंग:'},
  'label_guardian': {'en': 'Guardian:', 'mr': 'पालक:'},
  'label_location': {'en': 'Location:', 'mr': 'ठिकाण:'},
  'total_visits': {'en': 'Total Visits:', 'mr': 'एकूण भेटी:'},
  'visit_history': {'en': 'Visit History', 'mr': 'भेटीचा इतिहास'},
  'th_date': {'en': 'Date', 'mr': 'तारीख'},
  'th_age_months': {'en': 'Age (months)', 'mr': 'वय (महिने)'},
  'th_height_cm': {'en': 'Height (cm)', 'mr': 'उंची (सेमी)'},
  'th_weight_kg': {'en': 'Weight (kg)', 'mr': 'वजन (किग्रॅ)'},
  'th_haz_status': {'en': 'HAZ Status', 'mr': 'HAZ स्थिती'},
  'th_whz_status': {'en': 'WHZ Status', 'mr': 'WHZ स्थिती'},
  'no_measurement_data': {'en': 'No measurement data', 'mr': 'मोजमाप माहिती नाही'},
  'no_visits_yet': {
    'en': 'No visits recorded yet.',
    'mr': 'अद्याप कोणतीही भेट नोंदवलेली नाही.',
  },
  'back_to_children': {'en': 'Back to Children', 'mr': 'मुलांकडे परत'},
  'child_not_found': {'en': 'Child not found.', 'mr': 'मूल सापडले नाही.'},
  'growth_chart_title': {'en': 'Growth over visits', 'mr': 'भेटींनुसार वाढ'},
  'chart_height_cm': {'en': 'Height (cm)', 'mr': 'उंची (सेमी)'},
  'chart_weight_kg': {'en': 'Weight (kg)', 'mr': 'वजन (किग्रॅ)'},
  // Result page
  'banner_sam_title': {
    'en': 'Severe Acute Malnutrition (SAM)',
    'mr': 'तीव्र अल्पकालीन कुपोषण (SAM)',
  },
  'banner_sam_msg': {
    'en': 'Immediate referral for therapeutic feeding is required.',
    'mr': 'उपचारात्मक आहारासाठी तात्काळ रेफरल आवश्यक आहे.',
  },
  'banner_mam_title': {
    'en': 'Moderate Acute Malnutrition (MAM)',
    'mr': 'मध्यम तीव्र कुपोषण (MAM)',
  },
  'banner_mam_msg': {
    'en': 'Supplementary feeding and close monitoring recommended.',
    'mr': 'पूरक आहार आणि जवळून निरीक्षण शिफारस केले जाते.',
  },
  'banner_stunted_msg': {
    'en': 'Child shows signs of chronic undernutrition. Follow-up recommended.',
    'mr': 'दीर्घकालीन कुपोषणाची लक्षणे दिसतात. पाठपुरावा शिफारस केले जाते.',
  },
  'banner_normal_title': {
    'en': 'Normal Nutritional Status',
    'mr': 'सामान्य पोषण स्थिती',
  },
  'banner_normal_msg': {
    'en': 'No acute malnutrition detected. Continue routine monitoring.',
    'mr': 'तीव्र कुपोषण आढळले नाही. नियमित निरीक्षण सुरू ठेवा.',
  },
  'banner_unknown_title': {
    'en': 'Status Could Not Be Determined',
    'mr': 'स्थिती ठरवता आली नाही',
  },
  'banner_unknown_msg': {
    'en': 'Provide weight and height if possible for a complete assessment.',
    'mr': 'पूर्ण मूल्यांकनासाठी शक्य असल्यास वजन आणि उंची द्या.',
  },
  'months_unit': {'en': 'months', 'mr': 'महिने'},
  'pose_confidence': {'en': 'Pose confidence', 'mr': 'पोझ विश्वास'},
  'metric_height': {'en': 'Height', 'mr': 'उंची'},
  'metric_weight': {'en': 'Weight', 'mr': 'वजन'},
  'metric_muac': {'en': 'MUAC', 'mr': 'MUAC'},
  'badge_image': {'en': 'Image', 'mr': 'प्रतिमा'},
  'badge_manual': {'en': 'Manual', 'mr': 'हस्तलिखित'},
  'badge_undetected': {'en': 'Undetected', 'mr': 'आढळले नाही'},
  'badge_side_view_ok': {'en': 'Side view ✓', 'mr': 'बाजू दृश्य ✓'},
  'badge_estimated': {'en': 'Estimated', 'mr': 'अंदाजित'},
  'badge_na': {'en': 'N/A', 'mr': 'लागू नाही'},
  'badge_tape': {'en': 'Tape', 'mr': 'टेप'},
  'badge_est': {'en': 'Est.', 'mr': 'अंदा.'},
  'chest_depth': {'en': 'Chest depth:', 'mr': 'छातीची खोली:'},
  'abd_depth': {'en': 'Abd:', 'mr': 'पोट:'},
  'ml_wasting_title': {'en': 'ML Wasting Detection', 'mr': 'ML कुपोषण शोध'},
  'ml_wasting_sub': {
    'en': '— camera-based body proportions',
    'mr': '— कॅमेरा आधारित शरीर प्रमाण',
  },
  'sam_probability': {'en': 'SAM probability', 'mr': 'SAM संभाव्यता'},
  'mam_probability': {'en': 'MAM probability', 'mr': 'MAM संभाव्यता'},
  'normal_probability': {'en': 'Normal probability', 'mr': 'सामान्य संभाव्यता'},
  'ml_estimated_weight': {'en': 'ML estimated weight:', 'mr': 'ML अंदाजित वजन:'},
  'muac_note_strong': {'en': 'MUAC note:', 'mr': 'MUAC टीप:'},
  'muac_note_text': {
    'en':
        'Value estimated from WHZ — not a direct tape measurement. '
        'Confirm with a physical MUAC tape for clinical decisions.',
    'mr':
        'WHZ वरून अंदाजित मूल्य — थेट टेप मोजमाप नाही. '
        'Clinical निर्णयांसाठी भौतिक MUAC टेपने पुष्टी करा.',
  },
  'assess_another': {'en': 'Assess Another Child', 'mr': 'दुसऱ्या मुलाचे मूल्यांकन'},
  'view_all_children': {'en': 'View All Children', 'mr': 'सर्व मुले पहा'},
  'print_report': {'en': 'Print Report', 'mr': 'अहवाल छापा'},
  'age_outside_muac': {
    'en': 'Age outside 6–59m',
    'mr': 'वय ६–५९ महिन्यांच्या बाहेर',
  },
  'muac_range_sam': {'en': '<11.5', 'mr': '<११.५'},
  'muac_range_mam': {'en': '11.5–12.5', 'mr': '११.५–१२.५'},
  'muac_range_normal': {'en': '≥12.5', 'mr': '≥१२.५'},
  'required_field': {'en': 'Required', 'mr': 'आवश्यक'},
  'use_date_format': {'en': 'Use YYYY-MM-DD', 'mr': 'YYYY-MM-DD वापरा'},
  'dob_future_error': {
    'en': 'DOB cannot be in the future',
    'mr': 'जन्मतारीख भविष्यात असू शकत नाही',
  },
  'positive_number_error': {
    'en': 'Must be a positive number',
    'mr': 'धन संख्या असणे आवश्यक',
  },
  'front_image_required': {
    'en': 'Please select a front image.',
    'mr': 'कृपया समोरचा फोटो निवडा.',
  },
  'height_both_error': {
    'en': 'Please provide either height in cm OR height value/unit, not both.',
    'mr': 'कृपया उंची सेमी मध्ये किंवा उंची मूल्य/एकक द्या, दोन्ही नाही.',
  },
  'capture': {'en': 'Capture', 'mr': 'कॅप्चर'},
  'gallery': {'en': 'Gallery', 'mr': 'गॅलरी'},
  'not_selected': {'en': 'Not selected', 'mr': 'निवडलेले नाही'},
  'backend_healthy': {'en': 'Backend is healthy', 'mr': 'बॅकएंड चालू आहे'},
  'backend_unhealthy': {
    'en': 'Backend health check failed',
    'mr': 'बॅकएंड आरोग्य तपासणी अयशस्वी',
  },
  'api_base_url': {'en': 'FastAPI Base URL', 'mr': 'FastAPI बेस URL'},
  'check_health': {'en': 'Check Health', 'mr': 'आरोग्य तपासा'},
};
```

- [ ] **Step 2: Create l10n_provider.dart**

Create `flutter_app/lib/l10n/l10n_provider.dart`:

```dart
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
```

- [ ] **Step 3: Verify files compile**

```bash
cd flutter_app && flutter analyze lib/l10n/
```

Expected: `No issues found!`

- [ ] **Step 4: Commit**

```bash
git add flutter_app/lib/l10n/
git commit -m "Add i18n translations and locale provider (EN/MR)"
```

---

### Task 3: Riverpod Providers

**Files:**
- Create: `flutter_app/lib/providers/api_provider.dart`
- Create: `flutter_app/lib/providers/children_provider.dart`
- Create: `flutter_app/lib/providers/assessment_provider.dart`

- [ ] **Step 1: Create api_provider.dart**

Create `flutter_app/lib/providers/api_provider.dart`:

```dart
import 'package:flutter/foundation.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/api_service.dart';

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
/// Screens can update this with ref.read(baseUrlProvider.notifier).state = ...
final baseUrlProvider = StateProvider<String>(
  (ref) => kDebugMode ? _developmentBaseUrl : _productionBaseUrl,
);

/// Provider that creates an ApiService from the current base URL.
final apiProvider = Provider<ApiService>((ref) {
  final url = ref.watch(baseUrlProvider);
  return ApiService(baseUrl: effectiveBaseUrl(url));
});
```

- [ ] **Step 2: Create children_provider.dart**

Create `flutter_app/lib/providers/children_provider.dart`:

```dart
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
```

- [ ] **Step 3: Create assessment_provider.dart**

Create `flutter_app/lib/providers/assessment_provider.dart`:

```dart
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/assessment_result.dart';

/// Holds the latest assessment result. Set after successful submission,
/// read by ResultScreen. Cleared when starting a new assessment.
final assessmentResultProvider = StateProvider<AssessmentResult?>((ref) => null);
```

- [ ] **Step 4: Verify files compile**

```bash
cd flutter_app && flutter analyze lib/providers/
```

Expected: `No issues found!`

- [ ] **Step 5: Commit**

```bash
git add flutter_app/lib/providers/
git commit -m "Add Riverpod providers for API, children, and assessment state"
```

---

### Task 4: Shared Widgets (StatusBadge + AppScaffold)

**Files:**
- Create: `flutter_app/lib/screens/shared/status_badge.dart`
- Create: `flutter_app/lib/screens/shared/app_scaffold.dart`

- [ ] **Step 1: Create status_badge.dart**

Create `flutter_app/lib/screens/shared/status_badge.dart`:

```dart
import 'package:flutter/material.dart';

class StatusBadge extends StatelessWidget {
  const StatusBadge({super.key, required this.status});

  final String? status;

  @override
  Widget build(BuildContext context) {
    final label = status ?? 'Unknown';
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: _color(label),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        label,
        style: const TextStyle(color: Colors.white, fontSize: 12),
      ),
    );
  }

  static Color _color(String status) {
    switch (status.toLowerCase()) {
      case 'normal':
        return Colors.green;
      case 'stunted':
        return Colors.amber.shade700;
      case 'severely stunted':
      case 'sev. stunted':
        return Colors.orange;
      case 'mam':
      case 'at risk':
      case 'at risk (mam)':
        return Colors.orange;
      case 'sam':
      case 'severe':
        return Colors.red;
      case 'overweight':
      case 'obese':
        return Colors.purple;
      default:
        return Colors.grey;
    }
  }
}
```

- [ ] **Step 2: Create app_scaffold.dart**

Create `flutter_app/lib/screens/shared/app_scaffold.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../l10n/l10n_provider.dart';

class AppScaffold extends ConsumerWidget {
  const AppScaffold({
    super.key,
    required this.child,
    required this.currentIndex,
  });

  final Widget child;
  final int currentIndex;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return Scaffold(
      appBar: AppBar(
        title: Text(t('app_title', ref)),
        actions: [
          TextButton(
            onPressed: () => ref.read(localeProvider.notifier).toggle(),
            child: Text(
              ref.watch(localeProvider) == 'en'
                  ? t('lang_mr', ref)
                  : t('lang_en', ref),
              style: TextStyle(
                color: Theme.of(context).colorScheme.onPrimary,
              ),
            ),
          ),
        ],
      ),
      body: child,
      bottomNavigationBar: NavigationBar(
        selectedIndex: currentIndex,
        onDestinationSelected: (index) {
          switch (index) {
            case 0:
              context.go('/');
            case 1:
              context.go('/children');
          }
        },
        destinations: [
          NavigationDestination(
            icon: const Icon(Icons.assessment),
            label: t('nav_assess', ref),
          ),
          NavigationDestination(
            icon: const Icon(Icons.people),
            label: t('nav_children', ref),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 3: Verify files compile**

```bash
cd flutter_app && flutter analyze lib/screens/shared/
```

Expected: `No issues found!`

- [ ] **Step 4: Commit**

```bash
git add flutter_app/lib/screens/shared/
git commit -m "Add StatusBadge and AppScaffold shared widgets"
```

---

### Task 5: Router

**Files:**
- Create: `flutter_app/lib/router.dart`

- [ ] **Step 1: Create router.dart**

Create `flutter_app/lib/router.dart`:

```dart
import 'package:go_router/go_router.dart';

import 'screens/assessment/assessment_screen.dart';
import 'screens/assessment/result_screen.dart';
import 'screens/children/child_detail_screen.dart';
import 'screens/children/children_list_screen.dart';

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
  ],
);
```

Note: This file will not compile until the screen files exist (Tasks 6-9). That's expected — we'll create stub screens next, then fill them in.

- [ ] **Step 2: Commit**

```bash
git add flutter_app/lib/router.dart
git commit -m "Add GoRouter configuration with 4 routes"
```

---

### Task 6: AssessmentScreen (Form)

**Files:**
- Create: `flutter_app/lib/screens/assessment/assessment_screen.dart`

- [ ] **Step 1: Create assessment_screen.dart**

Create `flutter_app/lib/screens/assessment/assessment_screen.dart`:

```dart
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';
import 'package:intl/intl.dart';

import '../../l10n/l10n_provider.dart';
import '../../providers/api_provider.dart';
import '../../providers/assessment_provider.dart';
import '../../providers/children_provider.dart';
import '../shared/app_scaffold.dart';

class AssessmentScreen extends ConsumerStatefulWidget {
  const AssessmentScreen({super.key});

  @override
  ConsumerState<AssessmentScreen> createState() => _AssessmentScreenState();
}

class _AssessmentScreenState extends ConsumerState<AssessmentScreen> {
  final _formKey = GlobalKey<FormState>();
  final _baseUrlController = TextEditingController();
  final _childNameController = TextEditingController();
  final _dobController = TextEditingController();
  final _ageMonthsController = TextEditingController();
  final _weightController = TextEditingController();
  final _heightValueController = TextEditingController();
  final _muacController = TextEditingController();
  final _guardianController = TextEditingController();
  final _locationController = TextEditingController();

  final ImagePicker _picker = ImagePicker();

  String _sex = 'M';
  String _heightUnit = 'cm';
  bool _useDob = true;
  bool _loading = false;
  bool? _healthy;
  String? _error;

  XFile? _frontImage;
  XFile? _sideImage;
  XFile? _backImage;

  @override
  void initState() {
    super.initState();
    _dobController.text = DateFormat('yyyy-MM-dd').format(
      DateTime.now().subtract(const Duration(days: 365 * 3)),
    );
    _initBaseUrl();
  }

  Future<void> _initBaseUrl() async {
    final saved = await loadSavedBaseUrl();
    if (!mounted) return;
    _baseUrlController.text = saved;
    ref.read(baseUrlProvider.notifier).state = saved;
  }

  @override
  void dispose() {
    _baseUrlController.dispose();
    _childNameController.dispose();
    _dobController.dispose();
    _ageMonthsController.dispose();
    _weightController.dispose();
    _heightValueController.dispose();
    _muacController.dispose();
    _guardianController.dispose();
    _locationController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source, String role) async {
    final file = await _picker.pickImage(source: source, imageQuality: 90);
    if (!mounted || file == null) return;
    setState(() {
      switch (role) {
        case 'front':
          _frontImage = file;
        case 'side':
          _sideImage = file;
        case 'back':
          _backImage = file;
      }
    });
  }

  Future<void> _selectDob() async {
    final initial = DateTime.tryParse(_dobController.text) ?? DateTime.now();
    final selected = await showDatePicker(
      context: context,
      initialDate: initial,
      firstDate: DateTime(2000),
      lastDate: DateTime.now(),
    );
    if (!mounted || selected == null) return;
    setState(() {
      _dobController.text = DateFormat('yyyy-MM-dd').format(selected);
      // Sync age months
      final days = DateTime.now().difference(selected).inDays;
      _ageMonthsController.text = (days / 30.4375).toStringAsFixed(0);
    });
  }

  void _onAgeMonthsChanged(String value) {
    final months = double.tryParse(value);
    if (months == null || months < 0) return;
    final dob = DateTime.now().subtract(Duration(days: (months * 30.4375).round()));
    _dobController.text = DateFormat('yyyy-MM-dd').format(dob);
  }

  String _resolvedDob() {
    if (_useDob) return _dobController.text.trim();
    final months = double.tryParse(_ageMonthsController.text.trim());
    if (months != null && months >= 0) {
      final dob =
          DateTime.now().subtract(Duration(days: (months * 30.4375).round()));
      return DateFormat('yyyy-MM-dd').format(dob);
    }
    return _dobController.text.trim();
  }

  Future<void> _checkHealth() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final url = _baseUrlController.text.trim();
      await saveBaseUrl(url);
      ref.read(baseUrlProvider.notifier).state = url;
      final healthy = await ref.read(apiProvider).checkHealth();
      if (!mounted) return;
      setState(() => _healthy = healthy);
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _healthy = false;
        _error = e.toString();
      });
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  Future<void> _submit() async {
    if (!_formKey.currentState!.validate()) return;
    if (_frontImage == null) {
      setState(() => _error = t('front_image_required', ref));
      return;
    }

    final heightValue = double.tryParse(_heightValueController.text.trim());
    double? heightCm;
    if (heightValue != null) {
      heightCm = _heightUnit == 'inch' ? heightValue * 2.54 : heightValue;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final url = _baseUrlController.text.trim();
      await saveBaseUrl(url);
      ref.read(baseUrlProvider.notifier).state = url;

      final result = await ref.read(apiProvider).submitAssessment(
            frontImagePath: _frontImage!.path,
            sideImagePath: _sideImage?.path,
            backImagePath: _backImage?.path,
            childName: _childNameController.text.trim(),
            dateOfBirth: _resolvedDob(),
            sex: _sex,
            weightKg: double.tryParse(_weightController.text.trim()),
            heightCm: heightCm,
            muacCm: double.tryParse(_muacController.text.trim()),
            guardianName: _guardianController.text.trim(),
            location: _locationController.text.trim(),
          );
      if (!mounted) return;
      ref.read(assessmentResultProvider.notifier).state = result;
      // Invalidate children list so it refreshes when viewed
      ref.invalidate(childrenProvider);
      context.go('/result');
    } catch (e) {
      if (!mounted) return;
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return AppScaffold(
      currentIndex: 0,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Header
              Text(
                t('assess_heading', ref),
                style: Theme.of(context).textTheme.headlineSmall,
              ),
              const SizedBox(height: 4),
              Text(
                t('assess_subtitle', ref),
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 16),

              // Debug: API URL
              if (kDebugMode) ...[
                TextFormField(
                  controller: _baseUrlController,
                  decoration: InputDecoration(
                    labelText: t('api_base_url', ref),
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.check_circle_outline),
                      onPressed: _loading ? null : _checkHealth,
                      tooltip: t('check_health', ref),
                    ),
                  ),
                ),
                if (_healthy != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 4, bottom: 8),
                    child: Row(
                      children: [
                        Icon(
                          _healthy! ? Icons.check_circle : Icons.error,
                          color: _healthy! ? Colors.green : Colors.red,
                          size: 16,
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _healthy!
                              ? t('backend_healthy', ref)
                              : t('backend_unhealthy', ref),
                          style: TextStyle(
                            color: _healthy! ? Colors.green : Colors.red,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                const SizedBox(height: 8),
              ],

              // === IMAGES ===
              _sectionHeader(t('front_view_photo', ref), required: true),
              _photoGuidanceTips(),
              const SizedBox(height: 8),
              _imagePickerRow('front', _frontImage),
              const SizedBox(height: 16),

              _sectionHeader(
                '${t('side_view', ref)} (${t('optional_label', ref)})',
              ),
              Padding(
                padding: const EdgeInsets.only(bottom: 4),
                child: Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
                  decoration: BoxDecoration(
                    color: Colors.teal.shade50,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    t('side_accuracy_badge', ref),
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.teal.shade800,
                    ),
                  ),
                ),
              ),
              Text(
                t('side_view_help', ref),
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 4),
              _imagePickerRow('side', _sideImage),
              const SizedBox(height: 16),

              _sectionHeader(
                '${t('back_view', ref)} (${t('optional_label', ref)})',
              ),
              Text(
                t('back_view_help', ref),
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 4),
              _imagePickerRow('back', _backImage),

              const Divider(height: 32),

              // === CHILD INFO ===
              _sectionHeader(t('child_information', ref)),
              const SizedBox(height: 8),
              TextFormField(
                controller: _childNameController,
                decoration: InputDecoration(
                  labelText: '${t('child_name', ref)} *',
                  border: const OutlineInputBorder(),
                ),
                maxLength: 100,
                validator: (v) =>
                    (v == null || v.trim().isEmpty) ? t('required_field', ref) : null,
              ),
              const SizedBox(height: 12),
              // Sex selection
              Text(t('sex', ref),
                  style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 4),
              SegmentedButton<String>(
                segments: [
                  ButtonSegment(value: 'M', label: Text(t('male', ref))),
                  ButtonSegment(value: 'F', label: Text(t('female', ref))),
                ],
                selected: {_sex},
                onSelectionChanged: (v) => setState(() => _sex = v.first),
              ),
              const SizedBox(height: 12),

              // Age toggle
              if (_useDob) ...[
                TextFormField(
                  controller: _dobController,
                  decoration: InputDecoration(
                    labelText: t('date_of_birth', ref),
                    border: const OutlineInputBorder(),
                    suffixIcon: IconButton(
                      icon: const Icon(Icons.calendar_today),
                      onPressed: _selectDob,
                    ),
                  ),
                  readOnly: true,
                  onTap: _selectDob,
                  validator: (v) {
                    if (v == null || v.trim().isEmpty) {
                      return t('required_field', ref);
                    }
                    final parsed = DateTime.tryParse(v);
                    if (parsed == null) return t('use_date_format', ref);
                    if (parsed.isAfter(DateTime.now())) {
                      return t('dob_future_error', ref);
                    }
                    return null;
                  },
                ),
                TextButton(
                  onPressed: () => setState(() => _useDob = false),
                  child: Text(t('toggle_age_months', ref)),
                ),
              ] else ...[
                TextFormField(
                  controller: _ageMonthsController,
                  decoration: InputDecoration(
                    labelText: t('age_months', ref),
                    hintText: t('placeholder_age_months', ref),
                    border: const OutlineInputBorder(),
                  ),
                  keyboardType: TextInputType.number,
                  onChanged: _onAgeMonthsChanged,
                  validator: (v) {
                    if (v == null || v.trim().isEmpty) {
                      return t('required_field', ref);
                    }
                    final n = double.tryParse(v);
                    if (n == null || n < 0) {
                      return t('positive_number_error', ref);
                    }
                    return null;
                  },
                ),
                TextButton(
                  onPressed: () => setState(() => _useDob = true),
                  child: Text(t('toggle_dob', ref)),
                ),
              ],

              const Divider(height: 24),

              // === OPTIONAL MEASUREMENTS ===
              _sectionHeader(
                '${t('optional_measurements', ref)} ${t('optional_measurements_note', ref)}',
              ),
              const SizedBox(height: 8),
              TextFormField(
                controller: _weightController,
                decoration: InputDecoration(
                  labelText: t('weight_kg', ref),
                  hintText: t('weight_placeholder', ref),
                  helperText: t('weight_help', ref),
                  border: const OutlineInputBorder(),
                ),
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                validator: (v) => _validatePositive(v),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _muacController,
                decoration: InputDecoration(
                  labelText: t('muac_cm', ref),
                  hintText: t('muac_placeholder', ref),
                  helperText: t('muac_help', ref),
                  border: const OutlineInputBorder(),
                ),
                keyboardType:
                    const TextInputType.numberWithOptions(decimal: true),
                validator: (v) => _validatePositive(v),
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      controller: _heightValueController,
                      decoration: InputDecoration(
                        labelText: t('height', ref),
                        hintText: t('height_placeholder', ref),
                        helperText: t('height_fallback', ref),
                        border: const OutlineInputBorder(),
                      ),
                      keyboardType:
                          const TextInputType.numberWithOptions(decimal: true),
                      validator: (v) => _validatePositive(v),
                    ),
                  ),
                  const SizedBox(width: 8),
                  SegmentedButton<String>(
                    segments: [
                      ButtonSegment(
                          value: 'cm', label: Text(t('unit_cm', ref))),
                      ButtonSegment(
                          value: 'inch', label: Text(t('unit_inch', ref))),
                    ],
                    selected: {_heightUnit},
                    onSelectionChanged: (v) =>
                        setState(() => _heightUnit = v.first),
                  ),
                ],
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _guardianController,
                decoration: InputDecoration(
                  labelText: t('guardian_name', ref),
                  hintText: t('placeholder_optional', ref),
                  border: const OutlineInputBorder(),
                ),
              ),
              const SizedBox(height: 12),
              TextFormField(
                controller: _locationController,
                decoration: InputDecoration(
                  labelText: t('location_clinic', ref),
                  hintText: t('placeholder_optional', ref),
                  border: const OutlineInputBorder(),
                ),
              ),

              const SizedBox(height: 24),

              // === SUBMIT ===
              SizedBox(
                width: double.infinity,
                height: 48,
                child: FilledButton(
                  onPressed: _loading ? null : _submit,
                  child: _loading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : Text(t('run_assessment', ref)),
                ),
              ),

              if (_error != null)
                Padding(
                  padding: const EdgeInsets.only(top: 12),
                  child: Text(_error!, style: const TextStyle(color: Colors.red)),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _sectionHeader(String text, {bool required = false}) {
    return Text(
      text,
      style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
    );
  }

  Widget _photoGuidanceTips() {
    final tips = [
      t('tip_front_1', ref),
      t('tip_front_2', ref),
      t('tip_front_3', ref),
      t('tip_front_4', ref),
    ];
    return Container(
      margin: const EdgeInsets.only(top: 4),
      padding: const EdgeInsets.all(8),
      decoration: BoxDecoration(
        color: Colors.blue.shade50,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: tips
            .map((tip) => Padding(
                  padding: const EdgeInsets.only(bottom: 2),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text('• ', style: TextStyle(fontSize: 12)),
                      Expanded(
                        child: Text(tip, style: const TextStyle(fontSize: 12)),
                      ),
                    ],
                  ),
                ))
            .toList(),
      ),
    );
  }

  Widget _imagePickerRow(String role, XFile? image) {
    return Row(
      children: [
        OutlinedButton.icon(
          onPressed: _loading
              ? null
              : () => _pickImage(ImageSource.camera, role),
          icon: const Icon(Icons.camera_alt, size: 18),
          label: Text(t('capture', ref)),
        ),
        const SizedBox(width: 8),
        OutlinedButton.icon(
          onPressed: _loading
              ? null
              : () => _pickImage(ImageSource.gallery, role),
          icon: const Icon(Icons.photo_library, size: 18),
          label: Text(t('gallery', ref)),
        ),
        const SizedBox(width: 8),
        if (image != null)
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: Image.file(
              File(image.path),
              width: 48,
              height: 48,
              fit: BoxFit.cover,
            ),
          )
        else
          Text(
            t('not_selected', ref),
            style: Theme.of(context).textTheme.bodySmall,
          ),
      ],
    );
  }

  String? _validatePositive(String? value) {
    if (value == null || value.trim().isEmpty) return null;
    final n = double.tryParse(value);
    if (n == null || n <= 0) return t('positive_number_error', ref);
    return null;
  }
}
```

- [ ] **Step 2: Commit**

```bash
mkdir -p flutter_app/lib/screens/assessment
git add flutter_app/lib/screens/assessment/assessment_screen.dart
git commit -m "Add AssessmentScreen with image capture, child info, and measurements form"
```

---

### Task 7: ResultScreen

**Files:**
- Create: `flutter_app/lib/screens/assessment/result_screen.dart`

- [ ] **Step 1: Create result_screen.dart**

Create `flutter_app/lib/screens/assessment/result_screen.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../l10n/l10n_provider.dart';
import '../../models/assessment_result.dart';
import '../../providers/api_provider.dart';
import '../../providers/assessment_provider.dart';
import '../shared/app_scaffold.dart';
import '../shared/status_badge.dart';

class ResultScreen extends ConsumerWidget {
  const ResultScreen({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final result = ref.watch(assessmentResultProvider);

    if (result == null) {
      return AppScaffold(
        currentIndex: 0,
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('No assessment result available.'),
              const SizedBox(height: 16),
              FilledButton(
                onPressed: () => context.go('/'),
                child: Text(t('run_assessment', ref)),
              ),
            ],
          ),
        ),
      );
    }

    return AppScaffold(
      currentIndex: 0,
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _statusBanner(context, ref, result),
            const SizedBox(height: 16),
            _photoSection(context, ref, result),
            const SizedBox(height: 16),
            _metricCards(context, ref, result),
            if (result.mlPrediction != null) ...[
              const SizedBox(height: 16),
              _mlSection(context, ref, result.mlPrediction!),
            ],
            if (result.muac?.muacMethod == 'estimated_from_whz') ...[
              const SizedBox(height: 12),
              _muacNote(context, ref),
            ],
            const SizedBox(height: 24),
            _actionButtons(context, ref),
          ],
        ),
      ),
    );
  }

  Widget _statusBanner(
      BuildContext context, WidgetRef ref, AssessmentResult result) {
    final whz = result.nutrition.whzStatus;
    final haz = result.nutrition.hazStatus;

    String title;
    String message;
    Color color;

    if (whz != null && whz.toUpperCase().contains('SAM')) {
      title = t('banner_sam_title', ref);
      message = t('banner_sam_msg', ref);
      color = Colors.red;
    } else if (whz != null && whz.toUpperCase().contains('MAM')) {
      title = t('banner_mam_title', ref);
      message = t('banner_mam_msg', ref);
      color = Colors.orange;
    } else if (haz != null && haz.toLowerCase().contains('stunted')) {
      title = haz;
      message = t('banner_stunted_msg', ref);
      color = Colors.amber.shade700;
    } else if (whz != null && whz.toLowerCase() == 'normal') {
      title = t('banner_normal_title', ref);
      message = t('banner_normal_msg', ref);
      color = Colors.green;
    } else {
      title = t('banner_unknown_title', ref);
      message = t('banner_unknown_msg', ref);
      color = Colors.grey;
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        border: Border(left: BorderSide(color: color, width: 4)),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            '${result.childName} — ${result.ageMonths.toStringAsFixed(1)} ${t('months_unit', ref)}',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 4),
          Text(
            title,
            style: Theme.of(context)
                .textTheme
                .titleMedium
                ?.copyWith(color: color, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          Text(message),
        ],
      ),
    );
  }

  Widget _photoSection(
      BuildContext context, WidgetRef ref, AssessmentResult result) {
    final annotatedImage = result.measurement.estimationMethod;
    final confidence = result.measurement.confidenceScore;
    final baseUrl = ref.read(baseUrlProvider);

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (confidence != null) ...[
              Row(
                children: [
                  Text('${t('pose_confidence', ref)}: '),
                  Expanded(
                    child: LinearProgressIndicator(
                      value: confidence,
                      backgroundColor: Colors.grey.shade200,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Text('${(confidence * 100).toStringAsFixed(0)}%'),
                ],
              ),
              const SizedBox(height: 4),
            ],
            if (annotatedImage != null)
              Text(
                'Method: $annotatedImage',
                style: Theme.of(context).textTheme.bodySmall,
              ),
          ],
        ),
      ),
    );
  }

  Widget _metricCards(
      BuildContext context, WidgetRef ref, AssessmentResult result) {
    return Column(
      children: [
        _metricCard(
          context,
          ref,
          title: t('metric_height', ref),
          value: result.measurement.predictedHeightCm ??
              result.measurement.manualHeightCm,
          unit: 'cm',
          source: result.measurement.manualHeightCm != null
              ? t('badge_manual', ref)
              : result.measurement.predictedHeightCm != null
                  ? t('badge_image', ref)
                  : t('badge_undetected', ref),
          zscore: result.nutrition.hazZscore,
          status: result.nutrition.hazStatus,
        ),
        const SizedBox(height: 8),
        _metricCard(
          context,
          ref,
          title: t('metric_weight', ref),
          value: result.measurement.predictedWeightKg ??
              result.measurement.manualWeightKg,
          unit: 'kg',
          source: result.measurement.manualWeightKg != null
              ? t('badge_manual', ref)
              : result.measurement.predictedWeightKg != null
                  ? t('badge_image', ref)
                  : t('badge_undetected', ref),
          zscore: result.nutrition.whzZscore,
          status: result.nutrition.whzStatus,
          extras: _weightExtras(context, ref, result.measurement),
        ),
        const SizedBox(height: 8),
        _muacCard(context, ref, result.muac),
      ],
    );
  }

  Widget? _weightExtras(
      BuildContext context, WidgetRef ref, Measurement m) {
    if (!m.sideViewUsed) return null;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(t('badge_side_view_ok', ref),
            style: const TextStyle(fontSize: 12, color: Colors.teal)),
        if (m.chestDepthCm != null)
          Text(
            '${t('chest_depth', ref)} ${m.chestDepthCm!.toStringAsFixed(1)} cm',
            style: const TextStyle(fontSize: 11),
          ),
        if (m.abdDepthCm != null)
          Text(
            '${t('abd_depth', ref)} ${m.abdDepthCm!.toStringAsFixed(1)} cm',
            style: const TextStyle(fontSize: 11),
          ),
      ],
    );
  }

  Widget _metricCard(
    BuildContext context,
    WidgetRef ref, {
    required String title,
    required double? value,
    required String unit,
    required String source,
    required double? zscore,
    required String? status,
    Widget? extras,
  }) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Text(title,
                          style: Theme.of(context).textTheme.titleSmall),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 6, vertical: 1),
                        decoration: BoxDecoration(
                          color: Colors.grey.shade200,
                          borderRadius: BorderRadius.circular(4),
                        ),
                        child: Text(source,
                            style: const TextStyle(fontSize: 11)),
                      ),
                    ],
                  ),
                  const SizedBox(height: 4),
                  Text(
                    value != null
                        ? '${value.toStringAsFixed(1)} $unit'
                        : '—',
                    style: Theme.of(context).textTheme.headlineSmall,
                  ),
                  if (zscore != null)
                    Text(
                      'Z-score: ${zscore.toStringAsFixed(2)}',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  if (extras != null) ...[
                    const SizedBox(height: 4),
                    extras,
                  ],
                ],
              ),
            ),
            if (status != null) StatusBadge(status: status),
          ],
        ),
      ),
    );
  }

  Widget _muacCard(BuildContext context, WidgetRef ref, MuacDetail? muac) {
    if (muac == null) {
      return _metricCard(
        context,
        ref,
        title: t('metric_muac', ref),
        value: null,
        unit: 'cm',
        source: t('badge_na', ref),
        zscore: null,
        status: null,
      );
    }
    final source = muac.muacMethod == 'manual'
        ? t('badge_tape', ref)
        : t('badge_est', ref);
    return _metricCard(
      context,
      ref,
      title: t('metric_muac', ref),
      value: muac.muacCm,
      unit: 'cm',
      source: source,
      zscore: null,
      status: muac.muacStatus,
    );
  }

  Widget _mlSection(
      BuildContext context, WidgetRef ref, MlPrediction ml) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('ml_wasting_title', ref),
              style: Theme.of(context).textTheme.titleSmall,
            ),
            Text(
              t('ml_wasting_sub', ref),
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            _probabilityBar(
                t('sam_probability', ref), ml.samProbability, Colors.red),
            const SizedBox(height: 4),
            _probabilityBar(
                t('mam_probability', ref), ml.mamProbability, Colors.orange),
            const SizedBox(height: 4),
            _probabilityBar(t('normal_probability', ref),
                ml.normalProbability, Colors.green),
            if (ml.estimatedWeightKg != null) ...[
              const SizedBox(height: 8),
              Text(
                '${t('ml_estimated_weight', ref)} ${ml.estimatedWeightKg!.toStringAsFixed(2)} kg',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _probabilityBar(String label, double? value, Color color) {
    final pct = value ?? 0;
    return Row(
      children: [
        SizedBox(width: 120, child: Text(label, style: const TextStyle(fontSize: 12))),
        Expanded(
          child: LinearProgressIndicator(
            value: pct,
            backgroundColor: Colors.grey.shade200,
            valueColor: AlwaysStoppedAnimation(color),
          ),
        ),
        const SizedBox(width: 8),
        Text('${(pct * 100).toStringAsFixed(0)}%',
            style: const TextStyle(fontSize: 12)),
      ],
    );
  }

  Widget _muacNote(BuildContext context, WidgetRef ref) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.amber.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: Colors.amber.shade200),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline, color: Colors.amber.shade800, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: RichText(
              text: TextSpan(
                style: DefaultTextStyle.of(context).style.copyWith(fontSize: 13),
                children: [
                  TextSpan(
                    text: '${t('muac_note_strong', ref)} ',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  TextSpan(text: t('muac_note_text', ref)),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _actionButtons(BuildContext context, WidgetRef ref) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        FilledButton.icon(
          onPressed: () {
            ref.read(assessmentResultProvider.notifier).state = null;
            context.go('/');
          },
          icon: const Icon(Icons.refresh),
          label: Text(t('assess_another', ref)),
        ),
        OutlinedButton.icon(
          onPressed: () => context.go('/children'),
          icon: const Icon(Icons.people),
          label: Text(t('view_all_children', ref)),
        ),
      ],
    );
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add flutter_app/lib/screens/assessment/result_screen.dart
git commit -m "Add ResultScreen with status banner, metric cards, ML section, and MUAC note"
```

---

### Task 8: ChildrenListScreen

**Files:**
- Create: `flutter_app/lib/screens/children/children_list_screen.dart`

- [ ] **Step 1: Create children_list_screen.dart**

Create `flutter_app/lib/screens/children/children_list_screen.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../l10n/l10n_provider.dart';
import '../../providers/children_provider.dart';
import '../shared/app_scaffold.dart';

class ChildrenListScreen extends ConsumerStatefulWidget {
  const ChildrenListScreen({super.key});

  @override
  ConsumerState<ChildrenListScreen> createState() => _ChildrenListScreenState();
}

class _ChildrenListScreenState extends ConsumerState<ChildrenListScreen> {
  String _searchQuery = '';

  @override
  Widget build(BuildContext context) {
    final childrenAsync = ref.watch(childrenProvider);

    return AppScaffold(
      currentIndex: 1,
      child: Column(
        children: [
          // Header + search
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  t('registered_children', ref),
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
                const SizedBox(height: 12),
                TextField(
                  decoration: InputDecoration(
                    hintText: t('search_children_placeholder', ref),
                    prefixIcon: const Icon(Icons.search),
                    border: const OutlineInputBorder(),
                  ),
                  onChanged: (v) => setState(() => _searchQuery = v),
                ),
              ],
            ),
          ),

          // Children list
          Expanded(
            child: childrenAsync.when(
              loading: () =>
                  const Center(child: CircularProgressIndicator()),
              error: (error, _) => Center(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(error.toString(),
                          style: const TextStyle(color: Colors.red)),
                      const SizedBox(height: 8),
                      OutlinedButton(
                        onPressed: () => ref.invalidate(childrenProvider),
                        child: const Text('Retry'),
                      ),
                    ],
                  ),
                ),
              ),
              data: (children) {
                final filtered = _searchQuery.isEmpty
                    ? children
                    : children
                        .where((c) => c.name
                            .toLowerCase()
                            .contains(_searchQuery.toLowerCase()))
                        .toList();

                if (children.isEmpty) {
                  return Center(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(t('empty_children', ref)),
                        const SizedBox(height: 8),
                        TextButton(
                          onPressed: () => context.go('/'),
                          child: Text(
                            '${t('empty_children_link', ref)} ${t('empty_children_suffix', ref)}',
                          ),
                        ),
                      ],
                    ),
                  );
                }

                if (filtered.isEmpty) {
                  return Center(
                    child: Text(t('search_no_results', ref)),
                  );
                }

                return RefreshIndicator(
                  onRefresh: () async => ref.invalidate(childrenProvider),
                  child: ListView.separated(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    itemCount: filtered.length,
                    separatorBuilder: (_, __) => const Divider(height: 1),
                    itemBuilder: (context, index) {
                      final child = filtered[index];
                      return ListTile(
                        title: Text(child.name),
                        subtitle: Text(
                          '${t('th_dob', ref)}: ${child.dateOfBirth}  •  '
                          '${t('th_sex', ref)}: ${child.sex}  •  '
                          '${t('th_visits', ref)}: ${child.visitCount}',
                        ),
                        trailing: const Icon(Icons.chevron_right),
                        onTap: () => context.go('/children/${child.id}'),
                      );
                    },
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 2: Commit**

```bash
mkdir -p flutter_app/lib/screens/children
git add flutter_app/lib/screens/children/children_list_screen.dart
git commit -m "Add ChildrenListScreen with search and pull-to-refresh"
```

---

### Task 9: ChildDetailScreen with Growth Chart

**Files:**
- Create: `flutter_app/lib/screens/children/child_detail_screen.dart`

- [ ] **Step 1: Create child_detail_screen.dart**

Create `flutter_app/lib/screens/children/child_detail_screen.dart`:

```dart
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../l10n/l10n_provider.dart';
import '../../models/child_detail.dart';
import '../../providers/children_provider.dart';
import '../shared/app_scaffold.dart';
import '../shared/status_badge.dart';

class ChildDetailScreen extends ConsumerWidget {
  const ChildDetailScreen({super.key, required this.childId});

  final int childId;

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final detailAsync = ref.watch(childDetailProvider(childId));

    return AppScaffold(
      currentIndex: 1,
      child: detailAsync.when(
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => Center(
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(error.toString(),
                    style: const TextStyle(color: Colors.red)),
                const SizedBox(height: 8),
                OutlinedButton(
                  onPressed: () =>
                      ref.invalidate(childDetailProvider(childId)),
                  child: const Text('Retry'),
                ),
              ],
            ),
          ),
        ),
        data: (child) => SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _profileCard(context, ref, child),
              const SizedBox(height: 16),
              if (_hasChartData(child)) ...[
                _growthChart(context, ref, child),
                const SizedBox(height: 16),
              ],
              _visitHistory(context, ref, child),
              const SizedBox(height: 16),
              OutlinedButton.icon(
                onPressed: () => context.go('/children'),
                icon: const Icon(Icons.arrow_back),
                label: Text(t('back_to_children', ref)),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _profileCard(
      BuildContext context, WidgetRef ref, ChildDetail child) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              child.name,
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const SizedBox(height: 8),
            _profileRow(t('label_dob', ref), child.dateOfBirth),
            _profileRow(t('label_sex', ref), child.sex == 'M' ? 'Male' : 'Female'),
            _profileRow(t('label_guardian', ref), child.guardianName ?? '—'),
            _profileRow(t('label_location', ref), child.location ?? '—'),
            _profileRow(
                t('total_visits', ref), child.visits.length.toString()),
          ],
        ),
      ),
    );
  }

  Widget _profileRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4),
      child: Row(
        children: [
          SizedBox(
            width: 100,
            child: Text(label,
                style: const TextStyle(fontWeight: FontWeight.w500)),
          ),
          Expanded(child: Text(value)),
        ],
      ),
    );
  }

  bool _hasChartData(ChildDetail child) {
    int withData = 0;
    for (final v in child.visits) {
      if (v.measurement?.predictedHeightCm != null ||
          v.measurement?.predictedWeightKg != null) {
        withData++;
      }
    }
    return withData >= 2;
  }

  Widget _growthChart(
      BuildContext context, WidgetRef ref, ChildDetail child) {
    final visitsWithData = child.visits
        .where((v) =>
            v.measurement?.predictedHeightCm != null ||
            v.measurement?.predictedWeightKg != null)
        .toList()
      ..sort((a, b) => (a.ageMonths ?? 0).compareTo(b.ageMonths ?? 0));

    final heightSpots = <FlSpot>[];
    final weightSpots = <FlSpot>[];

    for (final v in visitsWithData) {
      final x = v.ageMonths ?? 0;
      final h = v.measurement?.predictedHeightCm;
      final w = v.measurement?.predictedWeightKg;
      if (h != null) heightSpots.add(FlSpot(x, h));
      if (w != null) weightSpots.add(FlSpot(x, w));
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('growth_chart_title', ref),
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 250,
              child: LineChart(
                LineChartData(
                  lineBarsData: [
                    if (heightSpots.isNotEmpty)
                      LineChartBarData(
                        spots: heightSpots,
                        isCurved: true,
                        color: Colors.blue,
                        barWidth: 2,
                        dotData: const FlDotData(show: true),
                        belowBarData: BarAreaData(show: false),
                      ),
                    if (weightSpots.isNotEmpty)
                      LineChartBarData(
                        spots: weightSpots,
                        isCurved: true,
                        color: Colors.orange,
                        barWidth: 2,
                        dotData: const FlDotData(show: true),
                        belowBarData: BarAreaData(show: false),
                      ),
                  ],
                  titlesData: FlTitlesData(
                    bottomTitles: AxisTitles(
                      axisNameWidget: Text(t('age_months', ref),
                          style: const TextStyle(fontSize: 12)),
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 30,
                        getTitlesWidget: (value, meta) => Text(
                          value.toStringAsFixed(0),
                          style: const TextStyle(fontSize: 10),
                        ),
                      ),
                    ),
                    leftTitles: AxisTitles(
                      axisNameWidget: Text(
                        t('chart_height_cm', ref),
                        style: const TextStyle(
                            fontSize: 12, color: Colors.blue),
                      ),
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        getTitlesWidget: (value, meta) => Text(
                          value.toStringAsFixed(0),
                          style: const TextStyle(
                              fontSize: 10, color: Colors.blue),
                        ),
                      ),
                    ),
                    rightTitles: AxisTitles(
                      axisNameWidget: Text(
                        t('chart_weight_kg', ref),
                        style: const TextStyle(
                            fontSize: 12, color: Colors.orange),
                      ),
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                        getTitlesWidget: (value, meta) => Text(
                          value.toStringAsFixed(0),
                          style: const TextStyle(
                              fontSize: 10, color: Colors.orange),
                        ),
                      ),
                    ),
                    topTitles: const AxisTitles(
                        sideTitles: SideTitles(showTitles: false)),
                  ),
                  gridData: FlGridData(
                    show: true,
                    drawHorizontalLine: true,
                    drawVerticalLine: false,
                  ),
                  borderData: FlBorderData(show: true),
                  lineTouchData: LineTouchData(
                    touchTooltipData: LineTouchTooltipData(
                      getTooltipItems: (touchedSpots) {
                        return touchedSpots.map((spot) {
                          final isHeight = spot.barIndex == 0 &&
                              heightSpots.isNotEmpty;
                          return LineTooltipItem(
                            '${spot.y.toStringAsFixed(1)} ${isHeight ? 'cm' : 'kg'}',
                            TextStyle(
                              color: isHeight ? Colors.blue : Colors.orange,
                              fontSize: 12,
                            ),
                          );
                        }).toList();
                      },
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _legendDot(Colors.blue, t('chart_height_cm', ref)),
                const SizedBox(width: 16),
                _legendDot(Colors.orange, t('chart_weight_kg', ref)),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _legendDot(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 10,
          height: 10,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        const SizedBox(width: 4),
        Text(label, style: const TextStyle(fontSize: 12)),
      ],
    );
  }

  Widget _visitHistory(
      BuildContext context, WidgetRef ref, ChildDetail child) {
    final visits = child.visits.reversed.toList();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              t('visit_history', ref),
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 8),
            if (visits.isEmpty)
              Text(t('no_visits_yet', ref))
            else
              ...visits.map((v) => _visitRow(context, ref, v)),
          ],
        ),
      ),
    );
  }

  Widget _visitRow(BuildContext context, WidgetRef ref, ChildVisit visit) {
    final m = visit.measurement;
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                visit.visitDate ?? '—',
                style: const TextStyle(fontWeight: FontWeight.w500),
              ),
              const Spacer(),
              Text(
                '${visit.ageMonths?.toStringAsFixed(1) ?? '—'} ${t('months_unit', ref)}',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ),
          const SizedBox(height: 2),
          if (m != null)
            Row(
              children: [
                Text(
                  '${t('th_height_cm', ref)}: ${m.predictedHeightCm?.toStringAsFixed(1) ?? '—'}',
                  style: const TextStyle(fontSize: 13),
                ),
                const SizedBox(width: 12),
                Text(
                  '${t('th_weight_kg', ref)}: ${m.predictedWeightKg?.toStringAsFixed(1) ?? '—'}',
                  style: const TextStyle(fontSize: 13),
                ),
                const Spacer(),
                if (m.hazStatus != null) ...[
                  StatusBadge(status: m.hazStatus),
                  const SizedBox(width: 4),
                ],
                if (m.whzStatus != null) StatusBadge(status: m.whzStatus),
              ],
            )
          else
            Text(
              t('no_measurement_data', ref),
              style: Theme.of(context).textTheme.bodySmall,
            ),
          const Divider(height: 12),
        ],
      ),
    );
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add flutter_app/lib/screens/children/child_detail_screen.dart
git commit -m "Add ChildDetailScreen with profile card, growth chart, and visit history"
```

---

### Task 10: Rewrite main.dart and Delete Old Screen

**Files:**
- Rewrite: `flutter_app/lib/main.dart`
- Delete: `flutter_app/lib/screens/assessment_screen.dart`

- [ ] **Step 1: Rewrite main.dart**

Replace the entire contents of `flutter_app/lib/main.dart` with:

```dart
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'router.dart';

void main() {
  runApp(const ProviderScope(child: ChildGrowthApp()));
}

class ChildGrowthApp extends StatelessWidget {
  const ChildGrowthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp.router(
      title: 'SNEH Growth Monitor',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal),
        useMaterial3: true,
      ),
      routerConfig: appRouter,
    );
  }
}
```

- [ ] **Step 2: Delete old monolith screen**

```bash
rm flutter_app/lib/screens/assessment_screen.dart
```

- [ ] **Step 3: Run flutter analyze to verify everything compiles**

```bash
cd flutter_app && flutter analyze
```

Expected: `No issues found!` (or warnings only, no errors).

- [ ] **Step 4: Commit**

```bash
git add flutter_app/lib/main.dart
git add -u flutter_app/lib/screens/assessment_screen.dart
git commit -m "Wire up ProviderScope + GoRouter in main.dart, delete old monolith screen"
```

---

### Task 11: Verify and Fix

This is a cleanup task for any compilation errors from flutter analyze.

- [ ] **Step 1: Run full flutter analyze**

```bash
cd flutter_app && flutter analyze
```

- [ ] **Step 2: Fix any errors found**

Address any issues reported by the analyzer — missing imports, type mismatches, deprecated APIs, etc.

- [ ] **Step 3: Run flutter build to verify**

```bash
cd flutter_app && flutter build apk --debug 2>&1 | tail -5
```

Expected: `✓ Built build/app/outputs/flutter-apk/app-debug.apk`

- [ ] **Step 4: Commit fixes if any**

```bash
git add -A flutter_app/lib/
git commit -m "Fix analyzer issues from flutter web parity migration"
```
