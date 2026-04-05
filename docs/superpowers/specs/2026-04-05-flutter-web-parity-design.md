# Flutter App — Full Parity with Web Version

**Date:** 2026-04-05
**Goal:** Rebuild the Flutter app's UI layer to match all features of the working Jinja2 web version, while keeping the existing well-structured models and API service.

## Decisions

- **Architecture:** Hybrid — keep existing models + ApiService, rewrite UI layer
- **State management:** Riverpod (per CLAUDE.md)
- **Navigation:** GoRouter with 4 routes matching the web
- **Charts:** fl_chart for dual-axis growth charts
- **i18n:** Custom translations map ported from Python (no codegen)

## Folder Structure

```
flutter_app/lib/
├── main.dart                          # App entry, ProviderScope, GoRouter setup
├── models/                            # KEEP AS-IS
│   ├── assessment_result.dart
│   ├── child.dart
│   └── child_detail.dart
├── services/
│   └── api_service.dart               # KEEP AS-IS
├── l10n/
│   ├── translations.dart              # en/mr translation maps (ported from translations.py)
│   └── l10n_provider.dart             # Riverpod provider for current locale
├── providers/
│   ├── api_provider.dart              # ApiService singleton provider
│   ├── children_provider.dart         # Children list + detail state
│   └── assessment_provider.dart       # Assessment form + result state
├── screens/
│   ├── assessment/
│   │   ├── assessment_screen.dart     # Form: image capture, child info, measurements
│   │   └── result_screen.dart         # Rich result display
│   ├── children/
│   │   ├── children_list_screen.dart  # Searchable table
│   │   └── child_detail_screen.dart   # Profile + chart + history
│   └── shared/
│       ├── app_scaffold.dart          # Shared navbar/scaffold
│       └── status_badge.dart          # Color-coded status badges
└── router.dart                        # GoRouter config
```

## Routes

| Route | Screen | Web Equivalent |
|-------|--------|----------------|
| `/` | AssessmentScreen | `GET /` (index.html) |
| `/result` | ResultScreen | `POST /assess` (result.html) |
| `/children` | ChildrenListScreen | `GET /children` (children.html) |
| `/children/:id` | ChildDetailScreen | `GET /children/{id}` (child_detail.html) |

## Screen Specifications

### 1. AssessmentScreen (index.html parity)

**Image Capture Section:**
- Front photo: REQUIRED — camera or gallery picker
- Side photo: OPTIONAL — labeled "+30-40% weight accuracy"
- Back photo: OPTIONAL
- Each shows thumbnail preview after selection
- Photo guidance tips displayed

**Child Information Section:**
- Child name: required text field, max 100 chars
- Sex: M/F radio buttons (or SegmentedButton)
- Age input: toggle between age_months number field OR date_of_birth picker
  - Bidirectional sync: changing months updates DOB and vice versa
  - Conversion: months = days / 30.4375
  - Default DOB: 3 years before today

**Optional Measurements Section:**
- Weight (kg): 0.5-50 range
- MUAC (cm): 5-30 range, labeled "tape measurement"
- Height: number field + unit toggle (cm/inch) with live conversion
  - Cannot provide both height_cm and height_value
- Guardian name: text field
- Location/clinic name: text field

**Submit:**
- Validation: name required, DOB valid + not future, front image required
- Loading spinner during submission
- Error display on failure
- On success: navigate to `/result`

**Backend Health:**
- Health check indicator (green/red dot or similar)
- API base URL config in debug mode only

### 2. ResultScreen (result.html parity)

**Status Banner:**
- Full-width color-coded banner at top
- Colors: SAM=red, MAM=orange, Stunted=yellow, Normal=green, Unknown=gray
- Shows overall status text

**Photo Section:**
- Annotated pose image from API response (`annotated_image` field)
- Loaded from backend URL: `{baseUrl}/uploads/{annotated_image}`
- Pose confidence score with progress bar
- Estimation method label

**Three Metric Cards:**

Each card displays:
- Metric name (Height / Weight / MUAC)
- Value in appropriate unit (cm / kg / cm)
- Source badge: "Image", "Manual", "Estimated", "Undetected", "N/A"
- Z-score value (HAZ for height, WHZ for weight)
- Status classification with colored badge

Height card:
- `predicted_height_cm` or `manual_height_cm`
- HAZ z-score and status

Weight card:
- `predicted_weight_kg` or `manual_weight_kg`
- WHZ z-score and status
- Side view flag, chest/abd depth if available

MUAC card:
- `muac_cm` value
- MUAC status (SAM / At Risk / Normal)
- Method badge (manual / estimated)

**ML Wasting Detection Section** (shown only when `ml_prediction` is present):
- Horizontal probability bars for SAM%, MAM%, Normal%
- ML-estimated weight in kg
- Estimation method note

**MUAC Note** (shown when method is "estimated_from_whz"):
- Clarification text: estimated, not tape-measured
- Recommendation to confirm with physical tape

**Action Buttons:**
- Share/Print report (uses Flutter share or print)
- Assess Another Child → navigate to `/`
- View All Children → navigate to `/children`

### 3. ChildrenListScreen (children.html parity)

**Search:**
- Text field at top for real-time filtering by name (case-insensitive substring)

**Children Table/List:**
- Columns/fields: Name, DOB, Sex, Guardian, Location, Visit Count
- Tap row → navigate to `/children/{id}`
- Empty state message when no children registered
- Pull-to-refresh to reload from API

### 4. ChildDetailScreen (child_detail.html parity)

**Profile Card:**
- Child name, DOB, sex, guardian, location, total visits

**Growth Chart** (shown only when 2+ visits with measurements):
- fl_chart LineChart with dual Y-axes
- Left Y-axis: Height (cm) — line color: blue
- Right Y-axis: Weight (kg) — line color: orange
- X-axis: visit dates (timeline)
- Data points from visit history
- Touch tooltips showing values

**Visit History:**
- Reverse chronological list (newest first)
- Each visit shows: date, age (months), height (cm), weight (kg), HAZ status badge, WHZ status badge
- Missing data shown as "-"
- Status badges use StatusBadge widget (color-coded)

## Shared Components

### AppScaffold
- Wraps all screens with consistent AppBar
- AppBar contains:
  - App title: "Child Growth Monitor"
  - Navigation items: Assess, Children (as tabs or bottom nav on mobile)
  - Language toggle button (EN/MR) in actions
- Handles responsive layout

### StatusBadge
- Reusable widget for nutritional status labels
- Input: status string (e.g., "Normal", "Stunted", "SAM", "MAM", "Obese", etc.)
- Output: colored Container with text
- Color mapping:
  - Normal → green
  - Stunted / Severely Stunted → yellow / orange
  - MAM / At Risk → orange
  - SAM / Severe → red
  - Overweight / Obese → purple
  - Unknown / null → gray

## State Management (Riverpod)

### Providers

**`apiProvider`** — Provider<ApiService>
- Singleton ApiService instance
- Base URL from SharedPreferences (debug) or production default

**`localeProvider`** — StateNotifierProvider<LocaleNotifier, String>
- Current language code: "en" or "mr"
- Persisted to SharedPreferences
- Notifies all consumers on change

**`childrenProvider`** — FutureProvider<List<ChildSummary>>
- Calls `apiService.getChildren()`
- Supports refresh via `ref.refresh()`

**`childDetailProvider(int id)`** — FutureProvider.family<ChildDetail, int>
- Calls `apiService.getChildDetail(id)`

**`assessmentResultProvider`** — StateProvider<AssessmentResult?>
- Holds the latest assessment result
- Set after successful submission, read by ResultScreen
- Cleared when starting a new assessment

### No provider needed for:
- Form state (local to AssessmentScreen's StatefulWidget)
- Search filter text (local to ChildrenListScreen)

## i18n

### translations.dart
Port `app/web/translations.py` to Dart:
```dart
const Map<String, Map<String, String>> translations = {
  'app_title': {'en': 'Child Growth Monitor', 'mr': '...'},
  'nav_assess': {'en': 'Assess', 'mr': '...'},
  // ... all keys from translations.py
};
```

### l10n_provider.dart
- `localeProvider` StateNotifier with "en" default
- `t(String key, WidgetRef ref)` helper function that reads current locale and returns the translated string
- Fallback to English if key missing for current locale

## Dependencies to Add

```yaml
# pubspec.yaml additions
dependencies:
  flutter_riverpod: ^2.6.1
  riverpod_annotation: ^2.6.1
  go_router: ^14.8.1
  fl_chart: ^0.70.2

dev_dependencies:
  riverpod_generator: ^2.6.3
  build_runner: ^2.4.14
```

Note: `image_picker`, `shared_preferences`, `http`, `intl` are already present.

## What We're NOT Changing

- `models/assessment_result.dart` — already maps to API response correctly
- `models/child.dart` — already maps to children list response
- `models/child_detail.dart` — already maps to child detail response
- `services/api_service.dart` — already handles all 4 API endpoints
- Backend API — no changes needed, Flutter consumes the same endpoints

## What Gets Replaced

- `screens/assessment_screen.dart` (630-line monolith) → split into:
  - `screens/assessment/assessment_screen.dart` (form only)
  - `screens/assessment/result_screen.dart` (result display)
  - `screens/children/children_list_screen.dart` (children table)
  - `screens/children/child_detail_screen.dart` (profile + chart)
  - `screens/shared/app_scaffold.dart` (navbar)
  - `screens/shared/status_badge.dart` (reusable badge)

The old `assessment_screen.dart` will be deleted once all its functionality has been migrated to the new screens.
