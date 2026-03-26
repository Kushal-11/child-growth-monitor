#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$APP_DIR"

if ! command -v flutter >/dev/null 2>&1; then
  echo "Flutter SDK is not installed or not on PATH." >&2
  echo "Install Flutter: https://docs.flutter.dev/get-started/install" >&2
  exit 1
fi

if [ ! -d android ]; then
  echo "Android project missing at flutter_app/android; generating platform files..." >&2
  flutter create --platforms=android --project-name child_growth_monitor_app .
fi

flutter pub get
flutter analyze
flutter test

API_BASE_URL="${API_BASE_URL:-http://10.0.2.2:8000}"
flutter build apk --release --dart-define=API_BASE_URL="$API_BASE_URL"

echo "APK built at: $APP_DIR/build/app/outputs/flutter-apk/app-release.apk"
