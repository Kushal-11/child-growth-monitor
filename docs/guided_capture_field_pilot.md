# Guided live capture field-pilot runbook

Status: gated research and screening workflow. Camera results are estimates and
must not be used as measured clinical diagnoses.

## 1. Entry criteria

Do not start a field pilot until all of the following are recorded:

- backend and Flutter automated suites pass at the same Git SHA;
- the gated debug APK builds with `LIVE_CAPTURE=true`;
- the normal release gate remains disabled without that define;
- the backend URL, TLS/network path, test operator, and test child profiles are
  prepared;
- caregiver consent wording and the approved consent version are confirmed;
- the device clock, free storage, and battery level are suitable;
- no real child data will be copied to an unapproved location.

Build and record the exact artifact:

```bash
git rev-parse HEAD
cd flutter_app
flutter build apk --debug --dart-define=LIVE_CAPTURE=true
sha256sum build/app/outputs/flutter-apk/app-debug.apk
adb install -r build/app/outputs/flutter-apk/app-debug.apk
```

Record:

| Field | Value |
|---|---|
| Date/time and site | |
| Tester/operator code | |
| Git SHA | |
| APK SHA-256 | |
| Device manufacturer/model | |
| Android version/API level | |
| RAM/storage before test | |
| App version | |
| Backend version/base URL | |
| Network mode | online / offline / constrained |
| Consent version | |

Never record the child or caregiver name in the pilot evidence sheet. Use the
visit UUID or an approved pilot code.

## 2. Baseline device checks

```bash
adb shell getprop ro.product.manufacturer
adb shell getprop ro.product.model
adb shell getprop ro.build.version.release
adb shell getprop ro.build.version.sdk
adb shell dumpsys meminfo com.example.child_growth_monitor_app
adb logcat -c
```

Open the app, authenticate, select a test child, and confirm **New photo
assessment** is visible only in the gated build. Verify the consent screen
appears before the camera opens and that it requests no height, weight, or MUAC.

## 3. Capture matrix

For each case, retain the visit UUID, expected outcome, actual guidance,
accepted/rejected role, retry count, elapsed time, crash/ANR status, and a
redacted screenshot or screen recording where permitted.

| Case | Procedure | Expected outcome |
|---|---|---|
| Front orientation | Hold device portrait, child front-facing and full body | Front accepted only when orientation, pose, coverage, light, and sharpness pass |
| Side orientation | Child at true side view | Side accepted; front-like orientation rejected with a role-specific instruction |
| Camera rotation | Rotate before and during preview | Preview and saved frame remain correctly oriented |
| Lifecycle interruption | Background app, lock/unlock, answer permission prompt, then resume | Camera restores safely; accepted photos and draft remain |
| Background person | Add a second person behind/beside subject | Multiple/background pose is rejected or the selected pose is unambiguous; never silently use the wrong child |
| Low light | Dim the scene | Lighting failure and actionable retake guidance |
| Overexposure | Strong backlight/direct bright light | Exposure/lighting failure and retake guidance |
| Blur | Move phone during burst | Sharpness failure; blurred frame is not retained as accepted |
| Cropped head | Move framing upward/downward to remove head | Coverage failure |
| Cropped feet | Remove feet from frame | Coverage failure |
| Excessive tilt | Tilt/roll the phone beyond guidance | Orientation/tilt failure |
| Optional views | Skip back and arm views after front and side | Review remains available; optional roles are not required |
| Repeated required failure | Fail front or side three times | Visit is saved as **Incomplete capture** and can be resumed |
| Inference failure | Use the controlled failure build/fake | **Estimate failed — retry** appears; accepted photos remain |

Capture timing with a monotonic stopwatch or screen recording:

- consent confirmation to draft creation;
- camera open to first guidance;
- start burst to selected-photo review;
- required views complete to estimated report;
- retry to recovered report.

The pilot has no automatic clinical-performance pass based only on speed.
Escalate any visible freeze, ANR, crash, repeated frame backlog, or capture that
cannot be completed on the lower-end target device. Attach:

```bash
adb logcat -d > guided-capture-logcat.txt
adb shell dumpsys meminfo com.example.child_growth_monitor_app \
  > guided-capture-meminfo.txt
```

## 4. Offline, restart, and synchronization

1. Enable airplane mode before consent.
2. Create the draft and capture accepted front and side photos.
3. Confirm the estimated report is available and marked as estimated.
4. Force-stop the app:

   ```bash
   adb shell am force-stop com.example.child_growth_monitor_app
   ```

5. Reopen the app while still offline. Confirm the draft/report and retained
   photos survive.
6. Add a partial same-visit measured value, such as height only. Confirm the
   measurement-based report is primary and unavailable components say **Not
   measured**.
7. Confirm **Compare with estimate** still shows the original immutable result
   UUID/version/model and a signed/absolute difference.
8. Restore connectivity and trigger **Sync now**.
9. Verify each visit, front asset, side asset, camera result, and measured
   revision receives its own exact UUID acknowledgement.
10. Interrupt the app during sync, reopen, and retry. Confirm no duplicate
    server visit/assets/results/revisions are created.
11. Before acknowledgement, confirm local cleanup cannot remove the pending
    photo. After acknowledgement, confirm only acknowledged media is eligible
    for cleanup.
12. Request deletion of one selected asset. Confirm the byte is retained until
    the deletion acknowledgement, while child, visit, camera metadata,
    measured report, and revision history remain.

Record the server IDs/object IDs, acknowledgement timestamps, retry count, and
local-file presence before and after each acknowledgement. Do not copy raw
source paths into shared evidence.

## 5. Wording comprehension

Ask the worker to explain, in their own words:

- whether the photo result is measured or estimated;
- whether it replaces a scale, board, or MUAC tape;
- what changes after measured details are added;
- why **Compare with estimate** remains visible;
- what **Not measured** means.

Fail the wording check if the worker interprets a camera estimate as a clinical
measurement or believes missing measured components were inferred.

## 6. Low-memory and lower-end device pass

Repeat the required-view workflow on the lowest supported device while other
typical apps are present. Run at least five consecutive visits and record:

- peak and post-visit memory from `dumpsys meminfo`;
- camera reopen behavior after backgrounding;
- time to guidance, burst selection, and report;
- device temperature/throttling observations;
- any dropped preview, stale orientation, native interpreter error, crash, or
  ANR;
- whether retained media and the outbox survive process death.

Any data loss, wrong-child pose selection, silent required-view bypass,
unlabelled estimate, missing provenance, or deletion before acknowledgement is
an immediate no-go.

## 7. Pilot evidence and promotion boundary

For paired same-visit measurements, report capture success/rejection rate,
height and weight MAE, signed bias and Bland–Altman limits, stunting/wasting
sensitivity and specificity, and SAM, SAM+MAM, and severe-stunting false
negatives. Stratify by age band, sex, operator code, device model, and site
where governance permits.

SAM recall below `0.80` blocks promotion. Meeting that floor does not make the
camera workflow clinical. Broader enablement also requires reviewed agreement,
calibration, failure rate, external validation, privacy/governance evidence,
and an explicit product decision.

Final sign-off:

| Decision | Owner | Date | Evidence link |
|---|---|---|---|
| Android device behavior | | | |
| Offline/sync/retention | | | |
| Wording comprehension | | | |
| Privacy/export review | | | |
| Model metrics and SAM floor | | | |
| Go / limited pilot / no-go | | | |
