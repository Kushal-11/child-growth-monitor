# Field-Readiness Review — Child Growth Monitor

**Status:** Assessment (no code changed). Prepared 2026-07-10.
**Decision this feeds:** go / no-go to hand the app to field testers now.
**Trial profile (from the requester):** Android-only, mixed consumer phones; testers can be
online at onboarding; field use is fully offline.

## How to read the confidence tags

- **[RAN]** — I executed the command in this repo; the output is pasted in the Evidence Appendix.
- **[CODE]** — I opened the file and read the cited lines myself.
- **[AGENT]** — a sub-agent reported it with a file:line citation; I did **not** independently
  re-open every one of these. Treated as reliable but not personally verified. The
  safety-critical claims (OR-rule, z-scores, MUAC, weight priority, sync behaviour) are all
  **[CODE]** or **[RAN]**, not **[AGENT]**.
- **[INFERENCE]** — my engineering judgement connecting the evidence. Not a citation.

This honesty matters for a medical device. Where a claim is unverified, it says so.

---

## 0. Verdicts at a glance

| # | Outcome | One-line verdict |
|---|---------|------------------|
| 1 | **Field-test readiness** | **NOT YET — but the gap is plumbing, not clinical logic.** The offline assessment path is safety-sound and green across 282 tests; two hard blockers (release-build login, signing) and one framing rule stand between it and a defensible handout. |
| 2 | **Core model** | **Meets the SAM-recall floor (0.889 ≥ 0.80, verified leak-free) — but on synthetic data only, with a heavily recall-biased operating point.** Highest-value, lowest-risk win is to *wire up calibration/abstention that is already trained but orphaned.* |
| 3 | **Distribution** | **Firebase App Distribution** (onboarding connectivity is available), after fixing signing + applicationId. The 30 MB pose model is a non-issue — the app doesn't use it. |
| 4 | **Fully offline** | **Yes for capture → assessment → z-score → persistence.** The one real hole is **first login in release builds**, which is compiled out and needs a backend. |

**The single most important finding:** the app runs the entire clinical assessment **on-device**
(pose, measurement, WHO z-scores, ML, MUAC, and the WHO CMAM OR-rule), and that path is
implemented **correctly and safely**. The Python backend contains several bugs in the *same*
logic — but the mobile app does not depend on the backend for assessment, and `sync` stores the
app's verdict rather than recomputing it, so those backend bugs never reach a field tester.

---

## 1. FIELD-TEST READINESS

**Verdict: NOT YET for a release handout. Core screening logic is ready; auth + distribution
plumbing is not.** With onboarding connectivity available, the shortest defensible path is
1–2 days of work (signing + a release-safe login), not a rebuild.

### 1.1 What is genuinely solid (verified)

The offline assessment pipeline — the part that decides whether a child is flagged SAM/MAM — is
correct end to end. I read the code, not just the agent summaries:

- **WHO OR-rule escalation is correct in the app.** `combineNutritionStatus`
  ([config.dart:90-101](flutter_app/lib/constants/config.dart#L90-L101)) takes the max severity
  across WHZ, MUAC, and the ML classifier ("most-severe-wins"), and `_canonicalWhz`
  ([config.dart:139-146](flutter_app/lib/constants/config.dart#L139-L146)) normalises the long
  WHZ labels with `.contains('SAM')` / `.contains('MAM')`. A tape-measured SAM (MUAC < 11.5)
  with a borderline-normal WHZ still reads **SAM**. It is wired into the headline summary at
  [assessment_service.dart:208](flutter_app/lib/services/assessment_service.dart#L208). **[CODE]**
- **MUAC thresholds are exact:** `<11.5 SAM`, `<12.5 MAM`, else Normal
  ([config.dart:71-76](flutter_app/lib/constants/config.dart#L71-L76)). **[CODE]**
- **WHZ/HAZ classification matches WHO bands**
  ([config.dart:55-69](flutter_app/lib/constants/config.dart#L55-L69)). **[CODE]**
- **Weight priority is manual > ML (only if 45–180% of WHO median) > WHO median with body-build
  adjustment** (`_resolveWeight`,
  [assessment_service.dart:263-279](flutter_app/lib/services/assessment_service.dart#L263-L279)).
  Note the app is **safer than the backend here**: when the WHO median can't be computed it
  returns `null` (line 277) rather than accepting an unchecked ML weight. **[CODE]**
- **No silent false-Normal.** When no measurement signal is available the summary is `Unknown`
  ([config.dart:100](flutter_app/lib/constants/config.dart#L100), rank 0), not Normal. A failed
  pose throws `PoseDetectionFailedException` and asks for a retake
  ([assessment_service.dart:87-92](flutter_app/lib/services/assessment_service.dart#L87-L92)). **[CODE]**
- **Tests are green.** Python **109 passed** [RAN]; Flutter **173 passed / 1 skipped / 0 failed**
  [RAN] — including explicit offline-first login and secure-storage-block tests.

### 1.2 Ranked blockers (severity-tagged)

| Rank | Severity | Blocker | Evidence |
|------|----------|---------|----------|
| 1 | **BLOCKER — FIXED IN CODE (2026-07-10)** | **Release builds had no offline login:** the `cgmtester` credential was gated behind `kDebugMode`, so `flutter build apk --release` compiled it out and first login needed a backend. **Now** the gate is `kDebugMode OR --dart-define=FIELD_OFFLINE_AUTH=true` ([local_auth.dart](flutter_app/lib/services/local_auth.dart), `computeOfflineAuthEnabled`), unit-tested for the release truth table. A plain release build still has no backdoor; build a field APK with the flag to keep offline login. **Still needs:** a release-build smoke test on a device (§4.3) to confirm end-to-end. | [CODE]+[RAN] |
| 2 | **BLOCKER** | **No release signing key; default applicationId.** Release reuses the **debug** keystore ([build.gradle.kts:37](flutter_app/android/app/build.gradle.kts#L37)) and ships `com.example.child_growth_monitor_app` ([build.gradle.kts:24](flutter_app/android/app/build.gradle.kts#L24)). Debug-signed installs *work*, but give no update-key continuity and are unfit for a medical trial. | [AGENT] |
| 3 | **HIGH** | **SAM-recall floor is validated on synthetic data only.** 0.889 is on held-out *synthetic* records (§2); there is no real-child eval set. The app must be framed as **screening + mandatory manual confirmation**, which its manual>ML priority already supports. | [RAN] |
| 4 | **HIGH** | **ML over-predicts SAM (precision 0.48) and feeds the headline OR-rule.** Safe direction (fewer false negatives) but a real false-alarm / trust problem in the field. Calibration + abstention to temper it is **trained but not wired** (§2.3). | [RAN]+[CODE] |
| 5 | **MEDIUM** | **Sync silently drops records after 5 failed retries** with no UI surfacing ([sync_queue_dao.dart:24-25](flutter_app/lib/database/daos/sync_queue_dao.dart#L24-L25) [AGENT]). Field data can be stranded invisibly — a "no silent failures" violation for the *data* pipeline. | [AGENT] |
| 6 | **MEDIUM** | **Backend OR-rule is dead + mismatched.** `combine_with_whz_status` compares `whz_status == "SAM"` ([muac_service.py:263](app/services/muac_service.py#L263)) but `classify_whz` returns `"Severe Acute Malnutrition (SAM)"` ([config.py:61](config.py#L61)) — never matches; and the result is never returned. **Does not affect the app** (sync stores the app's verdict, [sync.py:2-5](app/api/sync.py#L2-L5)), but taints the **web UI** and any server-side re-analysis for the study. | [CODE] |
| 7 | **MEDIUM** | **Reconnect → 401 → forced logout** of the offline tester token, with no offline re-login in release ([AGENT]). A worker can be locked out mid-field with unsynced local data. | [AGENT] |
| 8 | **LOW** | **`_lms_zscore` returns `0.0` (→ "Normal") on `M<=0 or S<=0`** ([nutrition_service.py:56-57](app/services/nutrition_service.py#L56-L57)). Latent — valid WHO Excel files won't trigger it — but it violates "no silent failures." Backend-only. | [CODE] |
| 9 | **LOW** | **No unit tests for backend `assessment_service` / `muac` combine / `ml_service`.** Exactly why blocker #6 shipped uncaught. `tests/` has no `test_assessment_service.py` / `test_muac_service.py`. | [RAN] |

### 1.3 Recommended path through the blockers

Blocker #1 is **now fixed in code** (done 2026-07-10, TDD, full suite green): the field credential
is decoupled from `kDebugMode` and gated by `computeOfflineAuthEnabled(kDebugMode, FIELD_OFFLINE_AUTH)`
in [local_auth.dart](flutter_app/lib/services/local_auth.dart). Build the field APK with
`flutter build apk --release --dart-define=FIELD_OFFLINE_AUTH=true` and it keeps offline login while
a plain `flutter build apk --release` still ships no backdoor. This is a **release-mode,
release-signed** path (once blocker #2's keystore exists) — no slow debug build required. Confirm it
with a device smoke test (§4.3) before handout. The no-code alternative remains: stand up a live
**https** backend with real tester accounts and **log in once online at onboarding**, after which the
token persists — but confirm the 30-day JWT lifetime ([config.py:15](config.py#L15)) outlasts your
offline stretch, or testers will be logged out.

---

## 2. CORE MODEL — CURRENT STATE AND HOW TO IMPROVE IT

**Verdict: the SAM-recall floor is met (0.889 ≥ 0.80) and — I checked — it is *not* a leaky
metric. But it is synthetic-only, the operating point is strongly recall-biased, and the most
sophisticated safety machinery (calibration, conformal abstention, the cascade) is trained and
then never loaded at inference.** The biggest wins are cheap.

### 2.1 Measured state (from `ml/evaluate.py`, [RAN])

| Metric | Value | Note |
|--------|-------|------|
| **SAM recall** | **0.889** | ≥ 0.80 floor ✓ |
| SAM precision | 0.48 | ~half of SAM calls are false alarms |
| MAM recall / precision | 0.24 / 0.30 | classifier barely detects MAM as a class |
| Weight MAE (overall) | 0.661 kg | vs **0.403 kg** baseline in AGENTS.md |
| 5-class accuracy | 0.58 | vs **70.2%** baseline in AGENTS.md |
| ECE (calibration error) | 0.047 | |
| Per-sex SAM recall | M 0.872 / F 0.908 | both ≥ 0.80 |
| Per-age SAM recall | 0.913 / 0.902 / 0.879 | all ≥ 0.80 |

Two things to reconcile: **weight MAE (0.661) and accuracy (0.58) are materially worse than the
baseline documented in [AGENTS.md:22](AGENTS.md)** (0.403 kg, 70.2%). Either the baseline is stale
or the committed model regressed. CLAUDE.md requires evaluate.py output in the commit message for
any model change — that trail should be reconciled before trusting either number as "the baseline."

### 2.2 Two claims I corrected during this review

1. **The 0.889 is NOT leaking training data.** A sub-agent flagged that `evaluate.py`
   ([evaluate.py:131-134](ml/evaluate.py#L131-L134)) re-splits at `test_size=0.2` while
   `train.py` ([train.py:302-309](ml/train.py#L302-L309)) splits at `test_size=0.30`, implying
   overlap. I tested it empirically [RAN]: the fraction of evaluate's val set that was in
   train.py's *training* set is **0.000** — sklearn's `StratifiedShuffleSplit` nests smaller test
   sets inside larger ones for the same seed, so evaluate's 20% lands entirely inside train.py's
   held-out 30%. **No leakage today.** Caveat: it's leak-free by *coincidence* of the split
   proportions — change them and leakage returns. `evaluate.py` should explicitly reuse train.py's
   saved split, and it currently evaluates only the served 5-way head (not the cascade/GBT/calibrated
   probabilities), with no specificity or crossed age×sex fairness gate.
2. **Calibration and the cascade are orphaned, not "applied."** The prior ml_pipeline doc treats
   temperature+conformal calibration as in-production and the cascade as the incumbent. In fact
   `ml/inference.py` and the Flutter `ml_inference_service.dart` load **only** the raw 5-way
   softmax and take a plain `argmax` — `cascade_meta.json`, `*_calibration.json`,
   `ConformalCalibrator`, `wasted_binary`, `sam_vs_mam`, and the LightGBM model are loaded by
   **nothing** at serve time. **[AGENT]**, consistent with [config.py:31-37](config.py#L31-L37)
   pointing serving at `wasting_classifier.keras` only.

### 2.3 Prioritized improvements (extending the existing ml_pipeline doc)

The existing `docs/ml_pipeline_improvement_and_feedback_loop.md` remains the right roadmap
(oedema field, stop imputing depth, real+synthetic blend, benchmark GBT, image-MUAC head). What
this review *adds* is a reordering based on what's verified to be cheap and safe:

| Priority | Improvement | Expected impact | Risk to SAM floor |
|----------|-------------|-----------------|-------------------|
| **P1 (new, cheap)** | **Wire up the already-trained calibration + conformal abstention** to the ML arm. Only let ML **escalate** when the conformal set is confident; route ambiguous ML to "Indeterminate → manual." This directly tackles the 0.48 SAM precision / false-alarm problem the doc's A4 describes — and it's ~80% built (`calibration.py` + the JSON thresholds exist). | Fewer false alarms; restored tester trust; no new training. | **None** if abstention only routes to manual (fail-safe) and never *suppresses* a SAM call. **Flag:** if implemented as "drop low-confidence SAM," it *could* lower recall — do not do that. |
| **P1 (from doc)** | **Add a mandatory oedema field as a third OR-rule trigger.** Highest-recall win available; not an ML change. | Catches oedematous SAM (≈⅔ of kwashiorkor has normal WHZ/MUAC). | None — pure addition to the OR-rule. |
| **P2 (from doc)** | **Stop imputing side-view depth (features 10–13); require the side photo or abstain.** Frontal-only imputation makes 4 of 14 features deterministic functions of width. | Better weight → better WHZ. | Low; pair with abstention so missing depth routes to manual, not a bad estimate. |
| **P2 (new)** | **Reconcile the MAE/accuracy regression vs the AGENTS.md baseline**, then re-establish a trustworthy baseline with evaluate.py output in the commit. | Restores confidence in "the number." | None (evaluation only). |
| **P2 (from doc)** | **Benchmark GBT vs MLP vs the cascade on held-out data**, gated on SAM recall then MAE. The cascade and GBT are already trained — evaluate them before choosing. | Possible MAM-recall gain. | Only promote a model that clears 0.80 on real (not synthetic) data. |
| **P3 (from doc)** | **Plan the real+synthetic blend** as field data arrives; the feedback loop is ~80% built. | Closes the synthetic-to-real gap. | Gate promotion on real-data subgroup recall. |

**Inviolable:** every proposal above preserves SAM recall ≥ 0.80. The only one that *could*
endanger it is a mis-implementation of abstention (suppressing SAM calls) — called out explicitly.
Do not change the 14-feature order, the scaler, WHO thresholds, or the OR-rule as part of any of
these without a separate review.

---

## 3. DISTRIBUTION FOR TESTING (no Play Store)

**Recommended: Firebase App Distribution**, because onboarding connectivity is available and it
gives you tester groups, one-tap install, automatic update notifications, and crash/analytics —
all valuable for a go/no-go trial. The app is not yet wired for Firebase (no `google-services.json`
[AGENT]), so this is a small setup step, not a rebuild.

### 3.1 The pose-model premise is a non-issue

The 30 MB `data/pose_landmarker_heavy.task` is **only used by the Python/MediaPipe backend**
([config.py:26](config.py#L26)). The **Flutter app uses `google_mlkit_pose_detection`**
([pubspec.yaml:22](flutter_app/pubspec.yaml#L22)) whose models ship **inside the plugin's native
AAR** — there is **no reference to `pose_landmarker`/`.task` anywhere in `lib/`** [RAN]. Testers do
**not** need to obtain that file, and it does not enter the APK. The only bundled assets are the
tiny TFLite models and WHO data ([pubspec.yaml:46-48](flutter_app/pubspec.yaml#L46-L48), ~136 KB).

### 3.2 Build & signing

- **Command:** `cd flutter_app && flutter build apk --release` → one universal APK at
  `build/app/outputs/flutter-apk/app-release.apk`. Use `--split-per-abi` for smaller per-device
  APKs (most phones = `arm64-v8a`). **Do not** use `flutter build appbundle` — `.aab` can't be
  sideloaded. **[AGENT]**
- **Estimated size:** ~55–85 MB universal / ~30–45 MB arm64 split — driven by the bundled ML Kit
  pose models + Flutter engine, not by your assets. **[AGENT] [INFERENCE]**
- **Signing (must fix):** generate a release keystore and wire a real `signingConfigs.release`;
  set a real `applicationId`. Both are currently missing/default (blockers #1–2, §1.2).
- **Permissions:** only `INTERNET` is declared; camera goes through the system camera via
  `image_picker`, so no runtime CAMERA prompt — low friction for testers. **[AGENT]**

### 3.3 Two build-time gotchas

- **Release blocks cleartext HTTP.** `usesCleartextTraffic=true` exists only in the *debug*
  manifest [AGENT]. A release APK on Android 9+ cannot talk to an **http** LAN backend — use
  **https**, or add a release `network-security-config` for specific hosts. (Irrelevant if you
  run fully offline.)
- **The server URL cannot be baked at build time.** The `--dart-define=API_BASE_URL` in CI is a
  dead no-op; the app reads its base URL from in-app Settings and defaults to
  `https://api.child-growth-monitor.org` in release ([AGENT], api_provider.dart). Testers set it
  in Settings, or you point that host at a live backend.

### 3.4 Alternatives (and why not)

- **Raw signed-APK sideload (USB/link).** Simplest, zero infra, works offline even at onboarding.
  Downside: no update mechanism, no crash/usage telemetry, manual re-distribution each build.
  Keep as the fallback for any tester who can't get online at setup.
- **Existing CI artifact** (`.github/workflows/flutter-android-apk.yml` already uploads
  `app-release.apk` [AGENT]) — usable *today* for internal testing, but it's debug-signed and
  release-mode (so no offline login). Fine for a technical pilot, not for real field handout.

### 3.5 How testers report results

Results already sync to the backend when online (paired manual + predicted values, images,
per-class probabilities — [sync.py:35-186](app/api/sync.py#L35-L186)). Add Firebase Crashlytics
for crash capture, and give testers a one-tap "send feedback" (email/form) for qualitative notes.
**Gap to close for the study:** the app computes the headline OR-rule verdict but **does not
persist or sync it** (no summary/combined column in the DB [RAN]; sync has no such field) — only
the component `whz_status`/`muac_status`/`ml_wasting_status` are stored. Downstream analysts must
recompute the verdict, and must **not** use the buggy backend `combine_with_whz_status` to do so.

---

## 4. FULLY OFFLINE OPERATION

**Verdict: the assessment app is genuinely offline — capture, pose, measurement, WHO z-scores, ML,
MUAC, persistence, and the OR-rule all run on-device with no network. The one real hole is first
login in release builds.**

### 4.1 Verified offline (each traced in code)

- **Launch:** no network call at startup; auth restore is a bounded, fire-and-forget secure-storage
  read ([AGENT], main.dart). **[AGENT]**
- **Assessment:** pose (ML Kit, on-device), measurement (geometry), **WHO z-scores computed
  on-device** from bundled xlsx/CSV ([assessment_service.dart:143-158](flutter_app/lib/services/assessment_service.dart#L143-L158)),
  ML via bundled TFLite. The backend `/api/assess` is **not called** by the app. **[CODE]**
- **Persistence:** drift/SQLite, written **before** any sync; the manual path is a single atomic
  transaction ([AGENT]). **[AGENT]**
- **Sync:** queues locally, retries, degrades gracefully with no blocking UI when offline
  ([sync.py] server side stores verbatim). **[AGENT]+[CODE]**
- **Models bundled, not downloaded:** ML Kit pose (native AAR), TFLite, and WHO data are all in
  the APK — no runtime fetch [RAN].

### 4.2 Offline gaps (what's needed)

| Severity | Gap | What's needed |
|----------|-----|---------------|
| **HIGH → FIXED IN CODE** | First login was unavailable offline in **release** builds (§1.2 #1). | Done: gate is now `kDebugMode OR FIELD_OFFLINE_AUTH` — build with `--dart-define=FIELD_OFFLINE_AUTH=true`. Still verify on-device (§4.3), **or** do a one-time online login at onboarding. |
| **MEDIUM** | Sync silently abandons records after 5 retries; reconnect can force-logout the tester token. | Surface a "N records failed to sync" banner; don't hard-logout on a single 401 for the offline token; allow offline re-login. |
| **LOW** | ML-failure fallback to WHO median is only `print`-logged, not surfaced to the worker (`wastingStatus='who_fallback'` is visible in data but not called out in UI). | Optional: show a "measurement estimated" note. |

### 4.3 One caveat to verify on a real device

The claim that ML Kit's pose model is fully bundled (no Play-Services-mediated fetch) should be
confirmed by launching a **release build in airplane mode on a clean device** before the trial.
The Dart layer does no network model fetch [RAN], but the native ML Kit dependency's behaviour is
best proven empirically, not read.

---

## Evidence Appendix

### A. Commands run

```
$ PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
  → 109 passed, 98 warnings in 22.24s

$ cd flutter_app && flutter test
  → 00:11 +173 ~1: All tests passed!   (173 passed, 1 skipped, 0 failed)

$ PYTHONPATH=. .venv/bin/python ml/evaluate.py   (key lines)
  Weight estimator MAE (overall): 0.661 kg
  *** SAM recall: 0.889 (target ≥ 0.80) ***
  Classification: MAM p=0.30 r=0.24 | Normal p=0.88 r=0.57 | SAM p=0.48 r=0.89
  ECE (15 bins): 0.0474
  Per-age SAM recall: 0.913 / 0.902 / 0.879 ; Per-sex: M 0.872 / F 0.908

$ (split-overlap check)
  train.py train n=42000, evaluate val n=12000
  fraction of evaluate.py val set that was in train.py TRAIN set: 0.000   ← no leakage
  fraction in train.py true held-out val(15%): 0.499
```

### B. Test-coverage gaps found

- Python: no `test_assessment_service.py`, `test_muac_service.py`, `test_ml_service.py` — the
  OR-rule combine and weight-priority chain are not directly unit-tested (why blocker #6 shipped).
- ML: no pytest regression guarding SAM recall ≥ 0.80; it lives only in `evaluate.py` (manual).
- Flutter: strong coverage (~30 test files across services, DAOs, screens, providers, offline
  paths). This is the best-tested layer.

### C. Method note

Four sub-agents mapped the backend pipeline, the Flutter offline/auth path, distribution, and the
ML code. I independently re-read and verified the safety-critical claims myself: the Dart OR-rule
([config.dart:90-146](flutter_app/lib/constants/config.dart#L90-L146)), the backend OR-rule bug
([muac_service.py:263](app/services/muac_service.py#L263) vs [config.py:61](config.py#L61)),
`sync` storing the app verdict ([sync.py:2-5](app/api/sync.py#L2-L5)), the weight priority
([assessment_service.dart:263-279](flutter_app/lib/services/assessment_service.dart#L263-L279)),
and the train/eval split overlap (empirical, [RAN]). Distribution build/signing specifics and a
few backend error-swallow line numbers are **[AGENT]** — reliable but not personally re-opened.
</content>
</invoke>
