# Contactless ARCore Anthropometry Estimates

**Date:** 2026-08-07  
**Status:** Approved direction; implementation design  
**Scope:** `flutter_app/` Android guided-capture workflow

## 1. Product decision

The app is an estimation tool. On a supported Android device, one guided ARCore
capture should attempt to produce and visibly report all three contactless
estimates:

- height in centimetres;
- weight in kilograms;
- mid-upper-arm circumference (MUAC) in centimetres.

The estimates remain explicitly labelled as estimates. That label must not hide
or suppress a valid value. Manual measurements, when entered later, remain
separate comparison values and do not overwrite the original estimate.

ARCore measures scene geometry, not body mass. Height and MUAC are geometric
estimates. Weight is an on-device model inference from the measured geometry,
age, and sex.

## 2. Current state and gap

The released `arcore_guided_depth_v2` scan already acquires raw depth and depth
confidence, finds a floor plane, filters a bounded child point set, and applies
multi-frame movement and stability gates. It returns only height.

The height result is currently written into `visits.device_metadata` and shown
inside the scan card. It is not promoted into the immutable camera-result
record, estimated report, weight feature vector, or MUAC calculation.

The existing weight model accepts the exact geometry needed for an initial
contactless implementation:

1. age and sex;
2. height;
3. shoulder and hip width;
4. torso and upper-arm length;
5. chest and abdominal anterior-posterior depth;
6. ratios and body-build score derived from those values.

The existing MUAC fallback estimates circumference from WHO-centred landmark
ratios. It does not use an observed arm cross-section.

## 3. Goals

1. Produce height, weight, and MUAC estimates from a single guided AR session on
   ARCore Depth API devices.
2. Use observed depth geometry in place of WHO-median scaling wherever the scan
   has sufficient component-specific evidence.
3. Show each successful value prominently with method, quality, and an
   estimated range.
4. Preserve partial success. A failed arm slice must not discard a valid height
   or weight estimate.
5. Keep all computation offline and on-device.
6. Retain only bounded derived measurements and quality summaries. Do not retain
   RGB frames, raw depth images, a point cloud, or a mesh.
7. Preserve standard guided-photo fallbacks for unsupported phones or failed
   AR components.

## 4. Non-goals

- Claiming that ARCore directly measures body mass.
- Claiming a clinical or diagnostic accuracy level before paired field
  evaluation.
- Reconstructing or storing a photorealistic body mesh.
- Requiring a manual height board, weighing scale, MUAC tape, reference card, or
  physical contact to obtain an estimate.
- Replacing the separately stored manual-measurement workflow.

## 5. Guided scan protocol

The existing short 20-35 degree arc is sufficient for repeated height evidence
but not for torso depth or an arm cross-section. The contactless scan becomes a
single continuous session with three guided checkpoints.

### 5.1 Starting pose

- The child stands on a visible, approximately level floor.
- The whole body, including head and both feet, remains visible.
- Arms are held slightly away from the torso, with elbows approximately
  straight. The UI demonstrates this pose before the scan starts.
- The operator starts from the front at approximately 1.2-2.5 metres.

### 5.2 Capture checkpoints

1. **Front lock:** collect stable pose, floor, height, shoulder, hip, and limb
   evidence.
2. **Oblique sweep:** move slowly toward the child's left side while keeping the
   whole child centred. Target at least 70 degrees and 0.5 metres of accepted
   camera travel.
3. **Side lock:** collect anterior-posterior chest and abdomen depth plus upper
   arm surface evidence. Target 90-110 degrees total coverage when the child can
   remain still.

The activity provides component-specific guidance such as “include both feet”,
“move toward the side”, “hold the left arm away from the body”, or “ask the
child to stay still”. It may finish early after all component gates pass, and it
times out with partial results rather than failing the whole visit.

## 6. Native Android geometry pipeline

### 6.1 Inputs

Flutter passes the visit's age in months and sex to `startContactlessScan`.
These inputs are used only by the weight model and plausibility checks; they do
not scale the depth geometry.

### 6.2 Frame acquisition

`FullArScanActivity` continues to use:

- `DepthMode.AUTOMATIC` capability checks;
- `acquireRawDepthImage16Bits()`;
- the matching raw-depth confidence image;
- camera intrinsics and pose;
- a tracked horizontal floor plane;
- unique, timestamp-matched depth frames.

A native streaming pose detector supplies shoulder, elbow, hip, knee, heel, and
head landmarks for anatomical localisation. Pose images are closed immediately
after inference and are never written to disk.

### 6.3 Bounded transient representation

Accepted pixels are transformed into floor-relative world points and inserted
into a 15 mm voxel grid. The grid is capped at 60,000 occupied voxels and exists
only in memory for the active session. Each voxel retains position, accumulated
confidence, observation count, and view-angle span.

Points are associated with pose-defined body zones before the source frame is
closed:

- head/feet for height;
- shoulder and hip bands for lateral widths;
- chest and abdomen bands for anterior-posterior depths;
- left and right upper-arm cylinders for MUAC;
- shoulder-to-hip and shoulder-to-elbow landmarks for segment lengths.

Torso-depth clustering and pose zones exclude background surfaces and reduce
arm/torso merging. The activity retains only the voxel summaries and robust
per-frame measurements, not camera media.

## 7. Measurement estimators

### 7.1 Height

Height is the robust distance from the tracked floor to the pose-localised head
surface. The final estimate is the median of accepted keyframe estimates.

Height succeeds when:

- head, both feet, and floor evidence are present;
- at least 12 accepted keyframes contribute;
- floor stability, body drift, depth confidence, and within-scan spread pass;
- the result is within the configured child-height plausibility range.

The estimated range combines robust frame spread, floor-plane spread, and
head/heel localisation spread. It is a scan-consistency range, not a calibrated
population confidence interval.

### 7.2 Body geometry for weight

The scan derives the existing 14-feature model inputs without WHO-median pixel
scaling:

- height from section 7.1;
- shoulder and hip width from front/oblique surface extrema;
- chest and abdominal AP depth from oblique/side surface extrema;
- torso and upper-arm lengths from 3D pose landmarks;
- ratios and body-build score from the measured values.

Flutter constructs `WastingFeatures` from this geometry and runs the shipped
TFLite weight estimator. The initial implementation therefore changes the
input evidence while keeping the released model and feature order compatible.

Weight succeeds when all required geometry is finite, component quality gates
pass, and the predicted value is inside the existing age/height plausibility
envelope. It must not silently substitute a WHO median as a child-specific
weight estimate.

The estimated weight range is produced by deterministic perturbation inference:
run the model over bounded high/low geometry variants derived from each input's
scan spread, take the median as the estimate, and use the central result spread
with a conservative minimum error floor. The UI calls this an “estimated
range”, not a statistical confidence interval.

### 7.3 MUAC

For each upper arm, the estimator defines the 3D axis between shoulder and elbow
and selects a narrow slice at the midpoint. Points are projected onto the plane
normal to that axis. A robust ellipse is fitted after outlier removal, and its
perimeter is calculated using a stable ellipse-perimeter approximation.

The arm with the stronger evidence is selected. MUAC succeeds when:

- shoulder and elbow landmarks are stable;
- the selected slice has sufficient depth points and view-angle coverage;
- ellipse axes and eccentricity are plausible;
- fit residual and between-frame spread pass;
- clothing/torso overlap does not make the slice ambiguous.

The estimated range combines fit residual and between-view spread. If the
depth cross-section fails, the app may show the existing landmark-based MUAC as
a lower-quality fallback, with a different source label.

## 8. Result contract

Replace `FullArScanResult` with a backward-compatible contactless result whose
top-level method is `arcore_contactless_anthropometry_v3`.

```text
ContactlessArEstimate
  height_cm: double?
  height_range_cm: {lower, upper}?
  height_source: arcore_depth | photo_fallback | population_fallback | unavailable

  weight_kg: double?
  weight_range_kg: {lower, upper}?
  weight_source: arcore_geometry_ml | photo_ml | population_fallback | unavailable

  muac_cm: double?
  muac_range_cm: {lower, upper}?
  muac_source: arcore_arm_cross_section | photo_landmark | whz_derived | unavailable

  geometry: bounded 14-feature-compatible measurements
  component_quality: height, weight_geometry, muac
  scan_quality: frames, depth confidence, floor stability, coverage, travel, duration
  model_provenance: model version, manifest checksum, training-data label
  retained_media: false
  is_estimate: true
```

Every numeric field must be finite and pass a domain range check at both the
native-to-Dart boundary and the persistence boundary. Invalid components become
`unavailable`; they do not invalidate independent components.

## 9. Persistence and sync

The immutable `camera_results` record becomes the authoritative home for the
three estimates. Increment the Drift schema and add:

- `estimated_muac_cm`;
- lower and upper range columns for height, weight, and MUAC;
- `muac_source`;
- the AR scan method and component-quality JSON.

The existing `estimated_height_cm`, `estimated_weight_kg`, source, model, and
provenance fields remain compatible. Derived body geometry stays in
`body_proportion_features_json`. The full native scan summary stays in
`capture_quality_summary_json`.

For a new visit, the AR result and guided-photo result are fused into one new,
immutable camera-result version. Reprocessing appends a new UUID/version and
links `supersedes_result_uuid`; it never edits the historical estimate.

The visit metadata copy may remain for backward compatibility, but reports and
sync consumers read the camera-result record.

Backend Dart/Python schemas and sync payloads must be updated together so
estimated MUAC and estimate ranges survive round trips.

## 10. Estimate selection and fallbacks

Selection is per component, not all-or-nothing:

1. Prefer a component that passed the AR depth gate.
2. Otherwise use the existing guided-photo estimate when it is available.
3. A population-derived fallback may be shown only with the explicit source
   “population estimate”; it must never be labelled camera- or depth-measured.
4. Otherwise show “estimate unavailable” and a retry action.

The current manual-measurement record remains separate. The estimated report
may display an optional comparison once manual values exist.

## 11. Estimated report UI

After processing, show three equally prominent cards:

```text
Height  91.4 cm
Estimated range 89.8-93.0 cm
AR depth estimate · quality 84%

Weight  12.6 kg
Estimated range 11.5-13.7 kg
AR geometry + on-device model

MUAC  12.1 cm
Estimated range 11.5-12.7 cm
AR upper-arm cross-section
```

The report also shows:

- a single “Contactless estimates” heading;
- a compact scan-quality summary;
- a retry button for failed or low-quality components;
- the source and model version under expandable details;
- estimated HAZ/WHZ and estimated status only when their required estimated
  inputs exist, clearly grouped under the estimated report.

Avoid language that implies the values were manually measured. Do not hide a
valid estimate merely because it is not a clinical measurement.

## 12. Failure handling

- Unsupported ARCore or Depth API: continue with guided photos.
- Child movement: reject affected frames and give corrective guidance.
- Timeout: return every component that passed; mark the others unavailable.
- Pose detector failure: retain height if depth/floor evidence remains valid;
  reject pose-dependent geometry.
- MUAC arm/torso overlap: request an arm-away retry; never return a fabricated
  cross-section.
- TFLite load/inference failure: retain height and MUAC, mark weight unavailable,
  and expose a retryable error.
- Persistence/sync failure: keep the immutable local result and outbox retry;
  never discard a completed estimate.

## 13. Privacy and resource limits

- No RGB, raw depth, point cloud, or mesh is persisted or included in a method
  channel result.
- Close every ARCore depth/confidence/camera image in the frame that acquired it.
- Cap transient voxels, pose jobs, keyframes, scan duration, and inference
  perturbations.
- Keep only one pose inference and one scan session active.
- Preserve the existing fallback on low-memory or unsupported devices.

## 14. Test strategy

### 14.1 Native unit tests

- robust height aggregation and floor-error propagation;
- checkpoint progress, coverage, movement, and partial timeout;
- 3D body-zone assignment;
- width/depth extraction from deterministic synthetic point sets;
- upper-arm axis, ellipse fit, perimeter, outlier rejection, and ambiguity gates;
- independent component success/failure;
- bounded voxel memory and deterministic summaries.

### 14.2 Flutter unit/widget tests

- strict method-channel parsing and rejection of NaN/infinite/range-invalid data;
- AR geometry to 14-feature ordering;
- deterministic weight perturbation range;
- per-component fallback selection;
- database migration and immutable result versioning;
- sync payload round trip;
- estimated-report rendering of three values, ranges, sources, partial results,
  and retry states;
- unsupported-device guided-photo flow.

### 14.3 Device verification

At minimum, verify on one ARCore device with hardware depth support and one
supported device using motion-derived depth:

- scan launch and pose guidance;
- full-body and arm visibility;
- repeatability across three scans of one subject;
- process death and camera reopen;
- memory, temperature, scan duration, and fallback behaviour;
- installed release APK end-to-end result and persistence.

### 14.4 Paired improvement evaluation

Although the product displays estimates immediately, improvement claims require
paired reference observations. For each of height, weight, and MUAC report MAE,
median/p95 absolute error, signed bias, Bland-Altman limits, repeatability, and
capture success. Stratify by device model, operator, age, sex, clothing, and
site. Keep subject identities out of the exported evaluation set.

## 15. Delivery sequence

1. Add the versioned Dart/native result contract and database migration.
2. Refactor the current height estimator into independent component summaries.
3. Add native pose localisation and bounded voxel/body geometry extraction.
4. Add chest/abdomen/segment geometry and Flutter weight inference.
5. Add upper-arm cross-section MUAC estimation.
6. Fuse AR and guided-photo components into an immutable camera result.
7. Update the estimated report and comparison report.
8. Update backend sync schemas and round-trip tests.
9. Run Flutter analysis/tests, Android unit tests, release build, and installed
   device verification.

## 16. Acceptance criteria

- On a supported device and a successful scan, the estimated report displays
  height, weight, and MUAC with units, ranges, and sources.
- The weight estimate consumes observed AR geometry, not WHO-scaled pixel
  geometry.
- The primary MUAC estimate consumes a depth-derived upper-arm cross-section,
  not WHZ or arm-length ratios.
- A component can succeed independently of the others.
- Unsupported or incomplete scans retain the standard guided-photo fallback.
- No raw camera/depth/point-cloud/mesh data is persisted.
- All new native, Dart, database, sync, and widget tests pass.
- `flutter analyze` passes.
- A release APK builds and is exercised on a compatible Android device before
  calling the feature device-verified.
