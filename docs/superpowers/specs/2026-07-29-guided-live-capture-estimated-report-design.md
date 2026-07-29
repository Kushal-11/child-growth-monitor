# Guided Live Capture, Child Visits, and Estimated Reports — Design

**Date:** 2026-07-29

**Status:** Approved design, pending implementation plan

**Scope:** Extend the Flutter assessment workflow so a field worker can create or select a child profile, collect standardized marker-free live photos without entering measurements, receive an estimated classification report, and add same-day measured details to the same visit later.

## Context

The app already has child profiles, dated visits, offline storage, image-based estimates, classification services, and a feature-gated live camera with pose-quality checks. The existing field dataset also contains front, side, back, arm, and some video views, but the images do not consistently include a known-size marker. Those images remain useful for pose, quality, orientation, and model evaluation, but they cannot establish absolute centimetre scale from image pixels alone.

Future data collection should therefore standardize image capture while preserving honest provenance:

- workers do not enter height, weight, or MUAC during photo capture;
- a successful capture immediately produces a clearly labelled estimate;
- measured details may be added later to the exact same dated visit;
- camera estimates and measured values are stored separately and never overwrite one another;
- the app does not show an `Indeterminate` verdict after a successful photo assessment;
- clinical eligibility remains enforced internally so estimates cannot be mistaken for measured diagnoses.

## Goals

1. Make live capture the preferred path for new assessment photos.
2. Create a reusable, high-quality image dataset tied to stable child profiles and dated visits.
3. Produce an immediate estimated height, weight, stunting, wasting, and SAM/MAM/Normal report from available camera-model outputs.
4. Let an authorized worker add same-day height/length, weight, MUAC, and oedema later.
5. Recompute the measurement-based WHO/Poshan report without deleting the original estimate.
6. Preserve offline-first behavior and idempotent synchronization.
7. Collect sufficient provenance, quality, model, operator, and device metadata for future validation.

## Non-goals

- Claiming that marker-free images directly measure centimetres.
- Treating camera estimates as manual measurements or validated clinical diagnoses.
- Replacing a height board, weighing scale, or MUAC tape.
- Requiring manual measurements during the live capture session.
- Implementing uploaded or retained full-length video in the first release.
- Promoting a new image model to stronger operational use before field validation.
- Retrofitting absolute scale into historical images that lack calibration.

## Chosen approach

Use a **profile-first, dual-source visit**:

1. Create or select the child profile.
2. Create one dated visit and stable local UUID.
3. Collect guided front and side images, with optional back and arm images.
4. Store capture assets, quality metadata, camera estimates, and the estimated report.
5. Allow measured details to be attached later to that same visit.
6. Store and display the measurement-based report separately, with an optional comparison to the estimate.

This avoids child-matching errors, supports longitudinal history, and produces paired image/measurement data for validation.

## Domain model

### Child profile

Reuse the existing child profile as the stable parent record. Required fields are:

- stable local/server identifier;
- name or approved field code;
- date of birth;
- sex.

Guardian name, location, profile image, and notes remain optional. The profile must exist before a capture visit starts.

### Assessment visit

Reuse and extend the existing visit record. A visit represents one assessment occasion and owns:

- child identifier;
- local UUID;
- assessment date and capture timestamp;
- age at the assessment date;
- operator and device metadata;
- capture state;
- synchronization state;
- capture assets;
- one immutable camera-result snapshot;
- current measured details, when supplied;
- measured-detail revision history.

The assessment date is the clinical date used for age and WHO table selection. Manual details entered later must explicitly attach to this visit date. If the measurements were taken on a different date, the app creates a new visit.

### Capture assets

Introduce a dedicated capture-asset record rather than adding more path columns to `Visit`. Each asset stores:

- visit identifier;
- role: `front`, `side`, `back`, `arm_front`, or `arm_side`;
- local path and eventual remote object identifier;
- capture timestamp;
- selected-frame rank;
- pose, coverage, orientation, sharpness, lighting, and overall quality scores;
- quality verdict and rejection reason;
- image dimensions and orientation;
- device/camera metadata available without additional permissions;
- sync state.

Required assets are `front` and `side`. Back and arm views are optional in the first release. The schema supports a later calibration-marker or depth field without changing visit/report semantics.

### Camera result

The camera result is a snapshot of the successful capture-time inference. It stores:

- estimated height and weight;
- estimated HAZ and WHZ values when calculable;
- estimated stunting and wasting statuses;
- estimated overall SAM/MAM/Normal screening category when the active camera model supplies one;
- component probabilities and confidence;
- body-proportion features used by the model;
- capture-quality summary;
- method identifier, such as `camera_screening_v1`;
- model version, manifest checksum, and training-data label;
- `non_clinical=true`;
- creation timestamp.

Camera-derived HAZ/WHZ and statuses must use dedicated estimated fields. They must not be written into authoritative measured HAZ/WHZ or Poshan fields.

The camera overall category is an experimental screening output. It is not labelled `poshan_setu_v1`, because Poshan Setu requires eligible measured or separately validated inputs.

### Measured details

Measured details are optional and may be added after capture. They store:

- standing height or recumbent length;
- measurement mode: `standing_height` or `recumbent_length`;
- weight;
- tape MUAC;
- bilateral pitting oedema: `yes`, `no`, or `not_checked`;
- notes;
- measurer/editor identifier;
- entry timestamp.

Each value keeps `manual`/`tape` provenance. Updates create an audit record containing the previous values, new values, editor, timestamp, and reason when supplied.

## Capture workflow

### Entry

The worker starts from a child profile and selects **New photo assessment**. The app creates a draft visit locally before opening the camera so interruption cannot orphan unlinked images.

No height, weight, or MUAC fields appear in the capture sequence.

### Capture sequence

1. Required front full-body view.
2. Required side full-body view.
3. Optional back full-body view.
4. Optional front upper-arm view.
5. Optional side upper-arm view.
6. Review selected images.
7. Generate and save the estimated report.

The front and side roles are required because they provide complementary pose and body-depth information. Back and arm roles are collected for future model work and do not block the first-release report.

### Guided quality gate

The existing live pose gate is extended with role-specific checks:

- exactly one child pose;
- expected front or side orientation;
- head and heels inside the image;
- required joints visible;
- sufficient body coverage and centring;
- acceptable landmark confidence;
- acceptable blur/sharpness;
- acceptable brightness and contrast;
- excessive phone tilt rejected where sensor data is available;
- sustained acceptable quality across consecutive frames.

The screen gives one actionable instruction at a time, such as “Move back,” “Show both feet,” “Turn sideways,” or “Hold the phone steady.”

### Burst selection

After a stable quality streak, capture a short still-image burst rather than retaining a video. Score the burst frames, retain the best few required for research and inference, and discard low-ranked temporary frames after the visit is durably saved.

This provides multi-frame robustness while limiting storage, upload cost, and retention of unnecessary child imagery.

## Marker-free estimation boundary

The first release does not require a calibration card or mat. Consequently:

- pixel lengths are not presented as direct centimetre measurements;
- any centimetre value comes from the current statistical/model estimator and carries its method;
- confidence reflects model and capture quality, not metrological accuracy;
- the result screen describes values as estimated from photos;
- the data model remains ready for a future calibrated-marker or depth source.

A future validated calibration method must receive a new source name, such as `calibrated_marker_estimated`. It must not inherit clinical eligibility from the legacy generic `reference_object` label.

## Estimated report

After successful inference, show **Estimated Growth Screening Report** as the primary result. It contains:

- estimated height and weight;
- estimated stunting and wasting status;
- estimated SAM/MAM/Normal screening category when available;
- confidence and capture-quality indicators;
- the main factors or views used;
- “How this was estimated,” including method and model version;
- the notice: “Results are estimated from photos and may change after measured details are added”;
- **Add Measured Details** action.

The successful user-facing estimate does not display `Indeterminate`.

Internally, estimated values remain ineligible for authoritative measured classification unless their source is separately validated and explicitly promoted. This internal eligibility state prevents exports, sync, APIs, or downstream screens from treating an estimate as a measured diagnosis.

If the active estimator cannot produce a component, omit that estimated component and say it could not be estimated from the captured views. Do not fabricate a Normal result.

## Adding measured details later

The child timeline labels a camera-only visit as **Estimated report** and exposes **Add Measured Details**.

When measured details are saved:

- height/length alone can produce measured HAZ/stunting;
- height/length plus weight can produce measured WHZ/wasting;
- tape MUAC can produce its eligible component for ages 6 through 59 completed months;
- oedema can independently trigger an actionable WHO acute-malnutrition SAM result;
- current Poshan Setu completeness and severity rules remain unchanged internally;
- missing measured components are displayed as **Not measured**, not `Indeterminate`.

The result screen then makes **Measurement-Based Report** primary and moves the original camera result under **Compare with estimate**.

The measurement-based report keeps three interpretations separate:

- WHO HAZ stunting status;
- WHO acute-malnutrition status, using eligible WHZ, tape MUAC, and oedema signals;
- Poshan Setu v1, using its existing BMI and tape-MUAC contract.

Oedema participates in the WHO acute-malnutrition result but does not silently alter the separately named Poshan Setu v1 calculation.

The comparison shows, only to authorized users:

- estimated and measured values;
- absolute and signed differences;
- classification agreement;
- capture/model version used for the estimate.

Camera outputs are immutable for the visit. Reprocessing with a new model creates a new camera-result version rather than silently rewriting the original field result.

## Report and visit states

Use explicit states:

- `draft_capture`: visit exists but required views are incomplete;
- `incomplete_capture`: worker saved after required views could not pass;
- `processing`: usable photos saved and inference running;
- `estimated_report`: camera estimate available;
- `processing_failed`: usable photos saved but inference failed;
- `measured_report`: measured details added and report recomputed.

The UI labels are:

- **Incomplete capture**
- **Processing estimate**
- **Estimated report**
- **Estimate failed — retry**
- **Measured report added**

## Failure handling

### Capture failure

Quality failures produce a role-specific retake instruction. Repeated failure may be saved as `incomplete_capture`, preserving the visit and failure reasons. No camera classification is produced from unusable required views.

### Inference failure

If inference fails after images were durably saved:

- keep the child, visit, assets, and metadata;
- set `processing_failed`;
- show a retry action;
- never substitute a WHO median or fixed fallback without identifying it as the estimator method;
- never fabricate a classification.

### Manual-entry failure

Validate finite values, plausible ranges, age eligibility, and measurement date before saving. A failure changes neither the existing camera result nor the previous measured report.

### Synchronization failure

Keep data locally and retry through the sync queue. Use child and visit UUIDs as idempotency keys so reconnecting cannot duplicate profiles, visits, assets, or measured revisions.

## Offline-first data flow

1. Create/select child locally.
2. Create draft visit with UUID.
3. Capture and score assets locally.
4. Save selected assets and metadata.
5. Run available on-device inference and store the camera snapshot.
6. Queue child, visit, asset, and result synchronization.
7. Add measured details locally later if supplied.
8. Recompute the measured report locally from the same authoritative rules.
9. Queue the measured update and audit revision.
10. Server validates provenance and recomputes authoritative measured classifications.

Local media is not deleted until the server confirms durable receipt. Sync acknowledgements must identify every accepted asset/result rather than treating a partial upload as full success.

## Privacy and consent

- Obtain and record caregiver consent before the first child image is captured.
- Explain that images are used for estimated screening and model evaluation.
- Restrict profiles and identifiable media to authenticated, authorized users.
- Encrypt transport and protect stored media using platform/server controls.
- Use stable child IDs rather than names in research exports.
- Exclude guardian names, profile names, and direct identifiers from exported model datasets.
- Provide explicit media-retention and deletion controls.
- Deleting media must not silently delete measurement history; deleting a child profile remains a separate, confirmed action.
- Record the consent version, timestamp, and operator.

## Validation and model-promotion boundary

Marker-free guided capture is initially a research and screening feature. Its field validation must use later-added same-day measurements linked to the same visit.

Evaluation splits must be by child, with a separate location where feasible. Report:

- capture success and rejection rate;
- time and retry count per required view;
- height and weight MAE;
- signed bias and Bland–Altman limits of agreement;
- stunting and wasting sensitivity/specificity;
- SAM, SAM+MAM, and severe-stunting false negatives;
- performance by age band, sex, operator, device, and location;
- missing/manual-follow-up coverage.

SAM recall below 0.80 blocks promotion. Passing that floor alone does not make the method clinical; agreement, calibration, failure rate, external validation, and governance must also be acceptable.

## Testing

### Flutter unit tests

- every capture-quality rejection and instruction;
- role-specific front/side orientation;
- burst ranking and deterministic selection;
- visit-state transitions;
- camera/measured provenance isolation;
- partial measured-input classification behavior;
- estimate/model version immutability;
- sync serialization for every new record.

### Flutter widget tests

- create/select profile → front capture → side capture → estimated report;
- optional-view skip and capture paths;
- interrupted and resumed capture;
- repeated quality failure → incomplete visit;
- inference failure → retry;
- estimated report → add measured details → measured report;
- partial measured details and “Not measured” labels;
- offline creation and queued synchronization.

### Backend tests

- Pydantic request/response compatibility with Dart models;
- owner-scoped profile, visit, asset, result, and revision access;
- idempotent child/visit/asset synchronization;
- duplicate and partial-upload recovery;
- server-side provenance enforcement;
- authoritative WHO/Poshan recomputation from measured values;
- rejection of estimated fields submitted as manual;
- audit-history preservation.

### Device and field validation

- physical Android device capture;
- camera rotation and lifecycle interruption;
- low light, background people, blur, cropped body, and wrong orientation;
- low-memory and lower-end device behavior;
- offline capture followed by delayed synchronization;
- local asset retention until confirmed upload;
- field-worker comprehension of “estimated” versus “measured.”

## Rollout

1. Keep the enhanced live capture behind `LIVE_CAPTURE`.
2. Enable it for a controlled field-data pilot.
3. Review capture failures, operator/device effects, paired measurement coverage, and model metrics.
4. Tune quality thresholds only through a versioned configuration and freeze them before formal evaluation.
5. Enable more broadly only after device and field evidence is reviewed.

The first rollout does not promote marker-free outputs to measured or clinical provenance.

## Acceptance criteria

1. A worker can create/select a child and complete required front/side capture without entering measurements.
2. Poor captures receive actionable guidance and cannot silently generate a result.
3. A successful capture produces an estimated report without displaying `Indeterminate`.
4. Every displayed estimate includes method/model provenance and a concise estimate notice.
5. Manual height/length, weight, MUAC, and oedema can be added later to the exact visit date.
6. Adding measured details produces a separate measurement-based report and retains the original estimate.
7. Missing manual components display **Not measured** and do not become fabricated Normal values.
8. Camera estimates cannot populate authoritative measured WHO/Poshan fields.
9. Visits and assets synchronize idempotently after offline use.
10. Research export pairs images and measurements by child/visit ID while excluding direct identifiers.
11. The feature remains gated until Android device and controlled field-pilot evidence are reviewed.
