# Improving the Malnutrition ML Pipeline + a Safe "Learn-from-Mistakes" Feedback Loop

**Status:** Research report. Prepared 2026-06-09.
**Scope:** (A) where the existing wasting-detection pipeline can yield better results, ranked by your priorities, and (B) how to build a safe continuous feedback loop that learns from paired app-vs-traditional field records.

## How to read the confidence tags

Every load-bearing claim below carries a tag:

- **[VERIFIED]** — survived 3-vote adversarial fact-checking in this research effort, with a primary citation.
- **[SOURCED]** — drawn from an authoritative primary source that the research surfaced, but not put through the adversarial vote this run. Treat as well-established but confirm before betting the device on it.
- **[CODE]** — verified directly against this repository.
- **[INFERENCE]** — my engineering judgment connecting the evidence to your pipeline. Not a citation.

This honesty matters for a medical device: I am not going to dress up un-verified material as proven.

---

## Executive summary

The research **validates your pipeline's most important safety decisions** and concentrates the highest-value improvements in three places — two of which are *not machine-learning problems at all*.

1. **Your clinical core is right. Don't "improve" it into something worse.** The OR-rule (flag if MUAC < 115 mm **OR** WHZ < −3), the decision to estimate weight so you can compute WHZ, the manual-measurement-first priority, and surfacing ML probabilities only as a confidence signal — all are confirmed correct and WHO-consistent by verified multi-country evidence. **[VERIFIED]**

2. **The biggest accuracy wins are inputs, not models.** Two structural blind spots (nutritional **oedema** and clinical **triage**) cannot be fixed by any model and need new field-worker inputs. And the single most impactful ML-side fix is to **stop imputing the side-view depth features** — the production-grade comparator (Child Growth Monitor) uses a depth sensor precisely because that signal can't be faked from a frontal photo.

3. **Your feedback loop is ~80% built already, but auto-promotion on SAM-recall alone is unsafe.** The app already captures and syncs paired manual-vs-predicted records. The missing pieces are a provenance column, a held-out *real* evaluation set, a multi-gate promotion check (not a single recall number), and human sign-off before any model reaches a child.

**A myth this research killed:** "gradient-boosted trees always beat neural nets on tabular data" is **refuted** — neither family universally wins. So the move is not "replace the MLP with the orphaned LightGBM model"; it's "benchmark GBT vs MLP vs TabPFN on your own data." **[VERIFIED]**

---

# PART A — Pipeline improvements (ranked by your priorities)

Priority order, as you set it: **(1) SAM/MAM recall and the ≥0.80 SAM floor → (2) weight-estimate MAE → (3) app-vs-traditional agreement (κ).**

## A0. The two highest-impact fixes are not ML at all

### A0.1 Oedema (kwashiorkor) is a structural blind spot — Priority 1, recall

**[VERIFIED]** WHO defines SAM as *"nutritional oedema **and/or** WHZ/WLZ < −3 SD **and/or** MUAC < 115 mm"* — bilateral pitting oedema is an **independent, standalone** admission trigger, pathognomonic of SAM regardless of WHZ or MUAC.
*Sources: WHO 2023 guideline (NBK601655); WHO/UNICEF 2009 Joint Statement (NBK200776).*

**[VERIFIED]** *Nearly two-thirds of kwashiorkor cases have no acute wasting* (normal MUAC and WHZ). A WHZ/MUAC-only pipeline therefore misses **most** oedematous SAM — this is a large miss, not an edge case.

Oedema is detected by a physical pitting test (thumb pressure ~3 s on both feet), and **cannot** be inferred from monocular RGB or pose geometry. **[INFERENCE]**

> **Recommendation.** Add a **mandatory** oedema field to the assessment workflow — bilateral pitting (none / +/ ++ / +++) — and wire it into the OR-rule as a **third independent trigger**: `SAM if (oedema present) OR (MUAC < 11.5) OR (WHZ < −3)`. This is the single highest-recall improvement available, and it is a form field, not a model change. In `assessment_service.py` this slots directly into the existing `combine_with_whz_status` logic alongside MUAC and WHZ. **[INFERENCE on the code seam; [CODE] that the seam exists]**

### A0.2 The app screens; it cannot triage — Priority 1, safety framing

**[VERIFIED]** WHO 2023 routes a severely-wasted/oedematous child to **inpatient vs outpatient** care based on the **appetite test**, IMCI danger signs, acute medical problems, and severe oedema (+++) — none of which the app can capture. WHO explicitly separates SAM *identification* (anthropometry) from the *care-setting* decision.
*Source: WHO 2023 (NBK601655), Rec B2a/B2c.*

> **Recommendation.** Frame the app output as a **screening/status signal that triggers a structured clinical checklist** (appetite test, danger signs, oedema grade) — never a standalone disposition. This *reinforces* a design choice you already made (ML probabilities are surfaced as confidence, not diagnosis). Add the checklist as a post-screening step when status is SAM/MAM. **[INFERENCE]**

## A1. Stop imputing the side-view depth features — Priority 1→2, recall + weight MAE

**[CODE]** Features 10–13 (`chest_depth_cm`, `abd_depth_cm`, and their ratios) are **imputed from frontal widths** (`chest ≈ 0.45 × shoulder`, `abd ≈ 0.50 × hip`) when the side photo is missing — confirmed in `ml/models.py` `to_array()` / `ml/inference.py`.

Why this matters, from the best real-world evidence:

**[VERIFIED]** The leading deployed system, **Child Growth Monitor, does not use monocular RGB** — it uses a smartphone **Time-of-Flight (ToF) depth sensor + multi-pose RGB** feeding a CNN. *Even with real depth data and 87,131 training images*, only **~70.3%** of single captures fall within the SMART height tolerance (1.4 cm; 1.64% MAE) — i.e. ~30% miss.
*Sources: Trivedi/Jain et al., IEEE EMBC 2021 (arXiv:2105.01688); Lancet Reg. Health SE Asia 2026 Nepal validation.*

**[VERIFIED]** Independent 3D handheld scanners **failed the WHO technical-error-of-measurement threshold (<0.7 cm) by 4–8×** (inter-TEM 2.8 cm Guatemala, 3.4 cm Kenya, 5.5 cm China), producing stunting-prevalence errors of **8–50 percentage points**. Authors uniformly conclude image anthropometry is **not yet ready to replace manual measurement** at the per-child level.
*Sources: Bougma/Conkle et al., AJCN 2022 (PMC9576341); Conkle et al., PLOS ONE 2018 (PMC6200231).*

**[INFERENCE]** The depth signal you currently *fabricate from a fixed ratio* is exactly the signal real systems pay a hardware sensor to capture. Imputing it means the body-build / wasting cue from torso depth is, for those records, a deterministic function of width — contributing no independent information and potentially biasing WHZ through the weight estimate.

> **Recommendations, in order of effort:**
> 1. **Make the side photo required** (or strongly prompted) for any assessment that will rely on the ML weight estimate. When depth is imputed, **down-weight or abstain** from the ML weight rather than feeding it to WHZ (ties into A4 abstention and the existing 45–180% bounds check). **[INFERENCE]**
> 2. **Use ToF depth where the device exposes it** (ARCore Depth API / iOS LiDAR), matching the CGM approach. Treat `side_view_used` (already a column **[CODE]**) as a quality flag in evaluation and in the feedback loop.
> 3. **A scale reference** (ArUco / known-size card) is plausible but **the evidence that it helps was refuted** in verification — so pilot it, don't assume it. **[VERIFIED that the supporting claim failed]**

**Expected impact:** better depth → better weight estimate (Priority 2) → more accurate WHZ → fewer missed/false wasting flags (Priority 1).

## A2. The OR-rule and weight-target are correct — keep them — Priority 1

This is the strongest validation in the report; do not let any "simplification" undo it.

**[VERIFIED]** MUAC and WHZ identify **largely non-overlapping** children: only **~16.5%** of SAM cases meet both; **~45%** are WHZ-only, **~39%** MUAC-only. A single-indicator screen misses up to ~45–60% of the cases the other catches. WHO/UNICEF: use them as **independent criteria**.
*Sources: WHO/UNICEF 2009 (NBK200776); Grellety & Golden 2018 (PMC6138885); Schwinger/Golden/Grellety 2019 (PMC6684062).*

**[VERIFIED]** Estimating weight to compute WHZ (rather than going MUAC-only) is **clinically and arguably ethically required**: WHZ-only SAM children carry mortality risk comparable to MUAC-only (HR 3.69 vs 4.06). Verbatim: *"it would be unethical not to use WHZ whenever possible."*
*Source: Schwinger et al. 2019 (PMC6684062).*

**[VERIFIED]** Your exact thresholds — MUAC < 11.5 cm, WHZ < −3 — match the official WHO/UNICEF SAM definition for 6–59 months.

> **Recommendation.** Keep the OR-rule and the weight→WHZ target. The only addition is the **oedema trigger** (A0.1), which makes the OR-rule fully WHO-conformant. **[INFERENCE]**

**One nuance worth a footnote:** there is an active field debate (Briend et al.) that MUAC-alone suffices for logistical simplicity. The verified mortality evidence comes down against dropping WHZ — but you should be aware the debate exists when you talk to clinical partners. **[VERIFIED]**

## A3. Model architecture: benchmark, don't assume — Priority 1/2

You have an orphaned `train_gbt.py` (LightGBM trains, writes calibration, but **inference never loads it** **[CODE]**). The tempting move is "GBTs beat MLPs on tabular data, so switch." **The research refutes that as a blanket claim.**

**[VERIFIED]** Across 176 datasets / 19 algorithms, *neither neural nets nor GBDTs are universally superior; for many datasets the difference is negligible, and light GBT tuning matters more than the NN-vs-GBT choice.*
*Source: McElfresh et al., NeurIPS 2023 (arXiv:2305.02997).*

**[VERIFIED]** Independently confirmed: *"there is still no universally superior solution."*
*Source: Gorishniy et al., NeurIPS 2022 (arXiv:2106.11959); corroborated by arXiv:2407.04491.*

**[VERIFIED]** *Why* trees often do well, when they do: NNs (1) aren't robust to uninformative features, (2) are rotation-invariant (can't preserve the data's natural basis), (3) are biased toward smooth functions. Trees have the opposite biases by default.
*Source: Grinsztajn et al., NeurIPS 2022 (arXiv:2207.08815).*

**[VERIFIED]** **TabPFN v2** (a tabular foundation model) is the strongest evidence a transformer can *beat tuned GBTs* in the **small-data regime** — but its advantage is **bounded to ≤10,000 samples, ≤500 features, ≤10 classes**. Outside that, CatBoost/XGBoost likely win.
*Source: Hollmann et al., Nature 2025 (s41586-024-08328-6); Ye et al. (arXiv:2502.17361).*

> **Recommendation.** Run a **head-to-head benchmark on your own data**, gated on your real metric (SAM recall, then weight MAE):
> - **GBT (wire up the existing LightGBM)** — cheap; you already train it. The 14 hand-engineered features (widths/depths/ratios) are exactly the irregular-function / possibly-uninformative-feature setting where trees can shine. **[INFERENCE from Grinsztajn]**
> - **The current MLP / cascade** — incumbent.
> - **TabPFN v2** — but note **two cautions for your pipeline**: your 60k synthetic samples exceed the 10k cap (you'd subsample — *and subsampling must not starve the rare SAM class below the recall floor* **[INFERENCE]**), and you have a regression head (weight) plus a 5-class head plus the cascade.
>
> Decide by measured SAM recall + weight MAE, not by literature reputation. **[INFERENCE]**

## A4. Calibration, conformal prediction & an abstention path — Priority 1 (safety)

You already apply **temperature scaling + Mondrian (class-conditional) conformal calibration** **[CODE]** — this is ahead of most projects and is the right toolkit. The gap is an **explicit abstention / "refer to manual measurement" path**.

The following are **[SOURCED]** (authoritative sources surfaced by the research; not adversarially voted this run — confirm before relying):

- **Selective prediction / reject option**: a model may abstain on its least-confident inputs, trading **coverage** for **accuracy** on what it does answer — formalized via the **risk-coverage curve** (SelectiveNet; Geifman & El-Yaniv, ICML 2019). *Source: proceedings.mlr.press/v97/geifman19a.html.*
- **Conformal prediction for clinical ML**: produces prediction *sets* with a guaranteed coverage level (1−α). **Set size is itself an uncertainty signal** — a multi-label set ("could be SAM or MAM") is a natural "refer" trigger. *Source: JAMIA review, academic.oup.com/jamia/article/29/9/1525/6605096.*

> **Recommendation.** Add a third output state to the cascade: **SAM / MAM / Normal / "Indeterminate → manual measurement required."** Drive it from (a) conformal **set size > 1** at your chosen α, or (b) depth features were imputed (A1), or (c) the ML weight fell outside the **45–180% WHO bound** (you already compute this **[CODE]** — currently it silently falls back to WHO median; instead, *also* flag low confidence). Because your final status uses MUAC OR WHZ, abstaining on the *ML weight* simply routes to manual weight/MUAC — fail-safe, not fail-open. **[INFERENCE]**
>
> **Set the cascade thresholds against the risk-coverage curve**, not a fixed 0.5: pick the lowest threshold meeting SAM recall ≥ 0.80 (you already do a threshold search **[CODE]**), then let abstention absorb the ambiguous band instead of forcing a call. **[INFERENCE]**

## A5. Squeezing minority (SAM) recall — Priority 1

You currently use **class weights (SAM 1.5×) + threshold search** **[CODE]**. Beyond that, the candidate techniques are **[SOURCED]** / **[INFERENCE]** (this sub-area did not clear adversarial voting this run, so treat as standard practice to validate, not proven-for-your-data):

- **Threshold-moving is the most reliable, lowest-risk lever** for recall on a rare class, and you already do it. Decouple the *operating threshold* from training entirely and choose it on a validation set to hit the SAM-recall floor. **[INFERENCE — widely established]**
- **Focal loss** down-weights easy negatives to focus on the hard minority; a reasonable thing to A/B against class weights. **[SOURCED — Lin et al. focal-loss lineage]**
- **SMOTE / synthetic oversampling: be cautious.** On tabular medical data SMOTE can manufacture implausible points across the decision boundary and *hurt* calibration. You already control class balance directly in the synthetic generator (you oversample WHZ near −3/−2 **[CODE]**), which is cleaner than post-hoc SMOTE. **[INFERENCE]**

> **Recommendation.** Keep class-weights + threshold-moving as the backbone; A/B focal loss; **do not** add SMOTE on top of a generator you already control. Always report the precision paid for each recall gain (your `evaluate.py` threshold sweep already shows this trade-off **[CODE]**). **[INFERENCE]**

## A6. The synthetic-to-real gap is your deepest risk — Priority 1/2

Your model trains on **100% synthetic data** **[CODE]**. The verified evidence (from medical *imaging*, so magnitudes are analogical, not a tabular prior):

**[VERIFIED]** Pure-synthetic training carries a **real but modest in-distribution penalty** (e.g., 0.756 vs 0.772 AUROC on chest X-rays even at 565k synthetic images) that can **vanish or reverse out-of-distribution**.
*Source: Moroianu et al., Stanford 2025 (arXiv:2508.16783).*

**[VERIFIED]** **Mixing real + synthetic beats real-only** by a modest ceiling (~+0.02 AUROC), with the **largest gains on rare classes (<5% prevalence)** — favorable for a SAM-recall-critical model. *Source: Khosravi et al., EBioMedicine 2024.*

**[VERIFIED]** Mixing also **improves fairness** for underrepresented subgroups, especially OOD — directly relevant to your age/sex/ethnicity subgroups. *Source: Ktena et al., Nature Medicine 2024 (PMC11031395).*

**[VERIFIED]** The **real:synthetic ratio is a validation-tuned hyperparameter** — no universal value (observed optima ranged 50:50 to 100:0 across tasks). *Source: Ktena et al. 2024.*

**[VERIFIED]** Your **compartment-based wasting simulation is a legitimate instance of domain randomization** — the canonical sim-to-real technique; the goal is to randomize parameters widely enough that real children fall inside the training distribution. *Source: Tobin et al., IROS 2017 (arXiv:1703.06907).*

**Honesty note:** several appealing claims were **refuted** in verification and you should **not** rely on them: that synthetic-*pretrain-then-finetune* beats mixing (0-3); that *<10k* real samples suffice to match a full real baseline (0-3); that ~200% synthetic supplementation reaches parity (0-3). So I **cannot** give you a trustworthy "you need N real children" number — that remains an open question for *your* tabular setting.

> **Recommendation.** The field study you're already planning is the fix. As paired real records accumulate, **retrain on a real+synthetic blend with the mix ratio tuned on a held-out real validation set**, and watch subgroup recall. Do **not** deploy a pure-synthetic model as final once real data exists. This dovetails exactly with Part B. **[INFERENCE grounded in VERIFIED evidence]**

## A7. Is weight-then-WHZ the right ML target? — Priority 2

Three candidate targets: estimate **weight** (→ WHZ), classify **wasting directly**, or estimate **MUAC** from the image.

**[VERIFIED]** Because MUAC is only weakly correlated with WHZ (r ≈ 0.49–0.58) and catches a partly distinct at-risk population, **estimating MUAC from the image is a viable, non-redundant additional target** (shown feasible via regression on photos). It is **complementary** to weight/WHZ, not a replacement.
*Sources: WHO/UNICEF 2009 (NBK200776); PLOS One journal.pone.0195600.*

> **Recommendation.** Keep weight→WHZ as the primary target (A2). **Add an image-MUAC estimator as a secondary head** feeding the MUAC side of the OR-rule — but, exactly like the weight estimate, **gated and abstaining** (manual MUAC tape always wins). This adds an independent shot at the ~39% MUAC-only SAM cases when no tape reading is entered, without touching the validated WHZ path. **[INFERENCE]**

---

# PART B — A safe continuous "learn-from-mistakes" feedback loop

You want: collect paired (app, traditional) records → periodically retrain → gate on SAM recall → auto-promote if it passes. Below, what already exists, what to build, and where "auto-promote" must be tempered for a medical device.

**Verification honesty:** Part B's literature (continual learning, MLOps, PCCP, study statistics) was **fetched from authoritative sources but did not clear the adversarial vote this run** (the verifier budget was spent on Part A). So Part B is **[SOURCED]** + **[INFERENCE]**, anchored to primary FDA/biostatistics documents. It is solid engineering guidance; it is not adversarially-proven the way Part A's clinical findings are. I flag this so you weight it accordingly.

## B1. Data capture — you've already built ~80% of it

**[CODE]** The plumbing largely exists:
- DB (`app/models/measurement.py`) persists **both** `manual_weight_kg`/`manual_height_cm` **and** `predicted_*` and `ml_estimated_weight_kg`, plus all five class probabilities, HAZ/WHZ + statuses, and `muac_cm`/`muac_status`/`muac_method`.
- Flutter stores the same locally via **Drift/SQLite**, and an **offline-first `SyncQueue` → `POST /api/v1/sync`** already ships **both manual and predicted values** (and images) to the backend, with retry and stuck-entry recovery.

So the paired record you need for the comparison study is *already flowing*.

**Gaps to close [CODE-grounded gaps + [INFERENCE] on the fix]:**
1. **No explicit provenance column.** Which weight was used (`manual` / `ml_estimated` / `who_median`) is currently *inferred from nullness* in `assessment_service.py`. For a medical audit trail this is fragile. **Add `weight_source`, `height_source`, `muac_source` enum columns** and persist them. The service already computes `weight_source` as a local variable — just store it.
2. **No separate manual MUAC field.** `muac_method` distinguishes `manual` vs `estimated_from_whz`, but there isn't a clean paired `manual_muac_cm` vs `estimated_muac_cm`. Add it so MUAC can join the agreement study (B6) and the image-MUAC head (A7).
3. **Add an oedema field** (A0.1) to the schema, model, and sync payload.
4. **Provenance/audit metadata** the loop will need: app version, model version/hash, device model, whether ToF depth was available, `side_view_used` (exists **[CODE]**), field-worker ID (pseudonymized), timestamp, GPS region (consented). **[INFERENCE]**

## B2. Continual-learning pitfalls (and why "online" must really mean "periodic batch") — [SOURCED]/[INFERENCE]

- **Catastrophic forgetting** — naive fine-tuning on new field data erodes performance on the original distribution. Mitigate with **rehearsal/replay** (always retrain on a *frozen* representative base set + new data) and/or regularization (**EWC**). For your case, the simplest safe pattern is **full retrain on (frozen synthetic base ⊕ accumulating real records)** rather than incremental online updates. *Source surfaced: continual-learning survey arXiv:2209.03942; confident-learning arXiv:1911.00068 for label noise.*
- **Feedback-loop / bias amplification** — if the *app's own output* influences which children get measured or labeled, the training data drifts toward the model's blind spots. **Mitigation: the gold-standard manual measurement is the label, captured independently of the app's prediction** — your study design already breaks this loop, *provided the field worker measures manually without seeing the app's guess first.* Make that ordering a protocol rule. **[INFERENCE]**
- **Label noise from non-expert field workers** — manual MUAC/height have real measurement error (see TEM, B6). Mitigate with **plausibility checks, double-measurement on a subsample, and confident-learning–style filtering** of records whose label the model and physics strongly contradict. *Source: arXiv:1911.00068.* **Do not** auto-discard disagreements — they're the most valuable training signal (B4); flag for review instead.
- **Data drift** — monitor the *input feature distribution* (new regions, devices, seasons) and *label prevalence* over time; trigger retrain on drift, not just on a calendar. **[INFERENCE]**

> **The "model collapse" caveat:** the claim that iterative real+synthetic retraining *necessarily* causes collapse was **[VERIFIED as REFUTED]** this run — so I'm **not** telling you it will collapse. But the underlying risk (your generator's synthetic data + model-influenced real data compounding over cycles) is real enough to warrant the frozen-base-set discipline above. **[INFERENCE]**

## B3. Safe auto-retrain + auto-promote — gates, not a single number — [SOURCED]/[INFERENCE]

**Your stated plan (auto-promote on SAM recall) is necessary but not sufficient.** A single metric on a small eval set can pass by luck while the model regresses elsewhere. Use a **champion/challenger** pattern with a **multi-gate** promotion check, and—because this touches children—a **human in the loop before production**.

Pipeline shape (`retrain → evaluate → shadow → canary → promote`, with rollback at every step):

1. **Champion/challenger + shadow.** New model (challenger) runs in **shadow** on real traffic, predictions logged but **not shown**, compared against the champion. *Source: DataRobot champion/challenger; deployment-strategy literature.*
2. **Promotion gate set — ALL must pass (block on any failure):** **[INFERENCE, grounded in your `evaluate.py` [CODE] + GMLP [SOURCED]]**
   - **SAM recall ≥ 0.80** on a held-out **real** eval set (not synthetic) — your existing floor, but moved onto real data.
   - **No SAM-recall regression** vs champion beyond a non-inferiority margin (B6).
   - **Minimum eval-set size**, especially a minimum count of true SAM cases — a recall computed on 5 SAM children is noise. **[INFERENCE]**
   - **Calibration not worse** (ECE — you compute it **[CODE]**; add per-subgroup ECE).
   - **Per-subgroup checks across age × sex × region/ethnicity** — no subgroup's SAM recall falls below floor. Your `evaluate.py` does age *and* sex separately but **not crossed, and has no fairness gate** **[CODE]** — add this.
   - **Weight-MAE no regression** (Priority 2) — globally and per age band.
   - **Specificity floor** — so a "promote-by-crying-wolf" model that flags everyone (recall 1.0, useless) is rejected. `evaluate.py` lacks specificity today **[CODE]**.
3. **Canary** to a small fraction of sites, monitor, then full rollout. **Automatic rollback** if live metrics or error rates degrade.
4. **Human sign-off before production.** For a device where a false negative can be fatal, the safe reading of "auto-promote" is **auto-*qualify* a challenger and present it for one-click human approval** — not silent replacement. **[INFERENCE — strongly recommended]**

## B4. Active learning — disagreement is your highest-value signal — [SOURCED]/[INFERENCE]

You have a luxury most projects don't: a **gold-standard label on every record** (the manual measurement). So you don't need to *guess* which cases are informative — **the app-vs-traditional disagreements tell you directly.**

> **Recommendation.** Rank field records by **disagreement magnitude** (app status ≠ manual status; |app weight − scale weight|; |app MUAC − tape MUAC|) and **prioritize them for the next retraining batch and for human review.** Disagreement-based / uncertainty sampling is the textbook active-learning lever for sample efficiency. Combine with the conformal **set-size** signal (A4) to catch cases the model was *unsure* about even when it happened to be right. *Sources surfaced: active-learning literature in the continual-learning angle.* **[INFERENCE on the specific use of paired disagreement]**

This also feeds A6: disagreement cases are exactly the real data that closes the synthetic-to-real gap fastest.

## B5. Regulatory framing — what a compliant loop looks like — [SOURCED]

Even as a research-grade project, emulate the FDA structure; it *is* the blueprint for a safe self-updating medical model.

- **Predetermined Change Control Plan (PCCP).** FDA's framework for ML devices that change post-deployment: you **pre-specify** (a) exactly what modifications are allowed (e.g., "retrain weights on new field data; **no change to the 14-feature input or the OR-rule**"), (b) the **methodology** (the retrain→gate→canary protocol in B3), and (c) the **performance bounds** that must hold. Changes within the plan are pre-authorized; anything outside requires fresh review. *Sources: FDA PCCP guiding principles (fda.gov/.../predetermined-change-control-plans...); Foley analysis of the Jan-2025 final guidance.*
- **Good Machine Learning Practice (GMLP).** 10 guiding principles (FDA/Health Canada/MHRA) — notably **representative datasets, subgroup performance, human-AI team performance, and ongoing monitoring**. Your subgroup gates (B3) and oedema/triage human-in-the-loop (A0) map straight onto these. *Source: fda.gov/.../good-machine-learning-practice...*

> **Recommendation.** Write a one-page PCCP-style document now: *allowed changes = retrain on blended real+synthetic; frozen = inputs, OR-rule, thresholds-as-floors; gates = B3; rollback = automatic; human approval = required.* It will discipline the loop and make any future clinical partnership far easier. **[INFERENCE]**

## B6. The app-vs-traditional comparison study — do it right — [SOURCED]

This is the study your field workers' dual measurements enable. The statistics matter; the wrong tool gives falsely reassuring results.

**Continuous measures (height, weight, MUAC):**
- **Use Bland–Altman**, not correlation/regression. Correlation measures *association*, not *agreement* — two methods can correlate r≈0.99 yet disagree clinically. Report **mean bias** and **95% limits of agreement** (bias ± 1.96·SD of differences). *Source: Bland & Altman, Lancet 1986.*
- **Detect systematic bias:** a mean difference ≠ 0 is **fixed bias**; a trend in the difference-vs-mean plot is **proportional bias** (test by regressing differences on means). **[SOURCED]**
- **Anthropometry quality:** report **Technical Error of Measurement (TEM)** and compare to **WHO thresholds** — recall the verified benchmark that scanners *failed* the **<0.7 cm** height TEM by 4–8× **[VERIFIED, Part A]**; that 0.7 cm is your yardstick for the height/length comparison.

**Categorical status (SAM / MAM / Normal):**
- **Sensitivity, specificity, PPV, NPV** of the app vs the manual gold standard — with **SAM sensitivity** as the headline (it's your Priority 1 and the fatal-error direction). Report **confidence intervals**, following **STARD** reporting. **[SOURCED]**
- **Cohen's κ** for overall agreement; **weighted κ** because the categories are **ordered** (SAM→MAM→Normal) — a SAM-called-Normal error is far worse than MAM-called-Normal, and weighted κ encodes that. **[SOURCED]**

**Sample size:**
- Power the study for a **minimum acceptable SAM sensitivity** (e.g., lower 95% CI bound ≥ 0.80). Because SAM prevalence is low, **the binding constraint is the number of true SAM children**, not total N — you may need to **enrich** the sample (oversample known-malnourished sites) to get enough SAM cases for a tight sensitivity CI. **[SOURCED + INFERENCE]**

> **Recommendation.** Pre-register the analysis: Bland–Altman + TEM for the three continuous measures; sensitivity/specificity/PPV/NPV + weighted κ for status; sample size driven by SAM-case count for a sensitivity-CI target. This same pipeline doubles as the **promotion eval** in B3 — the study *is* the gate. **[INFERENCE]**

---

# Consolidated action list (by priority)

**Priority 1 — recall & safety (do first):**
1. **Add a mandatory oedema field** and make it a third OR-rule trigger. *(A0.1 — biggest recall win, not ML)*
2. **Frame output as screening + clinical-triage checklist**, never a disposition. *(A0.2)*
3. **Stop imputing depth; require the side photo / use ToF**, and abstain when depth is imputed. *(A1)*
4. **Add an abstention state** ("Indeterminate → manual measurement") driven by conformal set size + the existing 45–180% bound. *(A4)*
5. **Keep the OR-rule, thresholds, and weight→WHZ target.** *(A2 — validated; don't regress it)*

**Priority 2 — weight MAE & model quality:**
6. **Benchmark GBT vs MLP vs TabPFN** on your data, gated on SAM recall then MAE; don't assume GBT wins. *(A3)*
7. **Plan the real+synthetic blend** (ratio tuned on real validation data) as field data arrives. *(A6)*
8. **Add a gated image-MUAC head** feeding the MUAC arm of the OR-rule. *(A7)*

**Priority 1/3 — the feedback loop & study (build alongside):**
9. **Close the data gaps:** `weight_source`/`height_source`/`muac_source` provenance, `manual_muac_cm`, oedema, model-version/device metadata. *(B1)*
10. **Stand up champion/challenger + shadow/canary + multi-gate promotion** (real-data SAM recall, subgroup floors, calibration, specificity, weight MAE, min SAM-case count) with **human approval before production**. *(B3)*
11. **Prioritize app-vs-gold disagreements** for retraining and review. *(B4)*
12. **Write a PCCP-style change-control doc.** *(B5)*
13. **Run the comparison study correctly:** Bland–Altman + TEM (continuous), sensitivity/specificity/weighted-κ (categorical), SAM-case-driven sample size — and reuse it as the promotion gate. *(B6)*
14. **Extend `evaluate.py`:** add specificity, per-subgroup (age×sex×region) recall + calibration, and a fairness gate. *(B3 — [CODE] gaps today)*

---

## Evidence ledger (verification status)

**[VERIFIED] (adversarially fact-checked, cited):**
- OR-rule correctness; MUAC/WHZ non-overlap (~16.5%); WHZ-only mortality (HR 3.69 vs 4.06); thresholds match WHO. *(A2)*
- Oedema is an independent SAM trigger; ~2/3 of kwashiorkor has no wasting. *(A0.1)*
- WHO triage requires appetite test / danger signs the app can't capture. *(A0.2)*
- CGM uses ToF depth; ~30% of single captures miss tolerance; 3D scanners fail WHO TEM by 4–8×. *(A1)*
- "GBT always wins" is false — no universal winner; Grinsztajn inductive-bias reasons; TabPFN-v2 bounds. *(A3)*
- Synthetic-to-real: modest in-distribution penalty; mixing > real-only (esp. rare classes & fairness); mix ratio is a tuned hyperparameter; domain randomization framing. *(A6)*
- Image-MUAC is a viable complementary target. *(A7)*

**[SOURCED] (authoritative source surfaced, not voted this run):** selective prediction / conformal abstention (A4); focal loss / SMOTE cautions (A5); continual-learning & label-noise methods (B2); champion/challenger & deployment strategy (B3); active learning (B4); FDA PCCP & GMLP (B5); Bland–Altman, κ, TEM, STARD, sample size (B6).

**[CODE] (verified in repo):** imputed depth features; orphaned LightGBM; existing temperature+conformal calibration; 45–180% weight bound; threshold search; paired-data persistence + offline `SyncQueue`→`/api/v1/sync`; `evaluate.py` metric coverage and its gaps (no specificity, no crossed subgroups, no fairness gate).

**Refuted this run (do NOT rely on):** synthetic-pretrain-then-finetune > mixing; <10k real samples match full real baseline; ~200% synthetic = parity; iterative retraining ⇒ model collapse; trees remain SOTA after tuning on ~10k; ArUco/scale-reference reliably helps.

**Open (needs a dedicated future pass):** quantitative *tabular* synthetic-to-real gap & how many real children to fine-tune; adversarial verification of all Part B literature; TabPFN's effect on SAM coverage under subsampling.
