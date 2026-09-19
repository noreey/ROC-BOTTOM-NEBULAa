# Train Condition Monitoring — Nebula X Hackathon 2026, Problem Statement 3

A single operator-facing app that turns raw train condition-monitoring data into maintenance
decisions, covering three of the four PS3 subsystems: **Structural Health Monitoring**,
**Air Conditioning & Ventilation**, and **Rail Corrugation**.

The app is also what generated the submitted `predictions.zip` — the held-out test files were run
through this same interface, not a separate script.

---

## Results

Every figure below is **out-of-fold**, measured on the metric each subsystem is actually scored
on. None of them is training-set performance.

| Subsystem | Score | Metric | Validation |
|---|---|---|---|
| SHM | **0.975** | `max(0, 1 − MAPE)`, MAPE 2.54% | Leave-one-out over 64 files |
| ACV | **0.979** | Mean rank-decay | All 6 labelled cases (faulty car ranked 1st in 5 of 6) |
| Rail Corrugation | **0.850** | Macro F1 | Leave-one-out over 272 files |
| Door | not attempted | — | — |

**Why leave-one-out rather than a single train/test split.** The datasets are small and
imbalanced — 6 labelled ACV cases, 14 Side I rail examples out of 272 files. With 14 examples a
5-fold validation fold holds only 2–3 of them, so one flip moves the score by 20–30% and a single
split cannot distinguish a better model from a luckier one. LOOCV removes fold-composition noise
entirely, and it is what we used to make every model choice in this repo.

**A note on the Rail metric.** Predicting "Normal" for every file scores ~86% plain accuracy on
this dataset but **0.33 macro F1** — the metric is specifically designed to expose that shortcut,
and 0.850 is measured against it.

---

## Running the app

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

Select a subsystem, upload the held-out test files for it, press **Analyse**. Two downloads are
offered: the competition-format CSV, and a plain-language maintenance report for a depot.

Input files per subsystem:

| Subsystem | Expects | Shape |
|---|---|---|
| SHM | `test01.csv` … `test16.csv` | ~6 MB, one stress reading per line, no header |
| ACV | `acv_test_case.xlsx` | per-car columns, `Car 01 - <parameter>` |
| Rail | `Test1.csv` … `Test68.csv` | ~17 MB, 10,000 rows × 129 columns |

Rail files are large; upload 10–15 at a time rather than all 68.

---

## Method, per subsystem

### SHM — physics, not a fitted black box

Rainflow-counts each stress time series (ASTM E1049), then applies Miner's linear damage rule:

```
D = Σ nᵢ/Nᵢ ,  Nᵢ = C / σᵢᵐ   ⟹   D = S(m)/C ,  S(m) = Σ nᵢ · σᵢᵐ
```

Only two constants are fitted to the 64 training files — the S-N exponent `m` and the constant
`C` — so the model has almost no capacity to overfit. `m = 5.0` was selected by cross-validation
and sits in a sharp, well-defined minimum; `C` is fitted by minimising MAPE directly, because
MAPE is the scoring metric.

The operator layer expresses `D` two ways a maintainer can act on: what fraction of the fatigue
budget the segment consumed (Miner's rule defines failure at `D = 1.0`), and where it sits against
the distribution of the 64 healthy training segments.

### ACV — cross-car comparison, no training at all

With six labelled cases, fitting a model would overfit badly. Instead each car is compared
against its seven siblings **at the same timestamp**, which controls for route, weather and
schedule automatically. Cars are ranked by mean absolute deviation from their siblings across
every shared numeric parameter.

Two details carry most of the accuracy:

- **Ambient outdoor temperature is excluded.** It is weather — shared by the whole train — so it
  carries no cross-car diagnostic signal and only dilutes the parameters that do. Removing it
  raised the score from 0.938 to 0.979.
- **Cars with no usable telemetry are screened out.** One training file declares column headers
  for all eight cars but logs data for only four. Without the screen, the method produced a
  plausible-looking answer that was correct by coincidence.

Rows where a car reported its data invalid, or was in load-shedding, are masked out.

### Rail Corrugation — wavelength domain, then gradient boosting

Corrugation appears as vibration energy at a frequency that scales with train speed
(`frequency = speed / wavelength`), so the same physical defect lands in different frequency bins
depending on how fast the train was going. Each axle-box signal is therefore FFT'd and converted
from frequency to **wavelength**, using the train's own speed — which is not given, and is
reconstructed by counting rising edges on the raw 90-tooth wheel pulse sensor.

Energy is binned into wavelength bands matching the corrugation scales in the Info Kit, per rail
side (Side I = axle-box positions 1,3,5,7; Side II = 2,4,6,8). An XGBoost classifier with balanced
class weighting runs over those ~49 features.

---

## Repository layout

```
app/                  the deliverable — single Streamlit app, all subsystems
  app.py              UI, routing, charts, CSV export
  predictors/         one module per subsystem, independent of each other
modelling/            how each model was built and validated
  shm/ acv/ rail/
  experiments/        hypotheses that were tested and REJECTED — see below
predictions/          the submitted CSVs
```

`modelling/experiments/` is kept deliberately. Several plausible improvements were tested and
did not survive validation, and the scripts that rejected them are as much a part of the method as
the ones that worked:

| Experiment | Result |
|---|---|
| SHM mean-stress (Goodman-style) correction | Rejected — worse at every correction strength |
| SHM endurance limit / bilinear S-N curve | Real but worth +0.0006; not shipped |
| ACV leave-one-out z-scoring | No change in ranking |
| ACV robust median/MAD statistics | Rejected — 0.958 vs 0.979 |
| Rail per-side binary decomposition | Rejected — ~0.80 vs 0.850 |
| Rail mirror augmentation (exploiting side symmetry) | Rejected — 0.8455 vs 0.8501 |
| Rail side-contrast features | Rejected — 0.8436 vs 0.8501 |

The shipped Rail model came from one change that *did* survive: replacing sklearn's
`GradientBoostingClassifier` with XGBoost on identical features, which moved LOOCV macro F1 from
0.800 to 0.850.

---

## Extending it — adding the Door subsystem

`app/predictors/door.py` is a documented stub. The app discovers predictors dynamically, so
implementing it requires no changes to `app.py`:

```python
SUBSYSTEM_NAME, OUTPUT_FILENAME, OUTPUT_COLUMNS, ACCEPTED_FILE_TYPES
predict(uploaded_files) -> DataFrame          # required
interpret(row) -> dict                        # optional: adds the operator verdict
batch_chart(df) / detail_chart(row)           # optional: adds charts
```

Any extra columns a predictor returns are used for the operator view and stripped before the
competition CSV is written, so the submission format cannot drift.

---

## Honest limitations

- **Door is not implemented.** The PS3 "Overall Score" divides by four regardless, so this costs
  a quarter of that score.
- **ACV rests on six labelled cases.** 0.979 is a real out-of-fold figure, but six examples cannot
  establish that the method generalises — only that nothing in the labelled data contradicts it.
- **Rail's Side I class has 14 training examples.** A single file changes macro F1 by ~0.02, so
  differences smaller than that are noise, not improvement.
- **SHM assumes the training set is representative.** All 64 training files are healthy operating
  conditions, so the baseline percentiles the operator view reports against describe normal
  loading, not damage.
