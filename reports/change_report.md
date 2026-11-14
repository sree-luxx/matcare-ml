# MATCARE-AI — Change Report

This report lists what was changed, added or preserved during the phased improvement of the
project. It is a companion to the evaluation report and the leakage audit report.

---

## 1. Principle followed

The project was **audited first and improved in place**. Working functionality was preserved;
the original artefacts were backed up before any retraining so the change is fully reversible.

Backups:

- `models/general_risk/original_protocol/`
- `models/anemia/original_protocol/`
- `reports/original_protocol/`

No dataset file was modified: `datasets/*.csv` are untouched.

---

## 2. Added — modular application (`app/`)

| File | Purpose |
| :--- | :--- |
| `app/theme.py` | Design system: palette, status colours, CSS injection, `card`, `pill`, `kpi_card` |
| `app/clinical.py` | Reference intervals, `evaluate`, `interpret`, `anemia_severity`, `morphology`, `risk_plain_language`, `build_vitals_rows` |
| `app/model_loader.py` | Cached model/metadata loading (`st.cache_resource`) |
| `app/inference.py` | Both-module inference through the saved preprocessor; probability normalisation; model agreement |
| `app/components.py` | Reusable Plotly/HTML components (gauges, bars, SHAP chart, confusion matrix, ROC) |
| `app/reporting.py` | Markdown + PDF patient report (`build_report_markdown`, `build_report_pdf`) |
| `app/research.py` | Held-out evaluation reconstruction, ROC/PR curves, ablations, CV, bootstrap for the dashboard |
| `app/main.py` | Five-tab orchestrator + sidebar patient form + presets |
| `app/sections/assessment.py` | Tab 1 — Clinical Assessment |
| `app/sections/explainability.py` | Tab 2 — Explainable AI |
| `app/sections/model_intelligence.py` | Tab 3 — Model Intelligence |
| `app/sections/report.py` | Tab 4 — Medical Report |
| `app/sections/datasets.py` | Tab 5 — Dataset & Research |

## 3. Added — training / evaluation / explainability packages

| File | Purpose |
| :--- | :--- |
| `training/model_search.py` | Randomized hyper-parameter search with preprocessing inside each fold; `fit_preprocessor`, `fit_final_models`, `repeated_split_scores` |
| `training/hybrid_model.py` | `AdaptiveWeightedHybridClassifier` with validation-only weight search, stability, save/load |
| `training/train_general_risk.py` | Maternal pipeline (dedup + conflict removal, 70/15/15, weights, metrics, SHAP, save) |
| `training/train_anemia.py` | Anaemia pipeline (dedup, index-only features, ablation, weights, metrics, save) |
| `evaluation/audit.py` | `audit_*` and `prepare_*_modeling_frame` audit/preparation helpers |
| `evaluation/metrics.py` | Comprehensive metrics (accuracy, macro/weighted F1, ROC-AUC, PR-AUC, FPR/FNR, calibration) |
| `explainability/shap_explainers.py` | Global tree-SHAP importances + per-patient attributions |
| `explainability/labels.py` | Clinical display labels for features |
| `preprocessing/*.py` | Load/split helpers, feature/class constants, label-rule documentation |

## 4. Added — tests and reports

| File | Purpose |
| :--- | :--- |
| `tests/test_inference.py`, `tests/test_clinical.py`, `tests/test_model_loading.py`, `tests/test_reporting.py`, `tests/test_shap.py` | 25 `unittest` tests (all passing) |
| `reports/evaluation_report.md` | Held-out evaluation for both modules |
| `reports/data_leakage_audit_report.md` | Leakage audit and ablation |
| `reports/clinical_validation_notes.md` | Rewritten clinical limitations / compliance notes |
| `reports/test_results.md` | Test inventory and results |
| `reports/change_report.md` | This document |
| `README.md` | Run instructions |

## 5. Changed — models retrained under the audited protocol

Both modules were retrained and re-saved (original artefacts preserved in the backups).

| Module | Old protocol | New protocol |
| :--- | :--- | :--- |
| Maternal risk | raw 70/15/15, N=153 test | deduplicated, conflict-free, N=58 test, preprocessor saved, validation-only weights |
| Anaemia | full features (incl. Hb + sex) → 100% | index-only (MCV/MCH/MCHC), deduplicated, N=81 test, ablation recorded |

New artefacts written per module: `rf_model.pkl`, `xgb_model.pkl`, `hybrid_model.pkl`,
`scaler.pkl`, `preprocessor.pkl`, `weights.json`, `training_config.json`,
`evaluation_metrics.json` (+ `population_audit.json` for anaemia).

## 6. Fixed

- **Anaemia label circularity** — Hb and sex removed from the model; the 100% result replaced
  by an honest ~46% at chance, documented in the leakage audit.
- **Maternal duplicates / conflicting labels** — removed before splitting.
- **`HeartRate < 30`** impossible values — treated as missing, imputed from train medians.
- **PDF exporter bug** — `matplotlib.backends.backend_pdf` was referenced without importing
  the submodule; fixed (found by the test suite).
- **Hybrid probability normalisation** — hybrid vectors are renormalised to a valid
  distribution at inference.

## 7. Behaviour and claims

- The app now shows per-module probabilities, class-wise metrics, confusion matrices,
  ROC/PR curves, calibration, repeated CV and bootstrap intervals.
- The hybrid is reported as **equivalent to the stronger base model**, never as superior.
- Every surface (UI, report, docs) distinguishes model output from diagnosis and states that
  results are not clinically validated.
