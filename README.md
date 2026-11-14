# MATCARE-AI

Explainable adaptive-hybrid screening prototype with two modules:

1. **General maternal health risk** — Low / Mid / High risk from routine vital signs.
2. **General haematological anaemia classification** — Non-Anemic / Anemic from red-cell indices.

Both modules combine a Random Forest and an XGBoost model with a validation-weighted
probability mixture, and explain predictions with SHAP. See
`reports/evaluation_report.md`, `reports/data_leakage_audit_report.md` and
`reports/clinical_validation_notes.md` before using any output.

> This is a research screening prototype. It is **not** a diagnosis, it gives **no** treatment
> advice, and it has **not** been clinically validated.

---

## Requirements

- Python 3.11+
- Install dependencies:

```
pip install -r requirements.txt
```

## Run the dashboard

```
streamlit run app/main.py
```

Then open the URL Streamlit prints (usually http://localhost:8501). A permanent left sidebar is
the navigation; the pages are grouped into **Patient care** and **Model & research**.

Patient care flow: **Patient Assessment** (structured form — patient profile, the six maternal
vital signs and the CBC block) → **Results** (both modules run on submit) → **Save assessment**
(opens the patient's overview) → **Patient Profiles** (one card per patient, with search) →
**Patient Overview** (concise per-patient summary) → **View full dashboard →** →
**Patient Dashboard** (probability charts, blood-pressure/Haemoglobin/MCV trends, assessment
history, previous-vs-latest comparison, SHAP explanations and the Random Forest / XGBoost /
hybrid model comparison).

Model & research pages: Explainable AI, Model Intelligence, Medical Report and Dataset &
Research.

Every saved assessment is stored in SQLite under its own patient ID with a timestamp; earlier
assessments are never overwritten and every chart is drawn from the stored records.

The app loads the saved models from `models/`. If they are missing, the app prints the exact
training commands to run.

> Note: the app shows a PDF importer uses `matplotlib`'s PDF backend, which ships with
> `matplotlib` — no extra system dependency is required.

## Retrain the models (optional)

Training overwrites the artefacts in `models/<module>/`. The previous artefacts are preserved
under `models/<module>/original_protocol/`.

```
python -m training.train_general_risk
python -m training.train_anemia
```

Both pipelines are seeded (`random_state = 42`) and split the data before fitting any
preprocessing, so runs are reproducible.

## Run the tests

```
python -m unittest discover -s tests
```

67 tests cover inference, probability normalisation, invalid/missing inputs, model loading,
SHAP output, storage persistence, patient identity handling, clinical anaemia screening,
record deletion and the end-to-end multi-page navigation flow.

## Project layout

```
app/            Streamlit dashboard (page router, theme, clinical logic, inference, components,
                sections: assessment form, results, patient profiles/overview/dashboard, research)
training/       Pipelines, hyper-parameter search, adaptive hybrid
preprocessing/  Load/split helpers and feature constants
evaluation/     Data audit + comprehensive metrics
explainability/ SHAP importances and per-patient explanations
datasets/       Raw CSV files (never modified)
models/         Saved artefacts per module (+ original_protocol backups)
reports/        Evaluation, leakage audit, clinical notes, test results, change report
tests/          unittest suite
scripts/        Standalone audit scripts (e.g. phase2_leakage_audit.py)
```

## Key design points

- Preprocessing (median imputation + standard scaling) is fitted on the training split only and
  saved as `preprocessor.pkl`; inference reuses it verbatim, so the app cannot drift from training.
- Exact duplicate rows and conflicting-label rows are removed before splitting; the held-out
  test partition is created once and never used for tuning.
- The anaemia model uses MCV, MCH and MCHC only. Haemoglobin and the sex code are excluded
  because the dataset label is an exact sex-specific haemoglobin threshold.
