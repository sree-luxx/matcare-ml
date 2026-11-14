# MATCARE-AI — Test Results

**Framework:** Python standard library `unittest` (no third-party test dependency added)
**Command:** `python -m unittest discover -s tests`
**Result:** **25 tests, 25 passed, 0 failed** (~4.7 s)

The suite runs against the **saved artefacts** in `models/`, so it exercises the same
preprocessing, models and reporting code that the Streamlit app uses.

---

## 1. Test inventory

### `tests/test_inference.py` — 6 tests
| Test | Verifies |
| :--- | :--- |
| `test_probabilities_are_normalised` | Every probability vector sums to 1.0 for both modules |
| `test_predicted_class_is_known` | Predicted class is a member of the module class list |
| `test_full_input_needs_no_imputation` | Complete input reports no imputed features |
| `test_missing_input_is_imputed_and_still_valid` | Empty input is imputed and still yields valid probabilities |
| `test_hybrid_matches_weighted_base_models` | Hybrid equals `w_rf·P_rf + w_xgb·P_xgb` |
| `test_anemia_model_ignores_hemoglobin_and_sex` | Changing Hb/sex does not change the anaemia prediction |

### `tests/test_clinical.py` — 5 tests
| Test | Verifies |
| :--- | :--- |
| `test_evaluate_normal_and_out_of_range` | Reference-status logic (normal / warning / critical) |
| `test_unknown_feature_is_info` | Unknown features degrade gracefully to `info` |
| `test_anemia_severity_bands` | WHO haemoglobin severity bands + `None` handling |
| `test_morphology_bands` | MCV morphology categories |
| `test_build_vitals_rows_skips_unbounded` | Scorecard skips features without reference intervals |

### `tests/test_model_loading.py` — 6 tests
| Test | Verifies |
| :--- | :--- |
| `test_both_modules_present` | Bundle exposes `maternal` and `anemia` |
| `test_each_module_exposes_required_keys` | All expected bundle keys present |
| `test_weights_are_a_convex_combination` | `w_rf + w_xgb = 1` |
| `test_config_feature_names_match_expected_inputs` | Saved feature lists match code |
| `test_anemia_excludes_hemoglobin_and_sex` | Leakage exclusion recorded in config |
| `test_artefacts_exist_on_disk` | All model/config files present |

### `tests/test_reporting.py` — 4 tests
| Test | Verifies |
| :--- | :--- |
| `test_required_sections_present` | All 7 report sections rendered |
| `test_disclaimer_and_noncausal_note` | Disclaimer + "not causation" + exclusion note present |
| `test_patient_details_are_rendered` | Patient name and ID appear in the report |
| `test_pdf_is_produced` | PDF renders and starts with the `%PDF` header |

### `tests/test_shap.py` — 4 tests
| Test | Verifies |
| :--- | :--- |
| `test_local_frame_has_expected_columns_and_rows` | Per-patient SHAP frame shape for both modules |
| `test_direction_wording_matches_sign` | "Raises/Lowers" wording matches sign of contribution |
| `test_anemia_frame_only_contains_index_features` | Anaemia explanations only mention MCV/MCH/MCHC |
| `test_hybrid_importances_are_normalised` | Hybrid global importances sum to 1.0 |

---

## 2. Bug found and fixed by the suite

`tests/test_reporting.test_pdf_is_produced` failed on first run with
`AttributeError: module 'matplotlib.backends' has no attribute 'backend_pdf'`. The PDF
builder referenced `matplotlib.backends.backend_pdf.PdfPages` without importing the
submodule. Fixed in `app/reporting.py` by importing `PdfPages` explicitly. The suite then
passed in full.

---

## 3. Notes

- Tests use the real saved models; they are a fast integration check, not a replacement for
  the full evaluation in `reports/evaluation_report.md`.
- The fixed seed (42) and stored `preprocessor.pkl` make the results reproducible.
- `sklearn` emits a benign `X has feature names, but RandomForestClassifier was fitted
  without feature names` warning during prediction; it does not affect the outputs.
