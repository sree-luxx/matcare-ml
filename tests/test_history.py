"""History tests: duplicate detection and trend data transformations.

Covers the assessment-history requirements: duplicates are suppressed only for
accidental repeated submissions, chart series use real datetime timestamps
(oldest first), missing measurements are never interpolated, and indicators with
different units are never placed on a shared axis.
"""

import datetime
import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import storage
from app.sections.patient_dashboard import build_trend_series


def _maternal(high=0.8):
    return {
        "predicted_class": "High Risk",
        "confidence": high * 100.0,
        "probabilities": {
            "Random Forest": [0.1, 0.1, high],
            "XGBoost": [0.1, 0.1, high],
            "Adaptive hybrid": [0.1, 0.1, high],
        },
        "w_rf": 0.9,
        "w_xgb": 0.1,
    }


def _anemia(anemic=0.6):
    return {
        "predicted_class": "Anemic" if anemic >= 0.5 else "Non-Anemic",
        "confidence": anemic * 100.0,
        "probabilities": {
            "Random Forest": [1 - anemic, anemic],
            "XGBoost": [1 - anemic, anemic],
            "Adaptive hybrid": [1 - anemic, anemic],
        },
        "w_rf": 0.6,
        "w_xgb": 0.4,
    }


MATERNAL = {"Age": 30, "SystolicBP": 130, "DiastolicBP": 85,
            "BS": 8.0, "BodyTemp": 98.6, "HeartRate": 80}
CBC = {"Gender": 0, "Hemoglobin": 11.0, "RBC": 4.2, "MCV": 84.0, "MCH": 27.0, "MCHC": 32.0}


class _TempDB(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._orig_dir, self._orig_path = storage.DATA_DIR, storage.DB_PATH
        storage.DATA_DIR = self._dir.name
        storage.DB_PATH = os.path.join(self._dir.name, "history.db")
        storage.init_db()

    def tearDown(self):
        storage.DATA_DIR, storage.DB_PATH = self._orig_dir, self._orig_path
        self._dir.cleanup()

    def _save(self, patient_id="PT-1", maternal=None, cbc=None, created_at=None):
        return storage.save_assessment(
            patient_id=patient_id, patient_name="Test Patient",
            maternal_inputs=maternal or MATERNAL, cbc_inputs=cbc or CBC,
            maternal_result=_maternal(), anemia_result=_anemia(), created_at=created_at,
        )


class DuplicateTests(_TempDB):
    def test_no_records_returns_none(self):
        self.assertIsNone(storage.find_duplicate("PT-1", MATERNAL, CBC))

    def test_immediate_identical_submission_is_a_duplicate(self):
        record_id = self._save()
        self.assertEqual(storage.find_duplicate("PT-1", MATERNAL, CBC), record_id)

    def test_changed_values_are_not_duplicates(self):
        self._save()
        changed = {**MATERNAL, "SystolicBP": 150}
        self.assertIsNone(storage.find_duplicate("PT-1", changed, CBC))

    def test_old_identical_submission_is_not_a_duplicate(self):
        past = (datetime.datetime.now() - datetime.timedelta(hours=2)).isoformat(timespec="seconds")
        self._save(created_at=past)
        self.assertIsNone(storage.find_duplicate("PT-1", MATERNAL, CBC, within_seconds=30))

    def test_only_the_most_recent_record_is_checked(self):
        self._save(created_at="2026-01-01T09:00:00")
        # A different assessment is saved afterwards, so the earlier identical one
        # is no longer the "just submitted" record.
        self._save(maternal={**MATERNAL, "SystolicBP": 150}, created_at="2026-06-01T09:00:00")
        self.assertIsNone(storage.find_duplicate("PT-1", MATERNAL, CBC, within_seconds=10 ** 9))

    def test_other_patients_do_not_collide(self):
        self._save(patient_id="PT-1")
        self.assertIsNone(storage.find_duplicate("PT-2", MATERNAL, CBC))


class TrendSeriesTests(_TempDB):
    def _history(self, patient_id="PT-T"):
        return storage.get_history(patient_id)

    def test_single_assessment_yields_no_trend(self):
        self._save(patient_id="PT-T", created_at="2026-01-01T09:00:00")
        dates, charts = build_trend_series(self._history())
        self.assertEqual(dates, [])
        self.assertEqual(charts, [])

    def test_two_assessments_use_datetime_x_axis(self):
        self._save(patient_id="PT-T", created_at="2026-01-01T09:00:00")
        self._save(patient_id="PT-T", maternal={**MATERNAL, "SystolicBP": 140},
                   created_at="2026-02-01T09:00:00")
        dates, charts = build_trend_series(self._history())
        self.assertEqual(len(dates), 2)
        self.assertTrue(all(isinstance(d, datetime.datetime) for d in dates))
        # Oldest first, matching the stored timestamps.
        self.assertLess(dates[0], dates[1])

        blood_pressure = next((series for group, series in charts if group["title"] == "Blood pressure"), None)
        self.assertIsNotNone(blood_pressure)
        self.assertEqual([s["name"] for s in blood_pressure], ["Systolic BP", "Diastolic BP"])
        systolic = next(s for s in blood_pressure if s["name"] == "Systolic BP")
        self.assertEqual(systolic["values"], [130.0, 140.0])

    def test_different_units_are_never_shared_on_one_chart(self):
        self._save(patient_id="PT-T", created_at="2026-01-01T09:00:00")
        self._save(patient_id="PT-T", created_at="2026-02-01T09:00:00")
        _, charts = build_trend_series(self._history())
        for group, series in charts:
            self.assertEqual({s["unit"] for s in series}, {group["unit"]},
                             msg=f"{group['title']} mixes units")
        titles = {group["title"] for group, _ in charts}
        self.assertIn("Haemoglobin", titles)
        self.assertIn("Mean corpuscular volume", titles)

    def test_missing_measurement_is_not_interpolated(self):
        # Second assessment has no haemoglobin recorded.
        self._save(patient_id="PT-T", created_at="2026-01-01T09:00:00")
        self._save(patient_id="PT-T", cbc={k: v for k, v in CBC.items() if k != "Hemoglobin"},
                   created_at="2026-02-01T09:00:00")
        _, charts = build_trend_series(self._history())
        titles = {group["title"] for group, _ in charts}
        # Haemoglobin has only one value, so its chart is dropped rather than bridged.
        self.assertNotIn("Haemoglobin", titles)
        self.assertIn("Mean corpuscular volume", titles)

    def test_identical_values_are_kept_as_real_points(self):
        self._save(patient_id="PT-T", created_at="2026-01-01T09:00:00")
        self._save(patient_id="PT-T", created_at="2026-02-01T09:00:00")
        _, charts = build_trend_series(self._history())
        blood_pressure = next(series for group, series in charts if group["title"] == "Blood pressure")
        systolic = next(s for s in blood_pressure if s["name"] == "Systolic BP")
        self.assertEqual(systolic["values"], [130.0, 130.0])
        self.assertTrue(all(v is not None for v in systolic["values"]))

    def test_probability_trends_use_stored_probabilities(self):
        self._save(patient_id="PT-T", created_at="2026-01-01T09:00:00")
        self._save(patient_id="PT-T", created_at="2026-02-01T09:00:00")
        _, charts = build_trend_series(self._history())
        titles = {group["title"] for group, _ in charts}
        self.assertIn("Maternal high-risk probability", titles)
        self.assertIn("Anaemia probability", titles)
        risk = next(series for group, series in charts if group["title"] == "Maternal high-risk probability")
        self.assertEqual(risk[0]["unit"], "%")
        self.assertEqual(risk[0]["values"], [80.0, 80.0])

    def test_history_is_scoped_to_the_selected_patient(self):
        self._save(patient_id="PT-A", created_at="2026-01-01T09:00:00")
        self._save(patient_id="PT-A", created_at="2026-02-01T09:00:00")
        self._save(patient_id="PT-B", created_at="2026-03-01T09:00:00")
        dates, _ = build_trend_series(self._history("PT-A"))
        self.assertEqual(len(dates), 2)
        # PT-B's single assessment cannot form a trend of its own.
        dates_b, charts_b = build_trend_series(self._history("PT-B"))
        self.assertEqual((dates_b, charts_b), ([], []))


if __name__ == "__main__":
    unittest.main()
