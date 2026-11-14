"""Storage tests: save, retrieve, search, history, update, delete, persistence.

The tests point the storage module at a throw-away SQLite file in a temporary
directory so the real ``data/matcare.db`` is never touched.
"""

import os
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import storage


def _maternal_result(class_name="High Risk", confidence=0.91):
    return {
        "predicted_class": class_name,
        "confidence": confidence,
        "probabilities": {
            "Random Forest": [0.05, 0.10, 0.85],
            "XGBoost": [0.06, 0.09, 0.85],
            "Adaptive hybrid": [0.05, 0.09, 0.86],
        },
        "w_rf": 0.96,
        "w_xgb": 0.04,
    }


def _anemia_result(class_name="Anemic", confidence=0.77):
    return {
        "predicted_class": class_name,
        "confidence": confidence,
        "probabilities": {
            "Random Forest": [0.23, 0.77],
            "XGBoost": [0.25, 0.75],
            "Adaptive hybrid": [0.24, 0.76],
        },
        "w_rf": 0.61,
        "w_xgb": 0.39,
    }


class StorageTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._orig_dir = storage.DATA_DIR
        self._orig_path = storage.DB_PATH
        storage.DATA_DIR = self._dir.name
        storage.DB_PATH = os.path.join(self._dir.name, "test.db")
        storage.init_db()

    def tearDown(self):
        storage.DATA_DIR = self._orig_dir
        storage.DB_PATH = self._orig_path
        self._dir.cleanup()

    def _save(self, patient_id="PT-101", name="Ananya Sharma", created_at=None, notes=""):
        return storage.save_assessment(
            patient_id=patient_id,
            patient_name=name,
            maternal_inputs={"Age": 36, "SystolicBP": 155, "DiastolicBP": 98,
                             "BS": 15.2, "BodyTemp": 99.4, "HeartRate": 92},
            cbc_inputs={"Gender": 0, "Hemoglobin": 8.2, "RBC": 3.4,
                        "MCV": 72.0, "MCH": 21.0, "MCHC": 28.5},
            maternal_result=_maternal_result(),
            anemia_result=_anemia_result(),
            notes=notes,
            created_at=created_at,
        )

    def test_save_returns_incrementing_ids(self):
        first = self._save()
        second = self._save(patient_id="PT-102", name="Priya Nair")
        self.assertEqual(second, first + 1)

    def test_round_trip_preserves_payload(self):
        record_id = self._save()
        record = storage.get_record(record_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["patient_id"], "PT-101")
        self.assertEqual(record["patient_name"], "Ananya Sharma")
        self.assertEqual(record["maternal_prediction"], "High Risk")
        self.assertEqual(record["anemia_prediction"], "Anemic")
        self.assertAlmostEqual(record["maternal_confidence"], 0.91, places=3)
        # JSON columns come back decoded.
        self.assertEqual(record["maternal_inputs"]["SystolicBP"], 155)
        self.assertEqual(record["cbc_inputs"]["MCV"], 72.0)
        self.assertEqual(len(record["maternal_probabilities"]["Adaptive hybrid"]), 3)
        self.assertEqual(len(record["anemia_probabilities"]["Adaptive hybrid"]), 2)
        self.assertAlmostEqual(record["maternal_weights"]["w_rf"], 0.96, places=3)

    def test_get_missing_record_returns_none(self):
        self.assertIsNone(storage.get_record(9999))

    def test_list_is_newest_first(self):
        self._save(patient_id="PT-1", created_at="2026-01-01T10:00:00")
        self._save(patient_id="PT-2", created_at="2026-02-01T10:00:00")
        rows = storage.list_records()
        self.assertEqual([r["patient_id"] for r in rows], ["PT-2", "PT-1"])

    def test_search_matches_name_and_id(self):
        self._save(patient_id="PT-101", name="Ananya Sharma")
        self._save(patient_id="PT-102", name="Priya Nair")
        by_name = storage.list_records(query="Priya")
        self.assertEqual(len(by_name), 1)
        self.assertEqual(by_name[0]["patient_id"], "PT-102")
        by_id = storage.list_records(query="PT-10")
        self.assertEqual(len(by_id), 2)

    def test_history_is_oldest_first_and_scoped(self):
        self._save(patient_id="PT-101", created_at="2026-03-01T09:00:00")
        self._save(patient_id="PT-101", created_at="2026-03-05T09:00:00")
        self._save(patient_id="PT-200", created_at="2026-03-10T09:00:00")
        history = storage.get_history("PT-101")
        self.assertEqual(len(history), 2)
        self.assertTrue(history[0]["created_at"] < history[1]["created_at"])

    def test_new_assessment_does_not_overwrite_previous(self):
        self._save(patient_id="PT-101", created_at="2026-03-01T09:00:00")
        self._save(patient_id="PT-101", created_at="2026-03-05T09:00:00")
        self.assertEqual(storage.count_records(), 2)

    def test_update_name_and_notes(self):
        record_id = self._save()
        storage.update_record(record_id, patient_name="Renamed", notes="follow-up")
        record = storage.get_record(record_id)
        self.assertEqual(record["patient_name"], "Renamed")
        self.assertEqual(record["notes"], "follow-up")

    def test_delete_record(self):
        record_id = self._save()
        storage.delete_record(record_id)
        self.assertIsNone(storage.get_record(record_id))
        self.assertEqual(storage.count_records(), 0)

    def test_persistence_across_connections(self):
        record_id = self._save()
        # A fresh module-level DumpDatabase / re-open must still see the row.
        storage.init_db()
        record = storage.get_record(record_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["patient_name"], "Ananya Sharma")

    def test_get_latest_record_returns_newest_for_patient(self):
        self._save(patient_id="PT-101", created_at="2026-03-01T09:00:00")
        newest = self._save(patient_id="PT-101", created_at="2026-03-05T09:00:00")
        self._save(patient_id="PT-200", created_at="2026-03-09T09:00:00")
        latest = storage.get_latest_record("PT-101")
        self.assertEqual(latest["id"], newest)
        self.assertIsNone(storage.get_latest_record("PT-404"))

    def test_list_patients_groups_and_sorts(self):
        self._save(patient_id="PT-101", name="Ananya Sharma", created_at="2026-03-01T09:00:00")
        self._save(patient_id="PT-101", name="Ananya Sharma", created_at="2026-03-05T09:00:00")
        self._save(patient_id="PT-200", name="Priya Nair", created_at="2026-04-01T09:00:00")

        patients = storage.list_patients()
        self.assertEqual([p["patient_id"] for p in patients], ["PT-200", "PT-101"])
        first = patients[1]
        self.assertEqual(first["assessment_count"], 2)
        self.assertEqual(first["last_assessment"], "2026-03-05T09:00:00")
        self.assertEqual(first["latest"]["patient_name"], "Ananya Sharma")
        self.assertEqual(first["latest"]["maternal_prediction"], "High Risk")

    def test_list_patients_query_filters_by_name_and_id(self):
        self._save(patient_id="PT-101", name="Ananya Sharma", created_at="2026-03-01T09:00:00")
        self._save(patient_id="PT-200", name="Priya Nair", created_at="2026-04-01T09:00:00")
        self.assertEqual([p["patient_id"] for p in storage.list_patients(query="Priya")], ["PT-200"])
        self.assertEqual([p["patient_id"] for p in storage.list_patients(query="PT-10")], ["PT-101"])
        self.assertEqual(storage.list_patients(query="nobody"), [])

    def test_count_patients_counts_distinct_ids(self):
        self._save(patient_id="PT-101", created_at="2026-03-01T09:00:00")
        self._save(patient_id="PT-101", created_at="2026-03-05T09:00:00")
        self._save(patient_id="PT-200", created_at="2026-04-01T09:00:00")
        self.assertEqual(storage.count_patients(), 2)
        self.assertEqual(storage.count_records(), 3)

    def test_shap_frames_are_serialised(self):
        import pandas as pd

        frame = pd.DataFrame([
            {"Feature": "SystolicBP", "Display name": "Systolic BP", "Hybrid Impact": 0.4},
            {"Feature": "BS", "Display name": "Blood sugar", "Hybrid Impact": -0.2},
        ])
        record_id = storage.save_assessment(
            patient_id="PT-9", patient_name="Shap Patient",
            maternal_inputs={"Age": 30}, cbc_inputs={"MCV": 80.0},
            maternal_result=_maternal_result(), anemia_result=_anemia_result(),
            shap={"maternal": frame, "anemia": None},
        )
        record = storage.get_record(record_id)
        self.assertEqual(len(record["shap_maternal"]), 2)
        self.assertEqual(record["shap_maternal"][0]["Feature"], "SystolicBP")
        self.assertIsNone(record["shap_anemia"])


if __name__ == "__main__":
    unittest.main()
