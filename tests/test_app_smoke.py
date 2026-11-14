"""End-to-end smoke test for the multi-page Streamlit application.

Runs ``app/main.py`` headlessly with ``AppTest`` to confirm the app boots, the
sidebar navigation is present, both screening modules run on submit, saving
creates a patient record and the full navigation flow
(assessment → results → save → profiles → overview → dashboard) works.
"""

import os
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from streamlit.testing.v1 import AppTest

APP = os.path.join(PROJECT_ROOT, "app", "main.py")

NAV_PAGES = [
    "Patient Assessment",
    "Results",
    "Patient Profiles",
    "Patient Overview",
    "Patient Dashboard",
]

# Minimal prediction stubs for tests that seed the database directly.
_MOTHER_RESULT = {
    "predicted_class": "High Risk", "confidence": 80.0,
    "probabilities": {"Random Forest": [0.1, 0.1, 0.8], "XGBoost": [0.1, 0.1, 0.8],
                      "Adaptive hybrid": [0.1, 0.1, 0.8]},
    "w_rf": 0.9, "w_xgb": 0.1,
}
_ANEMIA_RESULT = {
    "predicted_class": "Anemic", "confidence": 60.0,
    "probabilities": {"Random Forest": [0.4, 0.6], "XGBoost": [0.4, 0.6],
                      "Adaptive hybrid": [0.4, 0.6]},
    "w_rf": 0.6, "w_xgb": 0.4,
}


class AppSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Redirect the SQLite database to a temp file so the real one is untouched.
        from app import storage

        cls._tmp = tempfile.mkdtemp()
        cls._orig_dir, cls._orig_path = storage.DATA_DIR, storage.DB_PATH
        storage.DATA_DIR = cls._tmp
        storage.DB_PATH = os.path.join(cls._tmp, "smoke.db")

    @classmethod
    def tearDownClass(cls):
        from app import storage

        storage.DATA_DIR, storage.DB_PATH = cls._orig_dir, cls._orig_path
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _fresh(self):
        return AppTest.from_file(APP, default_timeout=120).run()

    def _fresh_with_patient(self, name="Test Patient"):
        """Boot the app and register a patient name on the assessment form.

        The patient ID is auto-assigned by the app, so only the name is needed.
        """
        at = self._fresh()
        return at.text_input(key="p_name").set_value(name).run()

    def test_app_boots_on_the_assessment_page(self):
        at = self._fresh()
        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])

        sidebar_labels = [button.label for button in at.sidebar.button]
        for page in NAV_PAGES:
            self.assertIn(page, sidebar_labels)

        # The structured form lives in the main area, not the sidebar.
        self.assertIn("Submit assessment", [button.label for button in at.button])
        self.assertIn("Patient name", [box.label for box in at.text_input])
        self.assertIn("Patient ID", [box.label for box in at.text_input])
        # The patient ID is pre-filled with an auto-assigned identifier.
        self.assertTrue(at.session_state["p_id"].startswith("PT-"))

    def test_submit_runs_both_modules_and_opens_results(self):
        at = self._fresh_with_patient()
        at = next(button for button in at.button if button.label == "Submit assessment").click().run()

        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertEqual(at.session_state["page"], "Results")
        self.assertTrue(any("Assessment results" in block.value for block in at.markdown))
        self.assertIn("Save assessment", [button.label for button in at.button])
        # Both modules produced a result for the preset values.
        self.assertIsNotNone(at.session_state["live_context"])

    def test_submit_without_patient_name_is_blocked(self):
        at = self._fresh()
        at = next(button for button in at.button if button.label == "Submit assessment").click().run()

        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertEqual(at.session_state["page"], "Patient Assessment")
        self.assertTrue(any("Enter both a patient name" in item.value for item in at.error))
        self.assertIsNone(at.session_state["live_context"])

    def test_submit_with_conflicting_patient_id_is_blocked(self):
        from app import storage

        storage.save_assessment(
            patient_id="PT-CONFLICT", patient_name="Alice",
            maternal_inputs={"Age": 30, "SystolicBP": 120, "DiastolicBP": 78,
                             "BS": 6.0, "BodyTemp": 98.2, "HeartRate": 76},
            cbc_inputs={"Gender": 0, "Hemoglobin": 12.5, "RBC": 4.5, "MCV": 88.0,
                        "MCH": 29.0, "MCHC": 33.0},
            maternal_result=_MOTHER_RESULT, anemia_result=_ANEMIA_RESULT,
            created_at="2026-05-01T09:00:00",
        )

        at = self._fresh()
        at = at.text_input(key="p_name").set_value("Bob").run()
        at = at.text_input(key="p_id").set_value("PT-CONFLICT").run()
        at = next(button for button in at.button if button.label == "Submit assessment").click().run()

        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertEqual(at.session_state["page"], "Patient Assessment")
        self.assertTrue(any("already belongs to" in item.value for item in at.error))
        self.assertIsNone(at.session_state["live_context"])

    def test_next_patient_id_is_unused(self):
        from app import storage
        from app.sections import assessment_form

        fresh_id = assessment_form.next_patient_id()
        known_id = {patient["patient_id"] for patient in storage.list_patients()}
        self.assertTrue(fresh_id.startswith("PT-"))
        self.assertNotIn(fresh_id, known_id)

    def test_full_navigation_flow(self):
        from app import storage

        at = self._fresh_with_patient()
        at = next(button for button in at.button if button.label == "Submit assessment").click().run()
        at = next(button for button in at.button if button.label == "Save assessment").click().run()

        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertEqual(at.session_state["page"], "Patient Overview")
        self.assertGreaterEqual(storage.count_records(), 1)
        self.assertTrue(any(button.label == "View full dashboard →" for button in at.button))

        at = next(button for button in at.button if button.label == "View full dashboard →").click().run()
        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertEqual(at.session_state["page"], "Patient Dashboard")
        self.assertTrue(any("Historical trends" in block.value for block in at.markdown))

        at = next(button for button in at.sidebar.button if button.label == "Patient Profiles").click().run()
        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertEqual(at.session_state["page"], "Patient Profiles")
        self.assertTrue(any(button.label == "Open overview" for button in at.button))

    def test_registering_a_second_patient_creates_a_second_profile(self):
        from app import storage

        before = storage.count_patients()
        at = self._fresh_with_patient("Alice")
        at = next(button for button in at.button if button.label == "Submit assessment").click().run()
        at = next(button for button in at.button if button.label == "Save assessment").click().run()
        first_id = at.session_state["selected_patient_id"]

        # Return to the form and register a different patient under a fresh ID.
        at = next(button for button in at.sidebar.button if button.label == "Patient Assessment").click().run()
        at = next(button for button in at.button if button.label == "New patient").click().run()
        at = at.text_input(key="p_name").set_value("Bob").run()
        at = next(button for button in at.button if button.label == "Submit assessment").click().run()
        at = next(button for button in at.button if button.label == "Save assessment").click().run()
        second_id = at.session_state["selected_patient_id"]

        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertNotEqual(first_id, second_id)
        self.assertEqual(storage.count_patients(), before + 2)

    def test_dashboard_history_table_and_trends_render(self):
        from app import storage

        patient_id = "PT-MULTI"
        base = {"Age": 30, "SystolicBP": 130, "DiastolicBP": 85,
                "BS": 8.0, "BodyTemp": 98.6, "HeartRate": 80}
        cbc = {"Gender": 0, "Hemoglobin": 11.0, "RBC": 4.2, "MCV": 84.0, "MCH": 27.0, "MCHC": 32.0}
        maternal = {
            "predicted_class": "High Risk", "confidence": 80.0,
            "probabilities": {"Random Forest": [0.1, 0.1, 0.8], "XGBoost": [0.1, 0.1, 0.8],
                              "Adaptive hybrid": [0.1, 0.1, 0.8]},
            "w_rf": 0.9, "w_xgb": 0.1,
        }
        anemia = {
            "predicted_class": "Anemic", "confidence": 60.0,
            "probabilities": {"Random Forest": [0.4, 0.6], "XGBoost": [0.4, 0.6],
                              "Adaptive hybrid": [0.4, 0.6]},
            "w_rf": 0.6, "w_xgb": 0.4,
        }
        storage.save_assessment(patient_id=patient_id, patient_name="Multi Patient",
                                maternal_inputs=base, cbc_inputs=cbc,
                                maternal_result=maternal, anemia_result=anemia,
                                created_at="2026-01-01T09:00:00")
        storage.save_assessment(patient_id=patient_id, patient_name="Multi Patient",
                                maternal_inputs={**base, "SystolicBP": 145}, cbc_inputs=cbc,
                                maternal_result=maternal, anemia_result=anemia,
                                created_at="2026-02-01T09:00:00")

        at = AppTest.from_file(APP, default_timeout=120).run()
        at.session_state["page"] = "Patient Dashboard"
        at.session_state["selected_patient_id"] = patient_id
        at.run()

        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        # A View details action exists for the other historical assessment.
        self.assertIn("View details", [button.label for button in at.button])
        # Trend charts section is rendered for a patient with two assessments.
        self.assertTrue(any("Historical trends" in block.value for block in at.markdown))
        # Previous vs latest comparison is present.
        self.assertTrue(any("Previous vs latest assessment" in block.value for block in at.markdown))

    def test_delete_one_assessment_from_profiles(self):
        from app import storage

        patient_id, name = "PT-DELETE", "Deletable Patient"
        inputs = {"Age": 30, "SystolicBP": 120, "DiastolicBP": 78,
                  "BS": 6.0, "BodyTemp": 98.2, "HeartRate": 76}
        cbc = {"Gender": 0, "Hemoglobin": 12.5, "RBC": 4.5, "MCV": 88.0,
               "MCH": 29.0, "MCHC": 33.0}
        storage.save_assessment(patient_id=patient_id, patient_name=name,
                                maternal_inputs=inputs, cbc_inputs=cbc,
                                maternal_result=_MOTHER_RESULT, anemia_result=_ANEMIA_RESULT,
                                created_at="2026-03-01T09:00:00")
        storage.save_assessment(patient_id=patient_id, patient_name=name,
                                maternal_inputs={**inputs, "SystolicBP": 148}, cbc_inputs=cbc,
                                maternal_result=_MOTHER_RESULT, anemia_result=_ANEMIA_RESULT,
                                created_at="2026-03-15T09:00:00")
        self.assertEqual(len(storage.get_history(patient_id)), 2)

        at = AppTest.from_file(APP, default_timeout=120).run()
        at.session_state["page"] = "Patient Profiles"
        at.run()
        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])

        at = at.checkbox(key=f"delete_confirm_{patient_id}").set_value(True).run()
        at = at.button(key=f"delete_button_{patient_id}").click().run()

        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        history = storage.get_history(patient_id)
        # Exactly one assessment was removed; the chosen one was the newest.
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["created_at"], "2026-03-01T09:00:00")

    def test_profiles_search_filters_cards(self):
        from app import storage

        storage.save_assessment(
            patient_id="PT-SEARCH", patient_name="Searchable Patient",
            maternal_inputs={"Age": 30, "SystolicBP": 120, "DiastolicBP": 78,
                             "BS": 6.0, "BodyTemp": 98.2, "HeartRate": 76},
            cbc_inputs={"Gender": 0, "Hemoglobin": 12.5, "RBC": 4.5, "MCV": 88.0,
                        "MCH": 29.0, "MCHC": 33.0},
            maternal_result={"predicted_class": "Low Risk", "confidence": 70.0,
                             "probabilities": {"Random Forest": [0.7, 0.2, 0.1],
                                               "XGBoost": [0.7, 0.2, 0.1],
                                               "Adaptive hybrid": [0.7, 0.2, 0.1]},
                             "w_rf": 0.9, "w_xgb": 0.1},
            anemia_result={"predicted_class": "Non-Anemic", "confidence": 65.0,
                           "probabilities": {"Random Forest": [0.65, 0.35], "XGBoost": [0.65, 0.35],
                                             "Adaptive hybrid": [0.65, 0.35]},
                           "w_rf": 0.6, "w_xgb": 0.4},
            created_at="2026-04-01T09:00:00",
        )

        at = AppTest.from_file(APP, default_timeout=120).run()
        at.session_state["page"] = "Patient Profiles"
        at.run()
        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])

        at.text_input(key="patient_search").set_value("Searchable").run()
        self.assertEqual(list(at.exception), [], msg=[e.value for e in at.exception])
        self.assertTrue(any(button.label == "Open overview" for button in at.button))

        at.text_input(key="patient_search").set_value("no-such-patient").run()
        self.assertTrue(any("No patient profile matches" in block.value for block in at.markdown))


if __name__ == "__main__":
    unittest.main()
