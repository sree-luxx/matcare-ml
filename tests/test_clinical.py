"""Clinical reference logic tests."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.clinical import (
    anemia_screening,
    anemia_severity,
    build_vitals_rows,
    evaluate,
    morphology,
)


class ClinicalTests(unittest.TestCase):
    def test_evaluate_normal_and_out_of_range(self):
        self.assertEqual(evaluate("SystolicBP", 115)[0], "normal")
        self.assertEqual(evaluate("SystolicBP", 125)[0], "warning")
        self.assertEqual(evaluate("SystolicBP", 150)[0], "critical")
        self.assertEqual(evaluate("HeartRate", 50)[0], "warning")

    def test_unknown_feature_is_info(self):
        self.assertEqual(evaluate("NotAFeature", 1)[0], "info")

    def test_anemia_severity_bands(self):
        self.assertEqual(anemia_severity(6.5)[0], "Severe anemia")
        self.assertEqual(anemia_severity(8.5)[0], "Moderate anemia")
        self.assertEqual(anemia_severity(10.5)[0], "Mild anemia")
        self.assertEqual(anemia_severity(12.5)[0], "Non-anemic range")
        self.assertEqual(anemia_severity(None)[0], "Unknown")

    def test_morphology_bands(self):
        self.assertIn("Microcytic", morphology(72)[0])
        self.assertIn("Macrocytic", morphology(105)[0])
        self.assertIn("Normocytic", morphology(90)[0])

    def test_anemia_screening_severe_microcytic(self):
        label, headline, status, detail = anemia_screening(6.5, 72)
        self.assertEqual(label, "Anaemic")
        self.assertEqual(headline, "Anaemic \u2014 severe, microcytic")
        self.assertEqual(status, "critical")
        self.assertIn("microcytic", detail)

    def test_anemia_screening_moderate_matches_morphology(self):
        label, headline, status, _ = anemia_screening(8.5, 72)
        self.assertEqual(label, "Anaemic")
        self.assertEqual(headline, "Anaemic \u2014 moderate, microcytic")
        self.assertEqual(status, "critical")

    def test_anemia_screening_non_anaemic(self):
        label, headline, status, _ = anemia_screening(12.5, 90)
        self.assertEqual(label, "Non-anaemic")
        self.assertEqual(headline, "Non-anaemic \u2014 normocytic")
        self.assertEqual(status, "normal")

    def test_anemia_screening_missing_haemoglobin(self):
        label, headline, status, _ = anemia_screening(None, None)
        self.assertEqual(label, "Not available")
        self.assertEqual(headline, "Not available")
        self.assertEqual(status, "info")

    def test_build_vitals_rows_skips_unbounded(self):
        rows = build_vitals_rows({"Age": 30, "Hemoglobin": 12.0, "Gender": 0})
        parameters = {row["parameter"] for row in rows}
        self.assertIn("Maternal age", parameters)
        self.assertIn("Hemoglobin", parameters)
        self.assertNotIn("Recorded sex code", parameters)


if __name__ == "__main__":
    unittest.main()
