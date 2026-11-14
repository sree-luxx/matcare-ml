"""Model loading tests: artefact presence and bundle structure."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.model_loader import load_bundle, project_root

REQUIRED_KEYS = {
    "rf", "xgb", "hybrid", "scaler", "preprocessor",
    "w_rf", "w_xgb", "metrics", "weights", "config", "audit",
}


class ModelLoadingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_both_modules_present(self):
        self.assertEqual(set(self.bundle), {"maternal", "anemia"})

    def test_each_module_exposes_required_keys(self):
        for module, data in self.bundle.items():
            self.assertTrue(REQUIRED_KEYS.issubset(data), msg=f"{module} missing keys")

    def test_weights_are_a_convex_combination(self):
        for module, data in self.bundle.items():
            total = data["w_rf"] + data["w_xgb"]
            self.assertAlmostEqual(total, 1.0, places=6, msg=f"{module} weights sum to {total}")

    def test_config_feature_names_match_expected_inputs(self):
        maternal = self.bundle["maternal"]["config"]["feature_names"]
        self.assertEqual(maternal, ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"])
        anemia = self.bundle["anemia"]["config"]["feature_names"]
        self.assertEqual(anemia, ["MCV", "MCH", "MCHC"])

    def test_anemia_excludes_hemoglobin_and_sex(self):
        excluded = self.bundle["anemia"]["config"].get("excluded_features", [])
        self.assertIn("Hemoglobin", excluded)
        self.assertIn("Gender", excluded)

    def test_artefacts_exist_on_disk(self):
        root = project_root()
        for module, directory in (("maternal", "general_risk"), ("anemia", "anemia")):
            for name in ("rf_model.pkl", "xgb_model.pkl", "hybrid_model.pkl",
                         "scaler.pkl", "preprocessor.pkl", "evaluation_metrics.json"):
                path = os.path.join(root, "models", directory, name)
                self.assertTrue(os.path.exists(path), msg=f"missing {module}/{name}")


if __name__ == "__main__":
    unittest.main()
