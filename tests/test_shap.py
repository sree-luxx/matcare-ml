"""SHAP explainability tests: local attribution frame shape and content."""

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.inference import predict_module
from app.model_loader import load_bundle
from explainability.shap_explainers import (
    compute_hybrid_global_importances,
    explain_single_patient,
)

MATERNAL_INPUTS = {
    "Age": 36, "SystolicBP": 155, "DiastolicBP": 98, "BS": 15.2, "BodyTemp": 99.4, "HeartRate": 92,
}
ANEMIA_INPUTS = {"Gender": 0, "Hemoglobin": 8.2, "MCV": 72.0, "MCH": 21.0, "MCHC": 28.5}

EXPECTED_COLUMNS = {
    "Feature", "Display name", "Clinical Value", "Hybrid Impact",
    "RF Impact", "XGB Impact", "Direction", "Absolute Impact",
}


class ShapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def _explain(self, module, inputs):
        result = predict_module(self.bundle[module], inputs, module)
        frame = explain_single_patient(
            self.bundle[module]["rf"],
            self.bundle[module]["xgb"],
            result["w_rf"],
            result["w_xgb"],
            result["scaled"].to_numpy(),
            inputs,
            result["feature_names"],
            class_idx=result["predicted_index"],
            class_name=result["predicted_class"],
        )
        return result, frame

    def test_local_frame_has_expected_columns_and_rows(self):
        for module, inputs in (("maternal", MATERNAL_INPUTS), ("anemia", ANEMIA_INPUTS)):
            result, frame = self._explain(module, inputs)
            self.assertTrue(EXPECTED_COLUMNS.issubset(frame.columns))
            self.assertEqual(len(frame), len(result["feature_names"]))

    def test_direction_wording_matches_sign(self):
        _, frame = self._explain("maternal", MATERNAL_INPUTS)
        for _, row in frame.iterrows():
            if row["Hybrid Impact"] > 0:
                self.assertIn("Raises", row["Direction"])
            else:
                self.assertIn("Lowers", row["Direction"])

    def test_anemia_frame_only_contains_index_features(self):
        _, frame = self._explain("anemia", ANEMIA_INPUTS)
        self.assertEqual(set(frame["Feature"]), {"MCV", "MCH", "MCHC"})

    def test_hybrid_importances_are_normalised(self):
        rf = {"MCV": 0.5, "MCH": 0.3, "MCHC": 0.2}
        xgb = {"MCV": 0.4, "MCH": 0.4, "MCHC": 0.2}
        mixed = compute_hybrid_global_importances(rf, xgb, 0.6, 0.4, list(rf))
        self.assertAlmostEqual(sum(mixed.values()), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
