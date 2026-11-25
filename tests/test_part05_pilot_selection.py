"""Selection-policy tests using fictitious metadata, never scored patient data."""
import importlib.util
import json
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "tools/part05_prepare_pilot.py"
spec = importlib.util.spec_from_file_location("part05_prepare_pilot", MODULE)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotSelectionTests(unittest.TestCase):
    def test_reproducible_selection_and_stratum_coverage(self):
        records = [
            {"case_id": f"case-{i:03d}", "partition": "validation", "task": "product_identification"
              if i < 10 else "ingredient_comparison", "case_type": "exact" if i % 2 else "ambiguous",
             "family_group_ids_json": json.dumps([f"f{i}"])}
            for i in range(20)
        ]
        first = pilot.select_cases(records, 12, 42)
        second = pilot.select_cases(records, 12, 42)
        self.assertEqual([r["case_id"] for r in first], [r["case_id"] for r in second])
        self.assertEqual(len(first), 12)
        self.assertEqual(len({r["case_id"] for r in first}), 12)
        self.assertEqual(len({r["task"] for r in first}), 2)

    def test_rejects_test_row(self):
        with self.assertRaises(ValueError):
            pilot.select_cases([{"case_id": "x", "partition": "test", "task": "a",
                                "case_type": "b", "family_group_ids_json": '["f"]'}], 1, 42)


if __name__ == "__main__":
    unittest.main()
