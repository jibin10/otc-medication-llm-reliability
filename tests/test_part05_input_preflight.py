"""Unit tests using small local fixtures, not benchmark performance data."""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "part05_input_preflight.py"
SPEC = importlib.util.spec_from_file_location("part05_input_preflight", MODULE_PATH)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class TestPreflight(unittest.TestCase):
    def test_refuses_test_files(self):
        with self.assertRaises(ValueError):
            module.ensure_not_test(Path("data/processed/final_v5/sealed_test.csv"))
        with self.assertRaises(ValueError):
            module.ensure_not_test(Path("results/test/predictions.jsonl"))

    def test_detects_nested_shapes_without_contents(self):
        value = {"gold_ingredients": ["redacted"]}
        shape = module.shape_of(value)
        self.assertIn("gold_ingredients", shape)
        self.assertNotIn("redacted", shape)

    def test_preflight_audits_only_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "data/processed/final_v5").mkdir(parents=True)
            (repo / "results/final_v5/part03").mkdir(parents=True)
            (repo / "data/processed/final_v5/validation.csv").write_text(
                "case_id,task\nval-1,identify\nval-2,compare\n", encoding="utf-8"
            )
            (repo / "data/processed/final_v5/validation_retrieval_predictions.jsonl").write_text(
                '{"case_id":"val-1","prediction":{"status":"resolved"}}\n'
                '{"case_id":"val-2","prediction":{"status":"ambiguous"}}\n',
                encoding="utf-8",
            )
            (repo / "results/final_v5/part03/freeze_manifest.json").write_text("{}\n")
            # Explicitly create a sealed test file to verify it remains untouched.
            sealed = repo / "data/processed/final_v5/sealed_test.csv"
            sealed.write_text("leave unchanged\n", encoding="utf-8")
            before = sealed.stat().st_mtime_ns
            report = module.run(repo, Path("results/part05/input_contract_preflight.json"))
            self.assertEqual(report["validation"]["row_count"], 2)
            self.assertEqual(report["retrieval_predictions"]["row_count"], 2)
            self.assertFalse(report["test_set_opened"])
            self.assertEqual(sealed.stat().st_mtime_ns, before)
            self.assertTrue((repo / "results/part05/input_contract_preflight.json").is_file())


if __name__ == "__main__":
    unittest.main()
