"""Offline unit tests. Fixtures are not HRP data or model performance results."""
import importlib.util
import json
import unittest
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src/hrp/evidence_adapter.py"
spec = importlib.util.spec_from_file_location("part05_evidence_adapter", MODULE)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class EvidenceAdapterTests(unittest.TestCase):
    def setUp(self):
        self.first = {
            "product_ndc": "0001-001", "name": "Fictional Product A",
            "brand_name": "Fictional Product A", "generic_name": "",
            "dosage_form": "CREAM", "route": ["TOPICAL"],
            "active_ingredients": [{"name": "ingredient alpha", "strength": "1 mg"}],
            "ingredient_identities": ["not-for-prompt"],
            "fda_label": {"label_id": "L1", "label_set_id": "S1", "effective_time": "20200101",
                          "active_ingredient_text": ["ingredient alpha 1 mg"]},
            "rxnorm_product": {"rxnorm_rxcui": "123"},
            "rxnorm_ingredients": {"in_names": ["ingredient alpha"], "pin_names": [],
                                    "in_rxcuis": ["555"], "pin_rxcuis": []},
        }
        self.second = {**self.first, "product_ndc": "0001-002", "name": "Fictional Product A"}
        self.request = {"name": "Fictional Product A"}
        self.source = {"products": [
            {"description": self.request, "reason": "matched_source_candidates",
             "candidates": [self.second, self.first], "prediction": {"should": "not leak"}}
        ]}

    def test_no_answers_or_baseline_predictions_in_prompt(self):
        payload = adapter.build_evidence_payload(self.request, self.source)
        content = json.dumps(payload)
        self.assertNotIn("ingredient_identities", content)
        self.assertNotIn("not-for-prompt", content)
        self.assertNotIn('"prediction"', content)
        self.assertEqual(payload["products"][0]["candidates"][0]["product_ndc"], "0001-001")
        messages = adapter.build_messages(payload)
        self.assertEqual(len(messages), 2)
        self.assertEqual(len(adapter.prompt_sha256(messages)), 64)

    def test_overflow_is_explicit_and_no_unique_resolution_is_inferred(self):
        payload = adapter.build_evidence_payload(self.request, self.source, max_candidates=1)
        self.assertEqual(payload["products"][0]["candidate_count_full_catalog"], 2)
        self.assertTrue(payload["products"][0]["candidates_truncated"])
        self.assertEqual(len(payload["products"][0]["candidates"]), 1)
        self.assertNotIn("status", json.dumps(payload))

    def test_forbidden_field_rejected_at_any_depth(self):
        with self.assertRaises(AssertionError):
            adapter.build_messages({"a": [{"reference_json": "do not reveal"}]})

    def test_mismatched_retrieval_input_rejected(self):
        with self.assertRaises(ValueError):
            adapter.build_evidence_payload({"name": "Another product"}, self.source)

    def test_comparison_shape_and_one_product_contract(self):
        second_request = {"name": "Fictional Product B"}
        comparison = {"product_a": self.request, "product_b": second_request}
        evidence = {"products": [
            self.source["products"][0],
            {"description": second_request, "reason": "identifier_not_in_catalog", "candidates": []},
        ]}
        payload = adapter.build_evidence_payload(comparison, evidence)
        self.assertEqual(len(payload["products"]), 2)
        self.assertEqual(payload["products"][1]["candidate_count_full_catalog"], 0)

    def test_rejects_unapproved_input_metadata(self):
        for extra in ("reference_json", "partition", "ingredient_identities"):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                adapter.build_evidence_payload({"name": "X", extra: "leak"}, self.source)


if __name__ == "__main__":
    unittest.main()
