"""CPU regression tests with a synthetic challenge and source catalog."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import stage06_challenge as challenge  # noqa: E402
import stage06_original_gemma_pilot as pilot  # noqa: E402
import stage06_score_pilot as factual  # noqa: E402
from hrp import challenge_evaluation as contract  # noqa: E402


class ChallengeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def test_public_hash_replays_crlf_and_private_hash_is_exact(self):
        name = "synthetic.json"
        (self.root / name).write_bytes(b"{\n}\n")
        freeze = {"outputs": {name: challenge.sha(b"{\r\n}\r\n")}}
        self.assertEqual(challenge.artifact(self.root, name, freeze, public=True), b"{\n}\n")
        with self.assertRaisesRegex(pilot.PilotError, "SHA-256 mismatch"):
            challenge.artifact(self.root, name, freeze)

    def test_challenge_uses_separate_sources_and_compares_identical_eligible_cases(self):
        freeze = {"outputs": {}}
        self.enterContext(patch.object(challenge, "FROZEN", self.root))
        self.enterContext(patch.object(challenge, "ROLES", {
            "different_without_ndc": 1, "supplementary": 1,
        }))

        def save(name, data):
            (self.root / name).write_bytes(data)
            freeze["outputs"][name] = challenge.sha(data)

        def rows(name, data):
            save(name, "".join(contract.canonical(row) + "\n" for row in data).encode())

        ingredient = {"name": "synthetic ingredient", "strength": "1 mg"}
        candidate = {
            "product_ndc": "synthetic-ndc", "dosage_form": "tablet", "route": ["oral"],
            "fda_active_ingredients": [ingredient],
            "spl": {"active_ingredient_text": ["synthetic ingredient 1 mg"]},
            "rxnorm": {"cached_pairs_confirmed": True},
            "provenance": {"original_versions_confirmed": True, "raw_ndc_name_matches_catalog": True},
        }
        formation = {"dosage_form": "tablet", "route": ["oral"], "active_ingredients": [ingredient]}
        exact = {"status": "EXACT_NDC_IDENTIFIED", "product_ndc": "synthetic-ndc",
                 "active_ingredient_names": [ingredient["name"]], "source_ndcs": ["synthetic-ndc"],
                 "formulation": formation}
        no_match = {"status": "NO_MATCH", "product_ndc": None, "active_ingredient_names": [],
                    "source_ndcs": [], "formulation": None}
        definitions = [
            ("case-a", "supplementary", [], no_match, False),
            ("case-b", "different_without_ndc", [candidate], exact, True),
        ]
        evidence, benchmarks, deterministic = [], [], []
        for case_id, role, candidates, target, audited in definitions:
            payload = {"products": [{"candidates": candidates}]}
            digest = challenge.sha(contract.canonical(payload).encode())
            reference = {"products": [target], "comparison": None}
            evidence.append({"case_id": case_id, "payload_sha256": digest, "payload": payload,
                             "messages": contract.build_messages(payload)})
            benchmarks.append({
                "case_id": case_id, "partition": "validation", "scoring_role": role,
                "category": "synthetic", "families": ["shared-family"], "source_reviewed": True,
                "independently_clinically_verified": audited, "common_evidence_sha256": digest,
                "reference": reference,
            })
            deterministic.append({"case_id": case_id, "prediction": reference})
        rows(challenge.EVIDENCE, evidence)
        rows(challenge.BENCHMARK, benchmarks)
        rows(challenge.IDENTITIES, [{"name": ingredient["name"], "uniis": ["AAAAAAAAAA"]}])
        rows("deterministic_predictions.PRIVATE.jsonl", deterministic)
        save("summary.json", b'{"selected":{"groups":1,"categories":{"synthetic":2}}}')
        primary = {"cases": 1, "product_requests": 1, "ingredient_target_requests": 1,
                   "full_formulation_target_requests": 1, "unique_ndc_target_requests": 1}
        secondary = {"cases": 1, "product_requests": 1, "abstention_target_requests": 1}
        save("scoring_specifications.json", json.dumps({
            "fixed_denominators": dict(Counter(primary) + Counter(secondary)),
            "fixed_denominators_by_role": {"different_without_ndc": primary, "supplementary": secondary},
        }).encode())

        output = self.root / "run"
        requests = challenge.inference_requests(self.root, freeze)
        challenge.prepare(self.root, output, freeze, requests)
        schema, schema_hash = pilot.load_schema()
        requests, configuration = challenge.configure_run(output, requests, freeze, {}, schema_hash)
        self.assertEqual(list(requests), ["case-b", "case-a"])
        self.assertEqual(configuration["parser_version"], [pilot.NATIVE_RESPONSE_VERSION, pilot.FENCE_POLICY])
        self.assertTrue(all(set(request) == {"messages", "evidence", "prompt_version",
                                            "payload_sha256", "prompt_sha256"}
                            for request in requests.values()))

        generator = Mock(metadata={})
        generated = []

        def generate(_messages, record):
            case_id = record["case_id"]
            generated.append(case_id)
            wrong = {**exact, "formulation": {**formation, "active_ingredients": [
                {**ingredient, "strength": "2 mg"},
            ]}}
            result = {"products": [wrong if case_id == "case-b" else no_match], "comparison": None}
            text = f"```json\n{json.dumps(result)}\n```"
            record.update(response_text=text, raw_response=text + "<turn|>")

        generator.generate.side_effect = generate
        factory = Mock(return_value=generator)
        pilot.run_prepared(output, requests, schema, configuration, 1, generator_factory=factory)
        save("output_schema.json", pilot.SCHEMA.read_bytes())
        read_bytes = Path.read_bytes

        def evidence_only(path):
            if path.name in {challenge.BENCHMARK, challenge.IDENTITIES}:
                self.fail("Inference opened private scoring inputs")
            return read_bytes(path)

        with (patch.object(Path, "read_bytes", evidence_only),
              patch.object(challenge, "frozen_manifest", return_value=(freeze, {})),
              patch.object(pilot, "GemmaGenerator", factory),
              contextlib.redirect_stdout(io.StringIO())):
            self.assertEqual(challenge.main([
                "infer", "--challenge-dir", str(self.root), "--output", str(output), "--limit", "2",
            ]), 0)
        self.assertEqual(generated, ["case-b", "case-a"])
        challenge.score(self.root, output, freeze, requests, configuration, schema)
        report = json.loads((output / "factual_validation_metrics.json").read_bytes())
        gemma = report["models"]["original_gemma4"]
        retrieval = report["models"]["deterministic_retrieval"]
        self.assertEqual(gemma["overall"]["full_formulation_accuracy"]["numerator"], 0)
        self.assertEqual(retrieval["overall"]["full_formulation_accuracy"]["numerator"], 1)
        self.assertEqual(gemma["overall"]["full_formulation_accuracy"]["denominator"], 1)
        self.assertEqual(gemma["by_category"]["synthetic"]["family_groups"], 1)
        self.assertEqual(gemma["by_reference_provenance"]["independently_audited"]["overall"]["counts"]["cases"], 1)
        self.assertEqual(gemma["by_role"]["supplementary"]["abstention_accuracy"]["numerator"], 1)
        self.assertEqual(gemma["finding_counts"], {"formulation_error": 1})
        self.assertNotIn("case-b", json.dumps(report))

    def test_pilot_labels_supply_eligibility_and_identity_without_baseline_predictions(self):
        resolver = contract.IdentityResolver([{"name": "synthetic", "uniis": ["AAAAAAAAAA"]}])
        target = {"formulation_supported": True, "product_ndc": "synthetic",
                  "ingredient_identities": ["fda_unii:AAAAAAAAAA"], "dosage_form": "tablet",
                  "route": ["oral"], "active_ingredients": [
                      {"name": "synthetic", "identity": "fda_unii:AAAAAAAAAA", "strength": "1 mg"},
                  ]}
        actual = {"status": "EXACT_NDC_IDENTIFIED", "product_ndc": "synthetic",
                  "active_ingredient_names": ["unmapped name"], "formulation": None}
        counts = factual.score_case({"product": target}, {"products": [actual]}, resolver)
        report = factual.summarize(counts)
        self.assertEqual(report["full_formulation_accuracy"], {"numerator": 0, "denominator": 1, "value": 0})
        self.assertEqual(counts["ingredient_fp"], 1)
        self.assertEqual(counts["ingredient_fn"], 1)
        self.assertIsNone(report["overlap_label_accuracy"]["value"])


if __name__ == "__main__":
    unittest.main()
