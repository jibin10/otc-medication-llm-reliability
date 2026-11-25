"""Synthetic checks for the original Gemma pilot. No model or benchmark data is loaded."""

import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from jsonschema import Draft202012Validator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import stage06_original_gemma_pilot as pilot  # noqa: E402
import stage06_prepare_native_resume as native_resume  # noqa: E402
import stage06_prepare_pilot_prompts as prepare  # noqa: E402
from hrp.evidence_adapter import build_messages  # noqa: E402

EVIDENCE = {"products": [{"candidates": [{"product_ndc": "synthetic-source"}]}]}
OUTPUT = {
    "products": [{
        "status": "EXACT_NDC_IDENTIFIED",
        "product_ndc": "synthetic-source",
        "active_ingredient_names": ["synthetic ingredient"],
        "source_ndcs": ["synthetic-source"],
        "formulation": None,
    }],
    "comparison": None,
}


class Stage06PilotTests(unittest.TestCase):
    def setUp(self):
        self.schema, _ = pilot.load_schema()
        self.validator = Draft202012Validator(self.schema)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.input = root / "prepared.PRIVATE.jsonl"
        self.legacy = root / "legacy.PRIVATE.jsonl"
        self.output = root / "run"
        self.legacy_rows = []
        for i in (2, 1, 3):
            evidence = EVIDENCE if i != 3 else {
                "products": [*EVIDENCE["products"], {"candidates": [
                    {"product_ndc": "synthetic-source-2", "original_text": "synthetic café"},
                ]}],
            }
            messages = build_messages(evidence)
            messages[0]["content"] += f" Synthetic wording {i}  \n"
            messages[1]["content"] += " \n "
            self.legacy_rows.append({"case_id": f"synthetic-case-{i}",
                                     "prompt_version": pilot.LEGACY_PROMPT_VERSION,
                                     "messages": messages})
        self.legacy.write_text(pilot.jsonl(self.legacy_rows), encoding="utf-8")
        requests, _ = pilot.read_requests(self.legacy, legacy=True)
        self.rows = list(prepare.corrected_rows(requests, self.schema))
        self.input.write_text(pilot.jsonl(self.rows), encoding="utf-8")

    def test_correction_preserves_case_order_family_assignments_and_source_text(self):
        original, _ = pilot.read_requests(self.legacy, legacy=True)
        corrected, _ = pilot.read_requests(self.input)
        self.assertEqual(list(original), list(corrected))
        for before, after in zip(original.values(), corrected.values(), strict=True):
            before_messages, after_messages = before["messages"], after["messages"]
            self.assertEqual(before_messages[0], after_messages[0])
            self.assertEqual(
                before_messages[1]["content"].split(pilot.SOURCE_MARKER, 1)[1].encode("utf-8"),
                after_messages[1]["content"].split(pilot.SOURCE_MARKER, 1)[1].encode("utf-8"),
            )
            self.assertEqual(before["evidence"], after["evidence"])

        validation = self.input.with_name("validation.csv")
        with validation.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["case_id", "partition", "family_group_ids_json", "reference_json"])
            for index, case_id in enumerate(reversed(original)):
                writer.writerow([case_id, "validation", json.dumps([f"family-{index}"]), "unused"])
        before = prepare.selected_family_metadata(validation, original)
        self.assertEqual(before, prepare.selected_family_metadata(validation, corrected))
        self.assertEqual(before["unique_family_group_ids"], 3)
        self.assertNotIn("family-", self.input.read_text())

    def test_embedded_schema_supports_complete_formulation_and_pair_comparison(self):
        instructions, _, _ = self.rows[2]["messages"][1]["content"].partition(pilot.SOURCE_MARKER)
        embedded = json.loads(instructions.split(pilot.SCHEMA_MARKER, 1)[1])
        self.assertEqual(embedded, self.schema)
        product_schema = embedded["properties"]["products"]["items"]
        self.assertIn("formulation", product_schema["required"])
        self.assertNotIn("reason", product_schema["properties"])
        formulation = {"active_ingredients": [{"name": "synthetic ingredient", "strength": "1 mg"}],
                       "dosage_form": "synthetic tablet", "route": ["oral"]}
        first = {**OUTPUT["products"][0], "formulation": formulation}
        second = {**first, "status": "INGREDIENT_SET_IDENTIFIED", "product_ndc": None,
                  "source_ndcs": ["synthetic-source-2"], "formulation": None}
        output = {
            "products": [first, second],
            "comparison": {"ingredient_overlap": "shared", "same_active_ingredient_set": True,
                           "shared_active_ingredient_names": ["synthetic ingredient"]},
        }
        requests, _ = pilot.read_requests(self.input)
        evidence = requests[self.rows[2]["case_id"]]["evidence"]
        self.assertEqual(
            pilot.validate_response(json.dumps(output), Draft202012Validator(embedded), evidence),
            (output, None),
        )

    def test_preflight_rejects_legacy_and_modified_instructions_before_gpu_loading(self):
        with patch.object(pilot, "GemmaGenerator") as factory:
            with self.assertRaisesRegex(pilot.PilotError, "Expected prompt version"):
                pilot.run(self.legacy, self.output, 2)
            message = self.rows[2]["messages"][1]
            message["content"] = message["content"].replace('"formulation"', '"reason"', 1)
            self.input.write_text(pilot.jsonl(self.rows), encoding="utf-8")
            with self.assertRaisesRegex(pilot.PilotError, "Incompatible output instructions at line 3"):
                pilot.run(self.input, self.output, 2)
            factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_strict_json_and_frozen_contract(self):
        older_prompt_output = {**OUTPUT, "products": [dict(OUTPUT["products"][0])]}
        older_prompt_output["products"][0].pop("formulation")
        older_prompt_output["products"][0]["reason"] = "synthetic reason"
        cases = [
            ("```json\n" + json.dumps(OUTPUT) + "\n```", "invalid_json"),
            (json.dumps(OUTPUT) + "\nextra prose", "invalid_json"),
            ('{"products": [], "products": []}', "invalid_json"),
            ('{"number":NaN}', "invalid_json"),
            ('{"number":1e999}', "invalid_json"),
            ("null", "invalid_schema"),
            (json.dumps(older_prompt_output), "invalid_schema"),
        ]
        for text, category in cases:
            with self.subTest(category=category, text=text):
                value, error = pilot.validate_response(text, self.validator, EVIDENCE)
                self.assertIsNone(value)
                self.assertEqual(error["category"], category)
        value, error = pilot.validate_response(json.dumps(OUTPUT), self.validator, EVIDENCE)
        self.assertEqual(value, OUTPUT)
        self.assertIsNone(error)
        _, error = pilot.validate_response(json.dumps(OUTPUT), self.validator,
                                           {"products": [{"candidates": []}]})
        self.assertEqual(error["category"], "invalid_contract")

    def test_generation_preserves_raw_decoding_and_counts_eos(self):
        generator = pilot.GemmaGenerator.__new__(pilot.GemmaGenerator)
        generator.response_decoding = pilot.LEGACY_RESPONSE_VERSION
        generator.torch = Mock(inference_mode=contextlib.nullcontext)
        generator.device = "synthetic-device"
        generator.config = object()
        generator.processor = Mock()
        generator.model = Mock()
        ids = [71, 72, 106]
        generator.model.generate.return_value = MagicMock()
        generator.model.generate.return_value.__getitem__.return_value.tolist.return_value = ids
        tokenized = {"input_ids": Mock(shape=(1, 5))}
        generator.processor.apply_chat_template.return_value.to.return_value = tokenized
        generator.processor.decode.side_effect = ['{}<eos>', '{}']
        messages = self.rows[0]["messages"]
        record = {}
        generator.generate(messages, record)
        generator.model.generate.assert_called_once_with(
            **tokenized, generation_config=generator.config,
        )
        self.assertIs(generator.processor.apply_chat_template.call_args.args[0], messages)
        template_options = generator.processor.apply_chat_template.call_args.kwargs
        self.assertFalse(template_options["enable_thinking"])
        self.assertEqual([call.args[0] for call in generator.processor.decode.call_args_list],
                         [ids, ids[:-1]])
        self.assertNotIn("error", record)
        self.assertEqual(record["raw_response"], '{}<eos>')
        self.assertEqual(record["response_text"], '{}')
        self.assertEqual(record["input_tokens"], 5)
        self.assertEqual(record["output_tokens"], 3)
        self.assertEqual(record["stop_token_id"], 106)

    def test_native_content_goes_to_strict_validation_without_repair(self):
        valid = json.dumps(OUTPUT)
        processor = Mock()
        template = {"fields": {"content": {"content": "text"}}}
        processor.tokenizer.response_template = template
        prefix = "synthetic chat prefix"
        thought = "<|channel>thought\nSynthetic thought.<channel|>"
        for content, category in [
            (valid, None),
            ("```json\n" + valid + "\n```", "invalid_json"),
            ('{"products": [}', "invalid_json"),
            ("\N{NO-BREAK SPACE}" + valid, "invalid_json"),
            ("Intro " + valid, "invalid_json"),
            (valid + " trailing", "invalid_json"),
            ('{"wrong_field": null}', "invalid_schema"),
        ]:
            with self.subTest(category=category):
                processor.parse_response.return_value = {"role": "assistant", "content": content}
                raw = thought + content + "<turn|>"
                extracted = pilot.extract_assistant_content(
                    processor, raw, prefix=prefix, stop_token_id=106,
                )
                self.assertEqual(extracted, content)
                args, kwargs = processor.parse_response.call_args
                self.assertEqual(args, (raw,))
                self.assertEqual(kwargs["prefix"], prefix)
                self.assertFalse(kwargs["schema"]["fields"]["content"]["content_args"]["strip"])
                self.assertNotIn("content_args", template["fields"]["content"])
                value, error = pilot.validate_response(extracted, self.validator, EVIDENCE)
                if category is None:
                    self.assertEqual(value, OUTPUT)
                    self.assertIsNone(error)
                else:
                    self.assertEqual(error["category"], category)
        _, error = pilot.validate_response(thought + valid, self.validator, EVIDENCE)
        self.assertEqual(error["category"], "invalid_json")

    def test_native_decoder_rejects_envelopes_that_would_discard_answer_text(self):
        processor = Mock()
        processor.tokenizer.response_template = {"fields": {"content": {"content": "text"}}}
        processor.parse_response.return_value = {"role": "assistant", "content": "{}"}
        for raw, stop_id, code in [
            ("Intro<|channel>thought\n<channel|>{}<turn|>", 106, "assistant_content_changed"),
            ("intro<turn|>{}<turn|>", 106, "assistant_content_changed"),
            ("<|channel>thought\n{}<turn|>", 106, "unclosed_thought_channel"),
            ("{}<|tool_response>", 50, "unexpected_terminal_token"),
        ]:
            with self.subTest(code=code), self.assertRaisesRegex(pilot.ResponseEnvelopeError, code):
                pilot.extract_assistant_content(processor, raw, prefix="", stop_token_id=stop_id)

    def test_fence_policy_preserves_raw_text_and_exact_body(self):
        body = "\t" + json.dumps(OUTPUT) + " \r\n"
        responses = (
            (body, body, False),
            ("```json\r\n" + body + "```", body, True),
            ("\r\n  ~~~~ json\t\r\n" + body + " ~~~~~\t\r\n\t\n", "\r\n" + body + "\t\n", True),
        )
        for text, expected, removed in responses:
            with self.subTest(removed=removed):
                raw = text + "<turn|>"
                record = {"raw_response": raw, "response_text": text}
                value, error = pilot.validate_attempt(record, self.validator, EVIDENCE,
                                                      pilot.FENCE_POLICY)
                self.assertEqual(value, OUTPUT)
                self.assertIsNone(error)
                self.assertEqual(record["raw_response"], raw)
                self.assertEqual(record["response_text"], text)
                self.assertEqual(record["validation_text"], expected)
                self.assertEqual(record["fence_removed"], removed)
                self.assertEqual(record["raw_strict_json_valid"], not removed)

    def test_fence_policy_rejects_malformed_json_and_extra_or_incomplete_fences(self):
        valid = json.dumps(OUTPUT)
        fenced = "```json\n" + valid + "\n```"
        for text in (
            "Explanation\n" + fenced,
            fenced + "\nTrailing prose",
            fenced + "\n" + fenced,
            "```json\n" + valid,
            "````json\n" + valid + "\n```",
            "```json\n" + valid + "\n~~~",
            "```json extra\n" + valid + "\n```",
        ):
            with self.subTest(wrapper=text):
                record = {"response_text": text}
                _, error = pilot.validate_attempt(record, self.validator, EVIDENCE,
                                                 pilot.FENCE_POLICY)
                self.assertEqual(error["category"], "invalid_json")
                self.assertFalse(record["fence_removed"])
                self.assertEqual(record["validation_text"], text)

        for body, category in (
            ('{"products": [}', "invalid_json"),
            ('{"products": [], "products": []}', "invalid_json"),
            ("\N{NO-BREAK SPACE}" + valid, "invalid_json"),
            ('{"unexpected": null}', "invalid_schema"),
        ):
            with self.subTest(category=category, body=body):
                record = {"response_text": "```json\n" + body + "\n```"}
                value, error = pilot.validate_attempt(record, self.validator, EVIDENCE,
                                                      pilot.FENCE_POLICY)
                self.assertIsNone(value)
                self.assertEqual(error["category"], category)
                self.assertTrue(record["fence_removed"])
                self.assertEqual(record["validation_text"], body + "\n")

    def test_fence_resume_rehydrates_seeded_valid_output(self):
        # All synthetic requests use one product so the same response can pass each schema.
        rows = [{**self.rows[0], "case_id": f"synthetic-fence-{i}"} for i in range(3)]
        self.input.write_text(pilot.jsonl(rows), encoding="utf-8")
        requests, _, metadata = pilot.check_input(self.input)
        ids = list(requests)
        version, normalization = pilot.NATIVE_RESPONSE_VERSION, pilot.FENCE_POLICY
        configuration = pilot.run_configuration(requests, metadata, version, normalization)
        with self.assertRaisesRegex(pilot.PilotError, "requires gemma4_native_v1"):
            pilot.run_configuration(requests, metadata, pilot.LEGACY_RESPONSE_VERSION,
                                    normalization)
        fenced = "```json\n" + json.dumps(OUTPUT) + "\n```"
        records = {}
        for case_id in ids[:2]:
            record = {"case_id": case_id, "prompt_version": pilot.PROMPT_VERSION,
                      "prompt_sha256": requests[case_id]["prompt_sha256"],
                      "raw_response": fenced + "<turn|>", "response_text": fenced,
                      "generation_seconds": None, "gpu_memory": None, "stop_reason": "eos_token"}
            _, record["error"] = pilot.validate_attempt(record, self.validator, EVIDENCE,
                                                        normalization)
            records[case_id] = record
        raw = pilot.jsonl(records.values())
        self.output.mkdir()
        raw_path = self.output / "raw_responses.PRIVATE.jsonl"
        raw_path.write_text(raw, encoding="utf-8")
        frozen = {"synthetic_artifact": {"sha256": "synthetic-hash"}}
        self.enterContext(patch.object(pilot, "frozen_stage5c_artifacts", return_value=frozen))
        manifest = {
            "configuration": configuration, "frozen_stage5c_artifacts": frozen,
            "imported_attempts": 2,
            "initial_raw_responses_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "decoder_implementation": {"sha256": pilot.file_sha256(pilot.NATIVE_DECODER_PATH)},
            "normalizer_implementation": {
                "sha256": pilot.file_sha256(pilot.FENCE_NORMALIZER_PATH),
            },
            "cpu_revalidation_runtime": pilot.package_metadata(),
        }
        (self.output / pilot.FENCE_RESUME_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        pilot.save_summary(self.output, {
            "configuration": configuration, "model_load_attempts": [],
            "offline_revalidation_manifest": pilot.FENCE_RESUME_MANIFEST,
        }, records, ids)

        generator = Mock(metadata={})
        plain = json.dumps(OUTPUT)
        generator.generate.side_effect = lambda _messages, record: record.update(
            response_text=plain, raw_response=plain + "<turn|>",
        )
        with patch.object(pilot, "GemmaGenerator", return_value=generator) as factory:
            report = pilot.run(self.input, self.output, 3, response_decoding=version,
                               response_normalization=normalization)
            factory.assert_called_once_with(version)
        self.assertEqual(generator.generate.call_count, 1)
        self.assertEqual(generator.generate.call_args.args[1]["case_id"], ids[2])
        self.assertTrue(raw_path.read_text().startswith(raw))
        self.assertEqual(report["raw_strict_json_valid_cases"], 1)
        self.assertEqual(report["normalized_json_valid_cases"], 3)
        self.assertEqual(report["normalized_schema_valid_cases"], 3)
        self.assertEqual(report["fence_removed_cases"], 2)
        parsed = self.output / "validated_outputs.PRIVATE.jsonl"
        self.assertEqual([json.loads(row)["case_id"] for row in parsed.read_text().splitlines()], ids)

    def test_native_decoding_version_cannot_resume_legacy_results(self):
        with patch.object(pilot, "GemmaGenerator", side_effect=RuntimeError) as factory:
            with self.assertRaises(RuntimeError):
                pilot.run(self.input, self.output, 2, response_decoding=pilot.LEGACY_RESPONSE_VERSION)
            with self.assertRaisesRegex(pilot.PilotError, "different pilot configuration"):
                pilot.run(self.input, self.output, 2, response_decoding=pilot.NATIVE_RESPONSE_VERSION)
            factory.assert_called_once_with(pilot.LEGACY_RESPONSE_VERSION)
            native_output = self.output.with_name("native")
            with self.assertRaises(RuntimeError):
                pilot.run(self.input, native_output, 2, response_decoding=pilot.NATIVE_RESPONSE_VERSION)
        configuration = json.loads((native_output / "aggregate.json").read_bytes())["configuration"]
        self.assertEqual(configuration["runner_format_version"], 3)
        self.assertEqual(
            configuration["decoding"]["response_decoding"], pilot.NATIVE_RESPONSE_VERSION,
        )

    def test_native_resume_revalidates_fences_and_generates_only_pending_cases(self):
        rows = [{**self.rows[0], "case_id": f"synthetic-pilot-{i}"} for i in range(12)]
        self.input.write_text(pilot.jsonl(rows), encoding="utf-8")
        requests, _, metadata = pilot.check_input(self.input)
        target_ids = list(requests)
        fenced = "```json\n" + json.dumps(OUTPUT) + "\n```"
        classification = {"category": "invalid_json", "code": "syntax", "line": 1, "column": 1}
        legacy_records = {case_id: {
            "case_id": case_id, "prompt_version": pilot.PROMPT_VERSION,
            "prompt_sha256": requests[case_id]["prompt_sha256"],
            "raw_response": fenced + "<turn|>", "response_text": fenced,
            "input_tokens": 5, "stop_token_id": 106, "stop_reason": "eos_token",
            "generation_seconds": None, "gpu_memory": None, "error": classification,
        } for case_id in target_ids[:2]}
        processor = Mock()
        processor.apply_chat_template.return_value = {
            "input_ids": Mock(device=Mock(type="cpu"), shape=(1, 5)),
        }
        processor.tokenizer.response_template = {"fields": {"content": {"content": "text"}}}
        processor.parse_response.return_value = {"role": "assistant", "content": fenced}
        seeded = native_resume.revalidate_attempts(
            processor, legacy_records, requests, self.validator,
        )
        self.assertEqual(seeded, legacy_records)
        self.assertEqual(processor.parse_response.call_count, 2)

        version = pilot.NATIVE_RESPONSE_VERSION
        configuration = pilot.run_configuration(requests, metadata, version)
        raw = pilot.jsonl(seeded.values())
        frozen_artifacts = {"synthetic_artifact": {"sha256": "synthetic-hash"}}
        self.enterContext(patch.object(pilot, "frozen_stage5c_artifacts", return_value=frozen_artifacts))
        manifest = {
            "configuration": configuration,
            "frozen_stage5c_artifacts": frozen_artifacts,
            "imported_attempts": 2,
            "initial_native_raw_responses_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "decoder_implementation": {"sha256": pilot.file_sha256(pilot.NATIVE_DECODER_PATH)},
            "cpu_revalidation_runtime": pilot.package_metadata(),
        }
        self.output.mkdir()
        raw_path = self.output / "raw_responses.PRIVATE.jsonl"
        raw_path.write_text(raw, encoding="utf-8")
        (self.output / pilot.NATIVE_RESUME_MANIFEST).write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        pilot.save_summary(self.output, {
            "configuration": configuration, "model_load_attempts": [],
            "offline_revalidation_manifest": pilot.NATIVE_RESUME_MANIFEST,
        }, seeded, target_ids)

        command = ["--input", str(self.input), "--output", str(self.output), "--limit", "12",
                   "--response-decoding", version]
        stdout = io.StringIO()
        with patch.object(pilot, "GemmaGenerator") as factory:
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(pilot.main([*command, "--check-only"]), 0)
            factory.assert_not_called()
        self.assertIn("Already recorded in selection: 2. Pending: 10.", stdout.getvalue())

        generated_ids = []

        def generate(_messages, record):
            generated_ids.append(record["case_id"])
            record.update(response_text=fenced, raw_response=fenced + "<turn|>")

        generator = Mock(metadata={})
        generator.generate.side_effect = generate
        with patch.object(pilot, "GemmaGenerator", return_value=generator) as factory:
            report = pilot.run(self.input, self.output, 12, response_decoding=version)
            factory.assert_called_once_with(version)
        self.assertEqual(generated_ids, target_ids[2:])
        self.assertTrue(raw_path.read_text().startswith(raw))
        self.assertEqual(report["attempted_cases"], 12)
        self.assertEqual(report["pending_requested_cases"], 0)
        self.assertEqual(report["failure_categories"], {"invalid_json": 12})

        raw_path.unlink()
        with patch.object(pilot, "GemmaGenerator") as factory:
            with self.assertRaisesRegex(pilot.PilotError, "missing the expected revalidated attempts"):
                pilot.run(self.input, self.output, 12, response_decoding=version)
            factory.assert_not_called()

    def test_two_case_cli_then_resume(self):
        def generate(messages, record):
            record.update(input_tokens=10, generation_seconds=0.5,
                          gpu_memory={"peak_allocated_bytes": 20, "peak_reserved_bytes": 30})
            if messages == self.rows[2]["messages"]:
                record["error"] = {"category": "generation_error", "code": "OutOfMemoryError"}
                return
            response = json.dumps(OUTPUT) if messages == self.rows[0]["messages"] else "```json"
            record.update(response_text=response, raw_response=response + "<eos>",
                          output_tokens=10, stop_token_id=1, stop_reason="eos_token")

        generator = Mock(metadata={"loaded_revision": pilot.REVISION})
        generator.generate.side_effect = generate
        argv = ["--input", str(self.input), "--output", str(self.output), "--limit"]
        raw = self.output / "raw_responses.PRIVATE.jsonl"
        validated = self.output / "validated_outputs.PRIVATE.jsonl"
        report = self.output / "aggregate.json"
        stdout = io.StringIO()
        with patch.object(pilot, "GemmaGenerator", return_value=generator) as factory:
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(pilot.main([*argv, "2"]), 0)
                first = json.loads(report.read_text())
                self.assertEqual(first["attempted_cases"], 2)
                self.assertEqual(first["completed_cases"], 2)
                self.assertEqual(first["valid_json_cases"], 1)
                self.assertEqual(factory.call_count, 1)
                raw_before = raw.read_text()
                validated.unlink()  # A crash can leave the derived file missing or stale.
                self.assertEqual(pilot.main([*argv, "2"]), 0)
                self.assertEqual(factory.call_count, 1)
                self.assertEqual(raw.read_text(), raw_before)
                self.assertEqual(json.loads(validated.read_text())["output"], OUTPUT)
                self.assertEqual(pilot.main([*argv, "3"]), 0)

        self.assertEqual(factory.call_count, 2)
        self.assertEqual([call.args[0] for call in generator.generate.call_args_list],
                         [row["messages"] for row in self.rows])
        attempts = [json.loads(line) for line in raw.read_text().splitlines()]
        self.assertEqual([row["case_id"] for row in attempts],
                         [row["case_id"] for row in self.rows])
        self.assertEqual(attempts[1]["raw_response"], "```json<eos>")
        self.assertIsNone(attempts[2]["raw_response"])
        self.assertEqual(attempts[2]["input_tokens"], 10)
        self.assertEqual(raw.stat().st_mode & 0o777, 0o600)
        self.assertEqual(validated.stat().st_mode & 0o777, 0o600)
        metadata = json.loads(report.read_text())
        self.assertEqual(metadata["configuration"]["decoding"]["response_decoding"],
                         pilot.NATIVE_RESPONSE_VERSION)
        self.assertEqual(metadata["configuration"]["response_normalization"], pilot.FENCE_POLICY)
        self.assertEqual(metadata["requested_cases"], 3)
        self.assertEqual(metadata["pending_requested_cases"], 0)
        self.assertEqual(metadata["attempted_cases"], 3)
        self.assertEqual(metadata["completed_cases"], 2)
        self.assertEqual(metadata["valid_schema_cases"], 1)
        self.assertEqual(metadata["valid_contract_cases"], 1)
        self.assertEqual(metadata["failure_categories"], {"invalid_json": 1, "generation_error": 1})
        self.assertEqual(metadata["generation_latency_seconds"]["p95"], 0.5)
        self.assertEqual(metadata["generation_gpu_memory"]["peak_reserved_bytes"], 30)
        for private in ("synthetic-case", "synthetic-source", "```json"):
            self.assertNotIn(private, report.read_text())
            self.assertNotIn(private, stdout.getvalue())

    def test_load_failure_and_resume_rejects_changed_input(self):
        err = io.StringIO()
        with patch.object(pilot, "GemmaGenerator", side_effect=RuntimeError("private error text")):
            with contextlib.redirect_stderr(err):
                result = pilot.main([
                    "--input", str(self.input), "--output", str(self.output), "--limit", "2",
                ])
                self.assertEqual(result, 1)
        report = json.loads((self.output / "aggregate.json").read_text())
        self.assertEqual(report["attempted_cases"], 0)
        self.assertEqual(report["pending_requested_cases"], 2)
        self.assertEqual(report["model_load_attempts"][0]["error"], "RuntimeError")
        self.assertNotIn("private error text", err.getvalue())
        self.assertNotIn("private error text", json.dumps(report))

        self.input.write_text(pilot.jsonl(reversed(self.rows)), encoding="utf-8")
        with patch.object(pilot, "GemmaGenerator") as factory:
            with self.assertRaisesRegex(pilot.PilotError, "different pilot configuration"):
                pilot.run(self.input, self.output, 2)
            factory.assert_not_called()

    def test_rejects_reference_in_prompt_evidence(self):
        row = self.rows[0]
        row["messages"][1]["content"] = pilot.SOURCE_MARKER + json.dumps({
            **EVIDENCE, "reference_json": "private reference text",
        })
        self.input.write_text(pilot.jsonl([row]), encoding="utf-8")
        with self.assertRaisesRegex(pilot.PilotError, "^Invalid prepared request at line 1$"):
            pilot.read_requests(self.input)


if __name__ == "__main__":
    unittest.main()
