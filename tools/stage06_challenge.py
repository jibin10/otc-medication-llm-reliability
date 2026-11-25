"""Prepare, run, or score the frozen challenge_v1 validation evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import stage06_original_gemma_pilot as pilot
from hrp import challenge_evaluation as contract
from hrp.challenge_scoring_v3 import score_predictions
from jsonschema import Draft202012Validator

FROZEN = pilot.REPO / "results/challenge_v1"
AMENDMENT = pilot.REPO / "results/challenge_v1_scoring_amendment/scoring_amendment_manifest.json"
DEFAULT_OUTPUT = pilot.OUTPUT.with_name("stage06_challenge_v1_original_gemma4")
BENCHMARK = "benchmark.PRIVATE.jsonl"
EVIDENCE = "common_evidence.PRIVATE.jsonl"
IDENTITIES = "ingredient_identity_mapping.PRIVATE.jsonl"
ORDER = "inference_order.PRIVATE.jsonl"
ROLES = {"different_without_ndc": 34, "different_with_ndc": 86, "same_ingredient_control": 80,
         "ambiguity": 22, "supplementary": 8}
ELIGIBILITY_COUNTS = ("cases", "product_requests", "ingredient_target_requests",
                      "supported_comparison_cases", "abstention_target_requests",
                      "unique_ndc_target_requests", "full_formulation_target_requests")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def public_hash_matches(data, expected):
    # The frozen JSON manifests were written on Windows, then checked out with LF.
    if sha(data) == expected:
        return True
    lf = data.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return b"\r" not in lf and data in (lf, crlf) and expected in {sha(lf), sha(crlf)}


def frozen_manifest():
    amendment_data = AMENDMENT.read_bytes()
    amendment = pilot.strict_json(amendment_data)
    for name, path in (
        ("frozen_benchmark_manifest", FROZEN / "freeze_manifest.json"),
        ("original_scorer", pilot.REPO / "src/hrp/challenge_evaluation.py"),
        ("corrected_scorer", pilot.REPO / "src/hrp/challenge_scoring_v3.py"),
    ):
        if not public_hash_matches(path.read_bytes(), amendment[name]["sha256"]):
            raise pilot.PilotError("A frozen challenge or scorer hash differs from the amendment")
    freeze = pilot.strict_json((FROZEN / "freeze_manifest.json").read_bytes())
    if (freeze["benchmark"] != "challenge_v1" or freeze["status"] != "FROZEN"
            or freeze["contract_version"] != contract.VERSION
            or amendment["scoring_contract_version"] != pilot.SCORING_CONTRACT_VERSION):
        raise pilot.PilotError("Unexpected frozen benchmark or scoring contract")
    return freeze, {"freeze_manifest_sha256": amendment["frozen_benchmark_manifest"]["sha256"],
                    "scoring_amendment_sha256": sha(amendment_data)}


def artifact(directory, filename, freeze, *, public=False):
    path = directory / filename
    if any("sealed_test" in part.casefold() for part in path.resolve().parts):
        raise pilot.PilotError("A challenge artifact resolves to a sealed-test path")
    if not path.is_file():
        raise pilot.PilotError(f"Missing frozen challenge artifact {filename}")
    data = path.read_bytes()
    expected = freeze["outputs"][filename]
    if not (public_hash_matches(data, expected) if public else sha(data) == expected):
        raise pilot.PilotError(f"SHA-256 mismatch for frozen challenge artifact {filename}")
    return data


def unique_rows(data):
    rows = {}
    for line in data.splitlines():
        row = pilot.strict_json(line)
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in rows:
            raise pilot.PilotError("Invalid or duplicated challenge case ID")
        rows[case_id] = row
    return rows


def inference_requests(directory, freeze):
    requests = {}
    for case_id, row in unique_rows(artifact(directory, EVIDENCE, freeze)).items():
        if row.keys() != {"case_id", "payload_sha256", "payload", "messages"}:
            raise pilot.PilotError("Unexpected frozen common-evidence structure")
        payload, messages = row["payload"], row["messages"]
        payload_hash = sha(contract.canonical(payload).encode("utf-8"))
        if payload_hash != row["payload_sha256"] or messages != contract.build_messages(payload):
            raise pilot.PilotError("Frozen messages or evidence disagree with the source-only contract")
        requests[case_id] = {"messages": messages, "evidence": payload, "prompt_version": contract.VERSION,
                             "payload_sha256": payload_hash,
                             "prompt_sha256": pilot.prompt_sha256(messages)}
    if len(requests) != sum(ROLES.values()):
        raise pilot.PilotError("Unexpected frozen common-evidence size")
    return requests


def evaluation_inputs(directory, freeze, requests):
    cases = unique_rows(artifact(directory, BENCHMARK, freeze))
    if cases.keys() != requests.keys() or Counter(r["scoring_role"] for r in cases.values()) != ROLES:
        raise pilot.PilotError("Unexpected frozen challenge membership or role counts")
    summary = pilot.strict_json(artifact(FROZEN, "summary.json", freeze, public=True))["selected"]
    if (len({group for r in cases.values() for group in r["families"]}) != summary["groups"]
            or Counter(r["category"] for r in cases.values()) != summary["categories"]):
        raise pilot.PilotError("Unexpected frozen challenge groups or categories")
    for case_id, case in cases.items():
        request = requests[case_id]
        if (case["partition"] != "validation" or case["source_reviewed"] is not True
                or type(case["independently_clinically_verified"]) is not bool
                or case["common_evidence_sha256"] != request["payload_sha256"]
                or len(case["reference"]["products"]) != len(request["evidence"]["products"])):
            raise pilot.PilotError("Frozen validation labels do not align with the evidence ledger")
    entries = [pilot.strict_json(line) for line in artifact(directory, IDENTITIES, freeze).splitlines()]
    return cases, contract.IdentityResolver(entries)


def inference_order(cases):
    priority = {role: i for i, role in enumerate(ROLES)}
    return sorted(cases, key=lambda case_id: (priority[cases[case_id]["scoring_role"]], case_id))


def prepare(directory, output, freeze, requests):
    cases, _ = evaluation_inputs(directory, freeze, requests)
    order = pilot.jsonl({"case_id": case_id} for case_id in inference_order(cases))
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = output / ORDER
    if path.exists() and path.read_bytes() != order.encode():
        raise pilot.PilotError("Existing inference order differs from the frozen validation roster")
    if not path.exists():
        pilot.atomic_write(path, order)


def configure_run(output, requests, freeze, provenance, schema_hash):
    path = output / ORDER
    if not path.is_file():
        raise pilot.PilotError("Prepare the private inference order before running the challenge")
    data = path.read_bytes()
    order = unique_rows(data)
    if order.keys() != requests.keys() or any(row.keys() != {"case_id"} for row in order.values()):
        raise pilot.PilotError("Inference order does not match frozen evidence membership")
    requests = {case_id: requests[case_id] for case_id in order}
    configuration = pilot.run_configuration(requests, {
        **provenance, "dataset": "challenge_v1", "input_sha256": freeze["outputs"][EVIDENCE],
        "output_schema_sha256": schema_hash, "inference_order_sha256": sha(data),
        "parser_version": [pilot.NATIVE_RESPONSE_VERSION, pilot.FENCE_POLICY],
        "native_decoder_sha256": pilot.file_sha256(pilot.NATIVE_DECODER_PATH),
        "fence_normalizer_sha256": pilot.file_sha256(pilot.FENCE_NORMALIZER_PATH),
    }, *pilot.response_policy(None, None))
    configuration["prompt_version"] = contract.VERSION
    configuration["checkpoint_kind"] = "original"
    return requests, configuration


def saved_predictions(output, requests, configuration, validator):
    run_report, records = pilot.read_run_state(output, requests, configuration)
    if records.keys() != requests.keys():
        raise pilot.PilotError("Complete all 230 inference attempts before scoring")
    values = {}
    for case_id, record in records.items():
        if record["response_text"] is None:
            if record["error"] is None:
                raise pilot.PilotError("Unexplained missing response in the attempt ledger")
            continue
        value, error = pilot.validate_attempt(dict(record), validator,
                                              requests[case_id]["evidence"], pilot.FENCE_POLICY)
        if error != record["error"]:
            raise pilot.PilotError("Recorded validation disagrees with the frozen parser")
        if error is None:
            values[case_id] = value
    parsed = unique_rows((output / "validated_outputs.PRIVATE.jsonl").read_bytes())
    if values != {case_id: row["output"] for case_id, row in parsed.items()}:
        raise pilot.PilotError("Validated predictions differ from the attempt ledger")
    return values, run_report


def case_findings(case, prediction, payload, resolver):
    if prediction is None:
        return [{"kind": "missing_or_invalid_prediction"}]
    findings = []
    for side, (target, actual, item) in enumerate(zip(case["reference"]["products"],
            prediction["products"], payload["products"], strict=True), 1):
        def note(kind, **details):
            findings.append({"product_side": side, "kind": kind, **details})

        if target["status"] in contract.SUPPORTED and actual["status"] not in contract.SUPPORTED:
            note("inappropriate_abstention", status=actual["status"])
        if target["status"] not in contract.SUPPORTED and actual["status"] in contract.SUPPORTED:
            note("inappropriate_resolution", status=actual["status"])
        source_names = [ingredient["name"] for c in item["candidates"]
                        for ingredient in c["fda_active_ingredients"]]
        source_ids = resolver.map_output(source_names)[0]
        unsupported = [name for name in actual["active_ingredient_names"]
                       if resolver.map_output([name])[0] - source_ids]
        if unsupported:
            note("unsupported_ingredients", names=unsupported)
        if actual["formulation"]:
            unsupported_forms = [i["name"] for i in actual["formulation"]["active_ingredients"]
                                 if resolver.map_output([i["name"]])[0] - source_ids]
            if unsupported_forms:
                note("unsupported_formulation_ingredients", names=unsupported_forms)
        if target["formulation"] is not None:
            actual_key = contract._form_key(actual["formulation"], resolver)
            if actual_key != contract._form_key(target["formulation"], resolver):
                established = {contract.assess_candidate(c, resolver)["formulation_key"]
                               for c in item["candidates"]
                               if contract.assess_candidate(c, resolver)["formulation_supported"]}
                basis = ("missing" if actual["formulation"] is None else
                         "matches_other_source_candidate" if actual_key in established else
                         "unsupported_by_established_source_formulations" if established else
                         "source_formulation_unresolved")
                note("formulation_error", evidence_relation=basis)
    return findings


def family_count(cases):
    return len({family for row in cases for family in row["families"]})


def summarize(cases, predictions, payloads, resolver):
    metrics = score_predictions(cases, predictions, payloads, resolver)
    del metrics["format_errors"]
    metrics["overall"]["family_groups"] = family_count(cases)
    for key, field in (("by_role", "scoring_role"), ("by_category", "category")):
        for name, slice_ in metrics[key].items():
            slice_["family_groups"] = family_count(r for r in cases if r[field] == name)
    return metrics


def summarize_provenance(cases, predictions, payloads, resolver):
    groups = {}
    for name, verified in (("source_consistency_only", False), ("independently_audited", True)):
        selected = [r for r in cases if r["independently_clinically_verified"] is verified]
        subset = {r["case_id"]: predictions[r["case_id"]]
                  for r in selected if r["case_id"] in predictions}
        groups[name] = summarize(selected, subset, payloads, resolver)
    return groups


def score(directory, output, freeze, requests, configuration, schema):
    cases, resolver = evaluation_inputs(directory, freeze, requests)
    if list(requests) != inference_order(cases):
        raise pilot.PilotError("Saved inference order differs from the frozen role ordering")
    predictions, run_report = saved_predictions(
        output, requests, configuration, Draft202012Validator(schema),
    )
    payloads = {case_id: r["evidence"] for case_id, r in requests.items()}
    baseline = {case_id: contract.predict_common_evidence(payloads[case_id], resolver)
                for case_id in cases}
    baseline_rows = "".join(contract.canonical({"case_id": case_id, "prediction": pred}) + "\n"
                            for case_id, pred in baseline.items())
    baseline_hash = sha(baseline_rows.encode("utf-8"))
    if baseline_hash != freeze["outputs"]["deterministic_predictions.PRIVATE.jsonl"]:
        raise pilot.PilotError("Replayed retrieval comparator differs from the frozen prediction hash")

    specifications = pilot.strict_json(artifact(FROZEN, "scoring_specifications.json", freeze,
                                                 public=True))
    roster = list(cases.values())
    report = {"configuration": configuration, "primary_role": "different_without_ndc",
              "supplementary_role": "supplementary", "retrieval_replay_sha256": baseline_hash,
              "private_input_sha256": {name: freeze["outputs"][name]
                                       for name in (BENCHMARK, EVIDENCE, IDENTITIES)},
              "prediction_ledger_sha256": pilot.file_sha256(output / "raw_responses.PRIVATE.jsonl"),
              "inference_aggregate_sha256": pilot.file_sha256(output / "aggregate.json"),
              "model_load_attempts": run_report["model_load_attempts"],
              "models": {}}
    details = {case_id: {"case_id": case_id, "family_groups": row["families"],
                         "role": row["scoring_role"], "category": row["category"], "models": {}}
               for case_id, row in cases.items()}
    for name, values in (("original_gemma4", predictions), ("deterministic_retrieval", baseline)):
        metrics = summarize(roster, values, payloads, resolver)
        for slice_, expected in [(metrics["overall"], specifications["fixed_denominators"]), *[
            (slice_, specifications["fixed_denominators_by_role"][role])
            for role, slice_ in metrics["by_role"].items()
        ]]:
            if any(slice_["counts"].get(k, 0) != expected.get(k, 0) for k in ELIGIBILITY_COUNTS):
                raise pilot.PilotError("Evaluation denominator differs from frozen scoring specifications")
        metrics["by_reference_provenance"] = summarize_provenance(roster, values, payloads, resolver)
        core = [row for row in roster if row["scoring_role"] != "supplementary"]
        core_values = {row["case_id"]: values[row["case_id"]] for row in core
                       if row["case_id"] in values}
        metrics["core_excluding_supplementary"] = summarize(core, core_values, payloads, resolver)
        counts = Counter()
        for case_id, row in cases.items():
            value = values.get(case_id)
            findings = case_findings(row, value, payloads[case_id], resolver)
            counts.update(finding["kind"] for finding in findings)
            per_case = score_predictions([row], {case_id: value} if value else {}, payloads, resolver)
            details[case_id]["models"][name] = {"counts": per_case["overall"]["counts"],
                                                "findings": findings}
        metrics["finding_counts"] = dict(counts)
        report["models"][name] = metrics
    pilot.atomic_write(output / "deterministic_predictions.PRIVATE.jsonl", baseline_rows)
    pilot.atomic_write(output / "case_diagnostics.PRIVATE.jsonl", pilot.jsonl(details.values()))
    pilot.atomic_write(output / "factual_validation_metrics.json", json.dumps(report, indent=2) + "\n")


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("prepare", "infer", "score"))
    cli.add_argument("--challenge-dir", type=Path, default=FROZEN)
    cli.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    cli.add_argument("--limit", type=int, default=230, help="Inference prefix size")
    args = cli.parse_args(argv)
    if not 1 <= args.limit <= 230:
        cli.error("--limit must be between 1 and 230")
    try:
        if args.output.resolve() in {FROZEN.resolve(), args.challenge_dir.resolve()}:
            raise pilot.PilotError("Use a separate run directory for challenge results")
        freeze, provenance = frozen_manifest()
        if args.mode == "prepare":
            missing = [n for n in (BENCHMARK, EVIDENCE, IDENTITIES)
                       if not (args.challenge_dir / n).is_file()]
            if missing:
                raise pilot.PilotError("Missing private exports " + ", ".join(missing))
        requests = inference_requests(args.challenge_dir, freeze)
        schema, schema_hash = pilot.load_schema(FROZEN / "output_schema.json")
        artifact(FROZEN, "output_schema.json", freeze, public=True)
        if args.mode == "prepare":
            prepare(args.challenge_dir, args.output, freeze, requests)
            print("Verified and prepared the 230-case challenge inference order on CPU.")
            return 0
        requests, configuration = configure_run(args.output, requests, freeze, provenance, schema_hash)
        if args.mode == "infer":
            report = pilot.run_prepared(args.output, requests, schema, configuration, args.limit)
            print(f"Recorded attempts {report['attempted_cases']} of 230.")
        else:
            score(args.challenge_dir, args.output, freeze, requests, configuration, schema)
            print("Saved paired challenge metrics and private case diagnostics.")
    except pilot.PilotError as exc:
        print(f"Challenge {args.mode} failed. {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Challenge {args.mode} failed. {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
