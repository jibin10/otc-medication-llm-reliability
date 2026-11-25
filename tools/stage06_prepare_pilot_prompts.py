"""Create the schema-corrected Stage 6 prompts and metadata manifest on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys

from part05_prepare_pilot import sha256_file
from stage06_original_gemma_pilot import (
    BENCHMARK_CONTRACT_VERSION,
    INPUT,
    LEGACY_PROMPT_VERSION,
    MANIFEST,
    PROMPT_VERSION,
    REPO,
    SCHEMA,
    SCORING_CONTRACT_VERSION,
    SOURCE_MARKER,
    PilotError,
    atomic_write,
    canonical,
    load_schema,
    output_instructions,
    read_requests,
    strict_json,
)

SOURCE = REPO / "results/part05/validation_pilot_prompts.PRIVATE.jsonl"
PREPARATION = SOURCE.with_name("validation_pilot_preparation.json")
VALIDATION = REPO / "data/processed/final_v5/validation.csv"
CASE_COUNT = 12


def corrected_rows(requests, schema):
    instructions = output_instructions(schema)
    for case_id, request in requests.items():
        system, user = request["messages"]
        _, marker, source = user["content"].partition(SOURCE_MARKER)
        yield {
            "case_id": case_id,
            "prompt_version": PROMPT_VERSION,
            "messages": [system, {**user, "content": instructions + marker + source}],
        }


def selected_family_metadata(path, case_ids):
    selected = set(case_ids)
    families = {}
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.reader(stream)
        header = next(rows)
        case_column = header.index("case_id")
        partition_column = header.index("partition")
        family_column = header.index("family_group_ids_json")
        for row in rows:
            case_id = row[case_column]
            if case_id not in selected:
                continue
            if row[partition_column] != "validation" or case_id in families:
                raise PilotError("Pilot case is duplicated or outside the validation partition")
            ids = strict_json(row[family_column])
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise PilotError("Unexpected validation family metadata")
            families[case_id] = ids
    if families.keys() != selected:
        raise PilotError("Pilot case is missing from validation family metadata")
    ordered_assignments = [[case_id, families[case_id]] for case_id in case_ids]
    return {
        "unique_family_group_ids": len({value for ids in families.values() for value in ids}),
        "ordered_case_family_assignments_sha256": hashlib.sha256(
            canonical(ordered_assignments).encode("utf-8")
        ).hexdigest(),
    }


def prepare():
    requests, original_hash = read_requests(SOURCE, legacy=True)
    preparation_data = PREPARATION.read_bytes()
    summary = strict_json(preparation_data)
    if (
        len(requests) != CASE_COUNT
        or summary["selected_cases"] != CASE_COUNT
        or summary["prompt_version"] != LEGACY_PROMPT_VERSION
        or summary["benchmark_version"] != "final_v5"
        or summary["prompt_file_sha256"] != original_hash
        or summary["prompt_message_hashes"] != [row["prompt_sha256"] for row in requests.values()]
        or summary["references_loaded_into_prompts"] is not False
        or summary["retrieval_predictions_loaded"] is not False
        or summary["sealed_test_opened"] is not False
    ):
        raise PilotError("Original prompts differ from the Part 5 validation preparation record")

    validation_hash = sha256_file(VALIDATION)
    if validation_hash != summary["validation_csv_sha256"]:
        raise PilotError("Validation source differs from the Part 5 preparation record")
    families = selected_family_metadata(VALIDATION, requests)
    if families["unique_family_group_ids"] != summary["unique_family_group_ids"]:
        raise PilotError("Selected validation family count differs from the Part 5 preparation record")

    schema, schema_hash = load_schema()
    rows = list(corrected_rows(requests, schema))
    corrected = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    corrected_hash = hashlib.sha256(corrected.encode("utf-8")).hexdigest()
    sources = [row["messages"][1]["content"].split(SOURCE_MARKER, 1)[1] for row in rows]
    artifacts = {
        "original_prompts": (SOURCE, original_hash),
        "corrected_prompts": (INPUT, corrected_hash),
        "original_preparation_summary": (PREPARATION, hashlib.sha256(preparation_data).hexdigest()),
        "validation_family_metadata_source": (VALIDATION, validation_hash),
        "output_schema": (SCHEMA, schema_hash),
    }
    for key, relative_path in {
        "frozen_benchmark_manifest": "results/challenge_v1/freeze_manifest.json",
        "evidence_manifest": "results/challenge_v1/evidence_manifest.json",
        "scoring_specifications": "results/challenge_v1/scoring_specifications.json",
        "scoring_amendment_manifest": (
            "results/challenge_v1_scoring_amendment/scoring_amendment_manifest.json"
        ),
        "output_parser": "src/hrp/challenge_evaluation.py",
        "scoring_implementation": "src/hrp/challenge_scoring_v3.py",
    }.items():
        path = REPO / relative_path
        artifacts[key] = (path, sha256_file(path))
    manifest = {
        "manifest_version": 1,
        "original_prompt_version": LEGACY_PROMPT_VERSION,
        "corrected_prompt_version": PROMPT_VERSION,
        "benchmark_version": summary["benchmark_version"],
        "benchmark_contract_version": BENCHMARK_CONTRACT_VERSION,
        "scoring_contract_version": SCORING_CONTRACT_VERSION,
        "selected_cases": CASE_COUNT,
        **families,
        "ordered_evidence_blocks_sha256": hashlib.sha256(
            canonical(sources).encode("utf-8")
        ).hexdigest(),
        "correction": (
            "Replace the user output instructions and template with Stage 5C structural and "
            "semantic requirements, including the complete frozen schema and formulation field. "
            "Retain case order, case IDs, system messages and exact prepared source blocks. "
            "Family assignments stay linked through the original validation case IDs."
        ),
        "evidence_scope": "Original Part 5 pilot evidence and truncation limits.",
        "artifacts": {key: {"path": path.relative_to(REPO).as_posix(), "sha256": digest}
                      for key, (path, digest) in artifacts.items()},
    }
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(INPUT, corrected)
    atomic_write(MANIFEST, json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        manifest = prepare()
    except PilotError as exc:
        print(f"Prompt correction failed. {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Prompt correction failed. {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Prepared {manifest['selected_cases']} {PROMPT_VERSION} prompts and a metadata manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
