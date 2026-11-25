"""Seed the fence-normalized pilot from the verified CPU native-extraction ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import stage06_original_gemma_pilot as pilot


def prepare_resume():
    from jsonschema import Draft202012Validator

    output = pilot.FENCE_OUTPUT
    if output.exists():
        raise pilot.PilotError("The fence-normalized run directory already exists")

    requests, schema, input_metadata = pilot.check_input(pilot.INPUT)
    target_ids = list(requests)
    native_config = pilot.run_configuration(requests, input_metadata, pilot.NATIVE_RESPONSE_VERSION)
    configuration = pilot.run_configuration(requests, input_metadata, pilot.NATIVE_RESPONSE_VERSION,
                                            pilot.FENCE_POLICY)
    source = pilot.NATIVE_OUTPUT
    source_manifest = source / pilot.NATIVE_RESUME_MANIFEST
    source_files = {name: source / name for name in (
        "aggregate.json", "raw_responses.PRIVATE.jsonl", pilot.NATIVE_RESUME_MANIFEST,
    )}
    source_hashes = {name: pilot.file_sha256(path) for name, path in source_files.items()}
    previous_report, saved = pilot.read_run_state(source, requests, native_config)
    if (
        len(requests) != 12 or list(saved) != target_ids[:2]
        or previous_report.get("offline_revalidation_manifest") != source_manifest.name
        or previous_report["failure_categories"] != {"invalid_json": 2}
    ):
        raise pilot.PilotError("Expected the two verified native attempts at the start of the pilot")
    native_manifest = pilot.strict_json(source_manifest.read_bytes())
    validator = Draft202012Validator(schema)

    records, parsed, results = {}, [], []
    for ordinal, (case_id, original) in enumerate(saved.items(), 1):
        # The verified native ledger already contains byte-preserved assistant text
        # extracted offline from the original generation with the pinned processor.
        record = dict(original)
        value, error = pilot.validate_attempt(record, validator, requests[case_id]["evidence"],
                                              pilot.FENCE_POLICY)
        record["error"] = error
        records[case_id] = record
        if error is None:
            parsed.append({"case_id": case_id, "output": value})
        category = error["category"] if error else None
        results.append({
            "response_ordinal": ordinal,
            "raw_strict_json_valid": record["raw_strict_json_valid"],
            "fence_removed": record["fence_removed"],
            "normalized_json_valid": category != "invalid_json",
            "normalized_schema_valid": category not in {"invalid_json", "invalid_schema"},
            "normalized_contract_valid": error is None,
            "normalized_failure_category": category,
        })

    if any(pilot.file_sha256(path) != source_hashes[name] for name, path in source_files.items()):
        raise pilot.PilotError("The strict native ledger changed during offline revalidation")

    raw = pilot.jsonl(records.values())
    ongoing_report = {"configuration": configuration,
                      "model_load_attempts": previous_report["model_load_attempts"],
                      "offline_revalidation_manifest": pilot.FENCE_RESUME_MANIFEST}
    manifest = {
        "manifest_version": 1,
        "configuration": configuration,
        "decoder_implementation": native_manifest["decoder_implementation"],
        "normalizer_implementation": {
            "path": pilot.FENCE_NORMALIZER_PATH.relative_to(pilot.REPO).as_posix(),
            "sha256": pilot.file_sha256(pilot.FENCE_NORMALIZER_PATH),
        },
        "cpu_revalidation_runtime": pilot.package_metadata(),
        "frozen_stage5c_artifacts": native_manifest["frozen_stage5c_artifacts"],
        "source_strict_native_run": {"directory": str(source), "sha256_by_file": source_hashes},
        "source_legacy_run": native_manifest["source_legacy_run"],
        "imported_attempts": len(records),
        "initial_pending_cases": len(pilot.pending_cases(target_ids, records)),
        "initial_raw_responses_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "offline_results": results,
    }
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".stage06_fence_resume.") as folder:
        staging = Path(folder)
        pilot.atomic_write(staging / "raw_responses.PRIVATE.jsonl", raw)
        pilot.atomic_write(staging / "validated_outputs.PRIVATE.jsonl", pilot.jsonl(parsed))
        pilot.save_summary(staging, ongoing_report, records, target_ids)
        pilot.atomic_write(staging / pilot.FENCE_RESUME_MANIFEST, json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return ongoing_report


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        report = prepare_resume()
    except pilot.PilotError as exc:
        print(f"Fence resume preparation failed. {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Fence resume preparation failed. {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Revalidated {report['attempted_cases']} saved attempts on CPU. "
          f"Pending cases: {report['pending_requested_cases']}.")
    print(f"Raw strict JSON {report['raw_strict_json_valid_cases']}, "
          f"normalized JSON {report['normalized_json_valid_cases']}, "
          f"normalized schema {report['normalized_schema_valid_cases']}, "
          f"fences removed {report['fence_removed_cases']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
