"""Revalidate the two saved pilot attempts on CPU and seed the native run directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import stage06_original_gemma_pilot as pilot

DIAGNOSTIC = pilot.REPO / "results/stage06/original_gemma_json_diagnostic.v1.json"


def revalidate_attempts(processor, records, requests, validator):
    revalidated = {}
    for case_id, saved in records.items():
        request = requests[case_id]
        inputs = pilot.tokenize_request(processor, request["messages"])
        prefix = inputs["input_ids"]
        if prefix.device.type != "cpu" or prefix.shape[-1] != saved["input_tokens"]:
            raise pilot.PilotError("CPU prompt tokenization differs from the saved attempt")
        content = pilot.extract_assistant_content(
            processor, saved["raw_response"], prefix=prefix, stop_token_id=saved["stop_token_id"],
        )
        _, error = pilot.validate_response(content, validator, request["evidence"])
        revalidated[case_id] = {**saved, "response_text": content, "error": error}
    return revalidated


def prepare_resume():
    from jsonschema import Draft202012Validator

    output = pilot.NATIVE_OUTPUT
    if output.exists():
        raise pilot.PilotError("The native run directory already exists")

    requests, schema, input_metadata = pilot.check_input(pilot.INPUT)
    legacy_config = pilot.run_configuration(requests, input_metadata, pilot.LEGACY_RESPONSE_VERSION)
    native_config = pilot.run_configuration(requests, input_metadata, pilot.NATIVE_RESPONSE_VERSION)
    diagnostic = pilot.strict_json(DIAGNOSTIC.read_bytes())
    source_files = {
        "aggregate_sha256": pilot.OUTPUT / "aggregate.json",
        "raw_responses_sha256": pilot.OUTPUT / "raw_responses.PRIVATE.jsonl",
    }
    source_hashes = {name: pilot.file_sha256(path) for name, path in source_files.items()}
    if any(diagnostic["original_run"][name] != digest for name, digest in source_hashes.items()):
        raise pilot.PilotError("Saved legacy results differ from the verified diagnostic inputs")

    report, records = pilot.read_run_state(pilot.OUTPUT, requests, legacy_config)
    if len(requests) != 12 or list(records) != list(requests)[:2]:
        raise pilot.PilotError("Expected the two recorded attempts at the start of the 12-case pilot")

    frozen_artifacts = pilot.frozen_stage5c_artifacts()

    cpu_runtime = pilot.package_metadata()
    sessions = report["model_load_attempts"]
    for saved in records.values():
        original_runtime = sessions[saved["model_load_attempt"] - 1]
        if any(cpu_runtime["package_versions"][name] != original_runtime["package_versions"][name]
               for name in pilot.VALIDATION_PACKAGES):
            raise pilot.PilotError("CPU processor or validation package differs from the legacy run")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(pilot.MODEL_ID, **pilot.checkpoint_options())
    revalidated = revalidate_attempts(processor, records, requests, Draft202012Validator(schema))
    responses = []
    previous = diagnostic["responses"]
    for ordinal, (case_id, record) in enumerate(revalidated.items(), 1):
        reference = previous[ordinal - 1]
        if (
            reference["response_ordinal"] != ordinal
            or reference["failure_cause"] != "model_generated_markdown_json_fence"
            or record["response_text"] != records[case_id]["response_text"]
            or record["error"] != reference["result_after_native_extraction"]
            or record["error"]["category"] != "invalid_json"
        ):
            raise pilot.PilotError("Native revalidation differs from the saved fence diagnosis")
        responses.append({"response_ordinal": ordinal, "result": record["error"]})

    if any(source_hashes[name] != pilot.file_sha256(path) for name, path in source_files.items()):
        raise pilot.PilotError("Legacy results changed during CPU revalidation")

    native_records = pilot.jsonl(revalidated.values())
    manifest = {
        "manifest_version": 1,
        "configuration": native_config,
        "decoder_implementation": {
            "path": pilot.NATIVE_DECODER_PATH.relative_to(pilot.REPO).as_posix(),
            "sha256": pilot.file_sha256(pilot.NATIVE_DECODER_PATH),
        },
        "cpu_revalidation_runtime": cpu_runtime,
        "frozen_stage5c_artifacts": frozen_artifacts,
        "source_legacy_run": {
            "directory": str(pilot.OUTPUT),
            "response_decoding": pilot.LEGACY_RESPONSE_VERSION,
            **source_hashes,
            "diagnostic_path": DIAGNOSTIC.relative_to(pilot.REPO).as_posix(),
            "diagnostic_sha256": pilot.file_sha256(DIAGNOSTIC),
        },
        "native_run_directory": str(output),
        "imported_attempts": len(revalidated),
        "initial_pending_cases": len(pilot.pending_cases(requests, revalidated)),
        "initial_native_raw_responses_sha256": hashlib.sha256(native_records.encode()).hexdigest(),
        "offline_results": responses,
    }
    native_report = {"configuration": native_config, "model_load_attempts": sessions,
                     "offline_revalidation_manifest": pilot.NATIVE_RESUME_MANIFEST}
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".stage06_native_resume.") as folder:
        staging = Path(folder)
        pilot.atomic_write(staging / "raw_responses.PRIVATE.jsonl", native_records)
        pilot.atomic_write(staging / "validated_outputs.PRIVATE.jsonl", "")
        pilot.save_summary(staging, native_report, revalidated, list(requests))
        pilot.atomic_write(staging / pilot.NATIVE_RESUME_MANIFEST,
                           json.dumps(manifest, indent=2) + "\n")
        staging.rename(output)
    return manifest


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    try:
        manifest = prepare_resume()
    except pilot.PilotError as exc:
        print(f"Native resume preparation failed. {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Native resume preparation failed. {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Revalidated {manifest['imported_attempts']} saved attempts on CPU. "
          f"Pending cases: {manifest['initial_pending_cases']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
