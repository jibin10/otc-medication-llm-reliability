"""Prepare original Gemma 4 validation pilot prompts without reading the sealed test set.

From the repository root:
    python tools/part05_prepare_pilot.py --repo . --limit 12

Requires the Part 5 input-contract preflight and existing final_v5 files.
Writes private prompt JSONL and a metadata-only report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def select_cases(rows: list[dict], limit: int, seed: int) -> list[dict]:
    """Stratify by supplied task/case_type and favor distinct validation families.

    This uses no references, baseline predictions, or outcome labels.
    """
    if limit < 1:
        raise ValueError("Pilot size must be positive")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["partition"] != "validation":
            raise ValueError("Part 5 pilot is validation-only")
        groups[(row["task"], row["case_type"])].append(row)
    randomizer = random.Random(seed)
    for candidates in groups.values():
        candidates.sort(key=lambda row: row["case_id"])
        randomizer.shuffle(candidates)
    chosen: list[dict] = []
    used_ids: set[str] = set()
    used_families: set[str] = set()
    strata = sorted(groups, key=lambda key: (-len(groups[key]), key))
    while len(chosen) < min(limit, len(rows)):
        progress = False
        for stratum in strata:
            available = [r for r in groups[stratum] if r["case_id"] not in used_ids]
            if not available:
                continue
            independent = [
                row for row in available
                if not set(json.loads(row["family_group_ids_json"])) & used_families
            ]
            row = (independent or available)[0]
            chosen.append(row)
            used_ids.add(row["case_id"])
            used_families.update(json.loads(row["family_group_ids_json"]))
            progress = True
            if len(chosen) >= limit:
                break
        if not progress:
            break
    return chosen


def run(repo: Path, limit: int, seed: int, max_candidates: int) -> None:
    repo = repo.resolve()
    sys.path.insert(0, str(repo / "src"))
    from hrp import FINAL_BENCHMARK_VERSION
    from hrp.evidence_adapter import (
        PROMPT_VERSION, build_evidence_payload, build_messages, prompt_sha256
    )
    from hrp.retrieval_baseline import load_catalog, retrieve_evidence

    if FINAL_BENCHMARK_VERSION != "final_v5":
        raise RuntimeError("Expected final_v5 and will not silently switch benchmark versions")
    preflight_file = repo / "results/part05/input_contract_preflight.json"
    preflight = json.loads(preflight_file.read_text(encoding="utf-8"))
    if preflight["benchmark_version"] != "final_v5" or preflight["test_set_opened"]:
        raise RuntimeError("Invalid validation preflight or test-access flag")
    validation_file = repo / "data/processed/final_v5/validation.csv"
    if sha256_file(validation_file) != preflight["validation"]["sha256"]:
        raise RuntimeError("Validation CSV changed since the completed Part 5 preflight")
    with validation_file.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected_columns = {
            "case_id", "partition", "task", "case_type", "input_json", "reference_json",
            "family_group_ids_json", "provenance_json",
        }
        if set(reader.fieldnames or []) != expected_columns:
            raise RuntimeError("Unexpected validation schema")
        rows = [
            {
                "case_id": row["case_id"],
                "partition": row["partition"],
                "task": row["task"],
                "case_type": row["case_type"],
                "input_json": row["input_json"],
                "family_group_ids_json": row["family_group_ids_json"],
            }
            for row in reader
        ]
    if len(rows) != preflight["validation"]["row_count"]:
        raise RuntimeError("Validation count changed since preflight")
    if len({row["case_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate case_id in validation")
    selected = select_cases(rows, limit=limit, seed=seed)
    if not selected:
        raise RuntimeError("No eligible validation cases")

    # Full FDA/RxNorm source catalog is loaded once and queried only by input_json.
    catalog = load_catalog(repo)
    output_dir = repo / "results/part05"
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = output_dir / "validation_pilot_prompts.PRIVATE.jsonl"
    summary_path = output_dir / "validation_pilot_preparation.json"
    source_hashes = catalog.get("source_hashes", {})
    counts = Counter()
    candidates = Counter()
    reason_counts = Counter()
    family_sets = set()
    hashes = []
    with prompt_path.open("w", encoding="utf-8") as output:
        for row in selected:
            input_json = row["input_json"]
            evidence = retrieve_evidence(input_json, catalog)
            payload = build_evidence_payload(
                input_json, evidence, max_candidates=max_candidates
            )
            messages = build_messages(payload)
            hashes.append(prompt_sha256(messages))
            output.write(
                json.dumps(
                    {"case_id": row["case_id"], "prompt_version": PROMPT_VERSION, "messages": messages},
                    ensure_ascii=False,
                ) + "\n"
            )
            counts[(row["task"], row["case_type"])] += 1
            family_sets.update(json.loads(row["family_group_ids_json"]))
            for item in payload["products"]:
                total = item["candidate_count_full_catalog"]
                candidates["none" if total == 0 else "one" if total == 1 else "multiple"] += 1
                candidates["truncated" if item["candidates_truncated"] else "complete"] += 1
                reason_counts[item["retrieval_reason"]] += 1
    summary = {
        "protocol": "part05-original-gemma-validation-only-pilot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_version": FINAL_BENCHMARK_VERSION,
        "prompt_version": PROMPT_VERSION,
        "seed": seed,
        "selected_cases": len(selected),
        "unique_family_group_ids": len(family_sets),
        "available_validation_cases": len(rows),
        "sample_by_task_case_type": {
            " / ".join(key): n for key, n in sorted(counts.items())
        },
        "source_candidates": dict(sorted(candidates.items())),
        "retrieval_reasons": dict(sorted(reason_counts.items())),
        "max_candidates_per_request": max_candidates,
        "validation_csv_sha256": sha256_file(validation_file),
        "source_hashes": source_hashes,
        "prompt_file_sha256": sha256_file(prompt_path),
        "prompt_message_hashes": hashes,
        "references_loaded_into_prompts": False,
        "retrieval_predictions_loaded": False,
        "sealed_test_opened": False,
        "note": "Prompt JSONL contains source data. Keep it private and out of public GitHub.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PART 5 PILOT PREPARATION OK")
    print(f"Validation cases available: {len(rows):,}")
    print(f"Pilot selected: {len(selected)} cases across {len(family_sets)} family-group IDs")
    print(f"Candidates: {dict(candidates)}")
    print(f"Private request file: {prompt_path}")
    print(f"Metadata-only report: {summary_path}")
    print("Reference labels, saved baseline predictions and sealed test were NOT used as prompts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()
    run(args.repo, args.limit, args.seed, args.max_candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
