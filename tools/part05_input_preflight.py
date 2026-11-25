"""Part 5 input-contract audit for the frozen HRP final_v5 validation inputs.

Reads the validation split and its existing deterministic retrieval predictions.
Never opens the sealed test set. Reports schemas, counts, and checksums only.

Run from the repository root:
    python tools/part05_input_preflight.py --repo .
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def ensure_not_test(path: Path) -> None:
    """Prevent accidental use of test data during model development."""
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if "sealed_test" in name or "test_predictions" in name or any(
        p in {"test", "heldout_test", "held_out_test"} for p in parts
    ):
        raise ValueError(f"A sealed-test path is not permitted in Part 5: {path}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def shape_of(value: Any) -> str:
    """Return a safe structural description, not source-record contents."""
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object{" + ",".join(sorted(map(str, value.keys()))[:40]) + "}"
    if isinstance(value, list):
        head = shape_of(value[0]) if value else "empty"
        return f"array[{len(value)}] of {head}"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        if not value.strip():
            return "empty_string"
        if value.lstrip().startswith(("[", "{")):
            try:
                return "json_string:" + shape_of(json.loads(value))
            except (ValueError, TypeError):
                pass
        return "string"
    return type(value).__name__


def audit_csv(path: Path) -> dict[str, Any]:
    ensure_not_test(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if not fields:
            raise ValueError("Validation CSV has no header")
        count = 0
        nonempty = Counter()
        shapes: dict[str, set[str]] = {col: set() for col in fields}
        for row in reader:
            count += 1
            for col in fields:
                value = row[col]
                if value is not None and str(value).strip():
                    nonempty[col] += 1
                if count <= 8:
                    shapes[col].add(shape_of(value))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": count,
        "fields": fields,
        "nonnull_counts": {col: nonempty[col] for col in fields},
        "sample_shapes_only": {col: sorted(shapes[col]) for col in fields},
    }


def audit_jsonl(path: Path) -> dict[str, Any]:
    ensure_not_test(path)
    count = 0
    field_counts: Counter[str] = Counter()
    field_shapes: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc
            if not isinstance(record, dict):
                raise TypeError(f"Expected JSON object on line {line_number}")
            for key, value in record.items():
                field_counts[key] += 1
                if count <= 8:
                    field_shapes.setdefault(key, set()).add(shape_of(value))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": count,
        "fields": sorted(field_counts),
        "field_occurrences": dict(sorted(field_counts.items())),
        "sample_shapes_only": {k: sorted(v) for k, v in sorted(field_shapes.items())},
    }


def run(repo: Path, out: Path) -> dict[str, Any]:
    repo = repo.resolve()
    v5 = repo / "data" / "processed" / "final_v5"
    validation_path = v5 / "validation.csv"
    retrieval_path = v5 / "validation_retrieval_predictions.jsonl"
    freeze_path = repo / "results" / "final_v5" / "part03" / "freeze_manifest.json"
    for path in (validation_path, retrieval_path, freeze_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing existing final_v5 artifact: {path}")

    validation = audit_csv(validation_path)
    retrieval = audit_jsonl(retrieval_path)
    freeze_info = {"path": str(freeze_path), "sha256": sha256_file(freeze_path)}
    report = {
        "protocol": "Part 5 validation-only input-contract preflight",
        "benchmark_version": "final_v5",
        "validation": validation,
        "retrieval_predictions": retrieval,
        "freeze_manifest": freeze_info,
        "test_set_opened": False,
        "note": (
            "The existing baseline predictions are for evaluation/comparison only. "
            "Do not put reference answers or baseline predicted ingredient sets into the LLM prompt."
        ),
    }
    if validation["row_count"] == 0 or retrieval["row_count"] == 0:
        raise ValueError("Unexpected empty validation inputs")
    # Prediction-level counts may differ from case counts. Do not assert equality blindly.
    out = out if out.is_absolute() else repo / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PART 5 PREFLIGHT OK")
    print(f"Validation CSV: {validation['row_count']} rows, {len(validation['fields'])} columns")
    print(f"Retrieval JSONL: {retrieval['row_count']} objects, {len(retrieval['fields'])} top-level keys")
    print("The sealed test file was NOT accessed.")
    print(f"Safe schema report: {out}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("results/part05/input_contract_preflight.json"))
    args = parser.parse_args()
    try:
        run(args.repo, args.out)
    except (OSError, ValueError, TypeError) as exc:
        print(f"PART 5 PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
