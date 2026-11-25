"""Inventory and validate the FDA and RxNorm source files used in Part 2.2.

Run from the repository root with:
    python -m hrp.data_loading --source-dir ../data --inventory-only
    python -m hrp.data_loading --source-dir ../data

The audit reads source records without modifying them. It writes aggregate
summaries only and does not attempt to match or normalize drug ingredients.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

FDA_FIELDS = {
    "ndc": (
        "product_ndc",
        "product_id",
        "brand_name",
        "generic_name",
        "product_type",
        "marketing_category",
        "dosage_form",
        "active_ingredients",
    ),
    "label": (
        "id",
        "set_id",
        "effective_time",
        "openfda",
        "active_ingredient",
        "purpose",
        "dosage_and_administration",
    ),
}

RXNCONSO_COLUMNS = 18
RXNSAT_COLUMNS = 13


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Sources:
    ndc: list[Path]
    labels: list[Path]
    rxnorm: list[Path]


@dataclass
class FdaAudit:
    source: str
    records: int = 0
    duplicate_ids: int = 0
    missing_ids: int = 0
    invalid_ingredient_lists: int = 0
    fields_present: Counter[str] = field(default_factory=Counter)
    fields_filled: Counter[str] = field(default_factory=Counter)
    field_types: Counter[tuple[str, str]] = field(default_factory=Counter)
    _seen_ids: set[str] = field(default_factory=set, repr=False)
    _seen_sets: set[str] = field(default_factory=set, repr=False)

    def add(self, record: dict[str, Any]) -> None:
        if not isinstance(record, dict):
            raise TypeError(f"Expected a JSON object in {self.source} data")

        self.records += 1
        key = "product_ndc" if self.source == "ndc" else "id"
        identifier = record.get(key)

        if not isinstance(identifier, str) or not identifier.strip():
            self.missing_ids += 1
        elif identifier in self._seen_ids:
            self.duplicate_ids += 1
        else:
            self._seen_ids.add(identifier)

        if self.source == "label":
            set_id = record.get("set_id")
            if isinstance(set_id, str) and set_id.strip():
                self._seen_sets.add(set_id)

        # A blank field and an absent field both count as missing for the audit.
        for name, value in record.items():
            self.fields_present[name] += 1
            self.field_types[(name, type(value).__name__)] += 1
            if value is not None and (
                (bool(value.strip()) if isinstance(value, str) else bool(value))
                or value == 0
                or value is False
            ):
                self.fields_filled[name] += 1

        if self.source == "ndc":
            ingredients = record.get("active_ingredients")
            if ingredients is not None and (
                not isinstance(ingredients, list)
                or any(not isinstance(item, dict) for item in ingredients)
            ):
                self.invalid_ingredient_lists += 1

    def summary(self) -> dict[str, Any]:
        result = {
            "source": self.source,
            "records": self.records,
            "unique_primary_ids": len(self._seen_ids),
            "missing_primary_ids": self.missing_ids,
            "duplicate_primary_ids": self.duplicate_ids,
        }
        if self.source == "label":
            # A single SPL set may have multiple label versions.
            result["unique_set_ids"] = len(self._seen_sets)
        if self.source == "ndc":
            result["malformed_active_ingredient_lists"] = self.invalid_ingredient_lists
        return result

    def field_rows(self) -> list[dict[str, Any]]:
        rows = []
        for name in sorted(self.fields_present.keys() | set(FDA_FIELDS[self.source])):
            types = {
                field_type: count
                for (field_name, field_type), count in self.field_types.items()
                if field_name == name
            }
            filled = self.fields_filled[name]
            rows.append(
                {
                    "source": self.source,
                    "field": name,
                    "records": self.records,
                    "present": self.fields_present[name],
                    "nonempty": filled,
                    "missing_or_blank": self.records - filled,
                    "nonempty_pct": round(100 * filled / self.records, 2) if self.records else 0,
                    "observed_types": json.dumps(types, sort_keys=True),
                }
            )
        return rows


def _is_fda_json(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".json", ".json.zip"))


def _source_kind(path: Path) -> str | None:
    name = path.name.lower().replace("_", "-")
    if re.search(r"(?:^|-)drug-ndc(?:-|\.)", name):
        return "ndc"
    if re.search(r"(?:^|-)drug-label(?:-|\.)", name):
        return "label"
    return None


def _select_one_copy(paths: list[Path]) -> list[Path]:
    """Prefer a ZIP when both the archive and its extracted JSON exist."""
    by_name: dict[str, Path] = {}
    for path in sorted(paths):
        logical_name = path.name.lower().removesuffix(".zip")
        previous = by_name.get(logical_name)
        if previous is None or path.name.lower().endswith(".zip"):
            by_name[logical_name] = path
    return sorted(by_name.values())


def source_directory(value: Path | None = None) -> Path:
    return (value or Path(os.environ.get("HRP_SOURCE_DIR", "../data"))).resolve()


def full_rxnorm_files(paths: list[Path]) -> list[Path]:
    full = [p for p in paths if "prescribe" not in {s.lower() for s in p.parts}]
    directories = {p.parent for p in full if p.name.upper() == "RXNCONSO.RRF"}
    if len(directories) > 1:
        raise ValueError("Multiple full RxNorm releases found. Select one source directory")
    return sorted(p for p in full if not directories or p.parent in directories)


def find_sources(source_dir: Path) -> Sources:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source data folder not found: {source_dir}")

    ndc: list[Path] = []
    labels: list[Path] = []
    rxnorm: list[Path] = []

    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() == ".rrf":
            rxnorm.append(path)
        elif _is_fda_json(path):
            kind = _source_kind(path)
            if kind == "ndc":
                ndc.append(path)
            elif kind == "label":
                labels.append(path)

    return Sources(_select_one_copy(ndc), _select_one_copy(labels), full_rxnorm_files(rxnorm))


def _json_members(path: Path) -> list[str]:
    try:
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".json")]
    except BadZipFile as exc:
        raise ValueError(f"Not a valid ZIP archive: {path}") from exc
    if not members:
        raise ValueError(f"ZIP archive contains no JSON files: {path}")
    return members


def inventory_rows(source_dir: Path, sources: Sources) -> list[dict[str, Any]]:
    rows = []
    groups = (("ndc", sources.ndc), ("label", sources.labels), ("rxnorm", sources.rxnorm))
    for source, paths in groups:
        for path in paths:
            rows.append(
                {
                    "source": source,
                    "relative_path": path.relative_to(source_dir).as_posix(),
                    "size_mb": round(path.stat().st_size / 1_000_000, 2),
                    "json_members": len(_json_members(path))
                    if path.name.lower().endswith(".zip")
                    else "",
                }
            )
    return rows


def _iter_fda_records(path: Path) -> Iterator[dict[str, Any]]:
    from ijson import items

    if path.name.lower().endswith(".zip"):
        with ZipFile(path) as archive:
            for member in _json_members(path):
                with archive.open(member) as stream:
                    yield from items(stream, "results.item")
    else:
        with path.open("rb") as stream:
            yield from items(stream, "results.item")


def audit_fda(paths: list[Path], source: str) -> FdaAudit:
    if not paths:
        raise ValueError(f"No {source} JSON files were discovered")
    audit = FdaAudit(source=source)
    for index, path in enumerate(paths, start=1):
        before = audit.records
        for record in _iter_fda_records(path):
            audit.add(record)
        if audit.records == before:
            raise ValueError(f"No results array records found in {path}")
        print(
            f"{source.upper()} {index}/{len(paths)}: {path.name}, {audit.records:,} records so far"
        )
    return audit


def _columns(line: str) -> list[str]:
    parts = line.rstrip("\r\n").split("|")
    if parts and parts[-1] == "":
        parts.pop()
    return parts


def audit_rrf(path: Path) -> dict[str, Any]:
    name = path.name.upper()
    expected = {
        "RXNCONSO.RRF": 18,
        "RXNSAT.RRF": 13,
        "RXNREL.RRF": 16,
        "RXNATOMARCHIVE.RRF": 16,
        "RXNCUI.RRF": 5,
        "RXNCUICHANGES.RRF": 7,
        "RXNDOC.RRF": 4,
        "RXNSAB.RRF": 25,
        "RXNSTY.RRF": 6,
    }.get(name)
    stats: dict[str, Any] = {
        "file": name,
        "rows": 0,
        "wrong_column_count": 0,
        "missing_rxcui": 0 if name in {"RXNCONSO.RRF", "RXNSAT.RRF"} else None,
        "duplicate_rxaui": 0 if name == "RXNCONSO.RRF" else None,
        "ndc_attributes": 0 if name == "RXNSAT.RRF" else None,
    }
    rxaui_seen: set[str] = set()
    outcomes = Counter()

    # Read one line at a time since the complete RxNorm release can be large.
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            stats["rows"] += 1
            if expected is None:
                continue

            columns = _columns(line)
            if len(columns) != expected:
                stats["wrong_column_count"] += 1
                if stats["wrong_column_count"] <= 3:
                    print(f"Unexpected column count in {name}, line {line_no}")
                continue

            if name == "RXNCONSO.RRF":
                accepted = columns[16] == "N" and (
                    columns[11] == "RXNORM"
                    or (
                        columns[11] == "MTHSPL"
                        and columns[12] == "SU"
                        and re.fullmatch(r"[A-Z0-9]{10}", columns[13]) is not None
                    )
                )
            elif name == "RXNSAT.RRF":
                accepted = columns[8] == "NDC" and columns[9] == "RXNORM" and columns[11] == "N"
            elif name == "RXNREL.RRF":
                accepted = (
                    columns[10] == "RXNORM"
                    and columns[14] in {"", "N"}
                    and columns[2] == columns[6] == "CUI"
                )
            else:
                accepted = False
            outcomes[
                "accepted_for_active_source_index"
                if accepted
                else "not_applicable_to_active_otc_mapping"
            ] += 1

            if not columns[0] and stats["missing_rxcui"] is not None:
                stats["missing_rxcui"] += 1

            if name == "RXNCONSO.RRF":
                rxaui = columns[7]
                if rxaui:
                    if rxaui in rxaui_seen:
                        stats["duplicate_rxaui"] += 1
                    rxaui_seen.add(rxaui)
            elif name == "RXNSAT.RRF" and columns[8] == "NDC":
                stats["ndc_attributes"] += 1

    outcomes["rejected_schema"] = stats["wrong_column_count"]
    stats["record_outcomes_json"] = json.dumps(dict(outcomes), sort_keys=True)
    return stats


def rrf_duplicate_count(path: Path, temporary_dir: Path) -> int:
    import duckdb

    temporary_dir.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as connection:
        connection.execute("SET memory_limit='512MB'")
        connection.execute("SET threads=1")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute("SET enable_progress_bar=false")
        connection.execute("SET temp_directory = ?", [str(temporary_dir.resolve())])
        return connection.execute(
            """
            SELECT count(*) FROM (
                SELECT line, lag(line) OVER (ORDER BY line) previous
                FROM read_csv(?, delim='\x1f', header=false,
                    columns={'line':'VARCHAR'}, quote='', escape='')
            ) WHERE line=previous
        """,
            [str(path)],
        ).fetchone()[0]


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _label_shard_check(files: list[Path]) -> dict[str, Any]:
    shard_parts: set[int] = set()
    declared_total: set[int] = set()
    for path in files:
        match = re.search(r"(\d+)-of-(\d+)", path.name)
        if match:
            shard_parts.add(int(match.group(1)))
            declared_total.add(int(match.group(2)))
    if not declared_total:
        return {"shards_named": 0, "missing_shards": "unknown"}
    if len(declared_total) != 1:
        return {"shards_named": len(shard_parts), "missing_shards": "inconsistent totals"}
    total = declared_total.pop()
    missing = sorted(set(range(1, total + 1)) - shard_parts)
    return {"shards_named": len(shard_parts), "declared_shards": total, "missing_shards": missing}


def run(source_dir: Path, output_dir: Path, inventory_only: bool = False) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = find_sources(source_dir)
    inventory = inventory_rows(source_dir, sources)
    _write_csv(
        output_dir / "source_inventory.csv",
        inventory,
        ["source", "relative_path", "size_mb", "json_members"],
    )

    print("NDC JSON files:", len(sources.ndc))
    print("Drug Labeling JSON files:", len(sources.labels))
    print("RxNorm RRF files:", len(sources.rxnorm))
    print("Inventory:", output_dir / "source_inventory.csv")

    if inventory_only:
        return {"inventory": inventory}

    if not sources.ndc or not sources.labels:
        raise ValueError("FDA files not discovered. Review the filenames in your source directory")

    rrf_names = {path.name.upper() for path in sources.rxnorm}
    missing_rrf = {"RXNCONSO.RRF", "RXNSAT.RRF"} - rrf_names
    if missing_rrf:
        raise ValueError(f"Missing required RxNorm files: {sorted(missing_rrf)}")

    ndc = audit_fda(sources.ndc, "ndc")
    labels = audit_fda(sources.labels, "label")
    rxnorm = [audit_rrf(path) for path in sources.rxnorm]
    print("RxNorm files audited:", len(rxnorm))

    fields = ndc.field_rows() + labels.field_rows()
    _write_csv(
        output_dir / "fda_field_coverage.csv",
        fields,
        [
            "source",
            "field",
            "records",
            "present",
            "nonempty",
            "missing_or_blank",
            "nonempty_pct",
            "observed_types",
        ],
    )
    _write_csv(
        output_dir / "rxnorm_file_summary.csv",
        rxnorm,
        [
            "file",
            "rows",
            "wrong_column_count",
            "missing_rxcui",
            "duplicate_rxaui",
            "ndc_attributes",
            "record_outcomes_json",
        ],
    )

    summary = {
        "ndc": ndc.summary(),
        "label": labels.summary(),
        "label_shards": _label_shard_check(sources.labels),
        "rxnorm": rxnorm,
        "data_handling": "Aggregate source audit, no ingredient normalization or OTC filtering",
    }
    (output_dir / "source_audit_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("FDA and RxNorm audit complete")
    print(
        json.dumps(
            {
                "ndc": summary["ndc"],
                "label": summary["label"],
                "label_shards": summary["label_shards"],
            },
            indent=2,
        )
    )
    return summary


def audit_ndc_product_types(paths: list[Path]) -> dict:
    """Keep cross-type listings separate from ingredient contradictions."""
    from collections import defaultdict

    from hrp.data_cleaning import normalize_name, quantity_key, route_signature

    groups = defaultdict(list)
    for path in paths:
        for row in _iter_fda_records(path):
            ingredients = tuple(
                sorted(
                    (
                        normalize_name(item.get("name")),
                        quantity_key(item.get("strength", ""))
                        or (normalize_name(item.get("strength")), "literal"),
                    )
                    for item in (row.get("active_ingredients") or [])
                )
            )
            groups[row.get("product_ndc")].append(
                (
                    row.get("product_type"),
                    ingredients,
                    normalize_name(row.get("dosage_form")),
                    route_signature(row.get("route")),
                )
            )
    relationships = []
    for ndc, rows in sorted(groups.items()):
        types = sorted({r[0] for r in rows if r[0]})
        if "HUMAN OTC DRUG" not in types or len(types) == 1:
            continue
        conflicts = any(len({r[i] for r in rows if r[i]}) > 1 for i in (1, 2, 3))
        relationships.append(
            {
                "product_ndc": ndc,
                "product_types": types,
                "identity_outcome": "conflicting_identity" if conflicts else "compatible_identity",
                "classification_outcome": (
                    "unresolved_otc_and_prescription_classification"
                    if "HUMAN PRESCRIPTION DRUG" in types
                    else "separate_unfinished_listing_not_used_as_otc_evidence"
                ),
            }
        )
    return {
        "cross_type_otc_ndcs": len(relationships),
        "identity_conflicts": sum(
            r["identity_outcome"] == "conflicting_identity" for r in relationships
        ),
        "relationships": relationships,
        "policy": "An explicit human OTC row supports ingredient tasks. Other product types are not OTC evidence. "
        "Classification disagreement is reported separately from ingredient identity and does not prove exclusive OTC status.",
    }


def verify_sources(source_dir: Path, output_dir: Path) -> dict:
    """Audit every source row and retain hashes without extracting archives."""
    source_dir = source_directory(source_dir)
    sources = find_sources(source_dir)
    required = {
        "RXNCONSO.RRF",
        "RXNSAT.RRF",
        "RXNREL.RRF",
        "RXNDOC.RRF",
        "RXNSAB.RRF",
        "RXNATOMARCHIVE.RRF",
        "RXNCUI.RRF",
        "RXNCUICHANGES.RRF",
        "RXNSTY.RRF",
    }
    if missing := required - {p.name.upper() for p in sources.rxnorm}:
        raise ValueError(f"Incomplete full RxNorm release {sorted(missing)}")
    if _label_shard_check(sources.labels).get("missing_shards") != []:
        raise ValueError("Incomplete FDA labeling shards")
    cached_path = output_dir / "source_manifest.json"
    if cached_path.is_file():
        cached = json.loads(cached_path.read_text(encoding="utf-8"))
        unchanged = True
        for entry in cached["files"]:
            path = source_dir / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"]:
                unchanged = False
                break
            with path.open("rb") as stream:
                digest = sha256()
                for block in iter(lambda: stream.read(2**20), b""):
                    digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    unchanged = False
                    break
        selected_paths = {
            p.relative_to(source_dir).as_posix()
            for p in sources.ndc + sources.labels + sources.rxnorm
        }
        if unchanged and selected_paths == {
            e["path"] for e in cached["files"] if e["use"] == "selected"
        }:
            if "ndc_product_type_relationships" not in cached:
                cached["ndc_product_type_relationships"] = audit_ndc_product_types(sources.ndc)
                cached_path.write_text(json.dumps(cached, indent=2) + "\n", encoding="utf-8")
            print("Reused complete source audit after verifying every file hash", flush=True)
            return cached
    summary = run(source_dir, output_dir)
    for row, path in zip(summary["rxnorm"], sources.rxnorm, strict=True):
        row["duplicate_rows"] = rrf_duplicate_count(path, output_dir / ".audit_temp")
    files = []
    selected = set(sources.ndc + sources.labels + sources.rxnorm)
    all_files = sorted(selected | set(source_dir.rglob("*.RRF")))
    for path in all_files:
        with path.open("rb") as stream:
            digest = sha256()
            for block in iter(lambda: stream.read(2**20), b""):
                digest.update(block)
        entry = {
            "path": path.relative_to(source_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
            "use": "selected" if path in selected else "not_applicable_prescribable_subset",
        }
        if path.suffix.upper() == ".RRF" and path not in selected:
            entry["audit"] = audit_rrf(path)
        if path.name.endswith(".zip"):
            with ZipFile(path) as archive:
                entry["members"] = [
                    {"name": i.filename, "bytes": i.file_size, "crc": i.CRC}
                    for i in archive.infolist()
                ]
                with archive.open(_json_members(path)[0]) as stream:
                    from ijson import items

                    entry["metadata"] = next(items(stream, "meta"), {})
        files.append(entry)
    summary["files"] = files
    summary["rxnorm_release"] = [
        p.read_text(encoding="utf-8").splitlines()[0] for p in source_dir.rglob("Readme_Full_*.txt")
    ]
    summary["record_outcomes"] = {
        "ndc": {"examined_for_otc_catalog": summary["ndc"]["records"]},
        "label": {"examined_for_product_links": summary["label"]["records"]},
        "rxnorm": {r["file"]: json.loads(r["record_outcomes_json"]) for r in summary["rxnorm"]},
    }
    summary["selected_rxnorm_directory"] = (
        sources.rxnorm[0].parent.relative_to(source_dir).as_posix()
    )
    summary["ndc_product_type_relationships"] = audit_ndc_product_types(sources.ndc)
    (output_dir / "source_manifest.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the downloaded FDA and RxNorm data")
    parser.add_argument("--source-dir", type=Path, default=source_directory())
    parser.add_argument("--out", type=Path, default=Path("results/part02"))
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_sources(args.source_dir, args.out)
    else:
        run(args.source_dir, args.out, args.inventory_only)


if __name__ == "__main__":
    main()
