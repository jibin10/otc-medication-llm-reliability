# Link OTC products to FDA Drug Labeling records

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import ijson


def _strings(value: object) -> list[str]:
    """Return nonempty identifiers without changing their formatting."""
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []

    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]

    return []


def match_fda_labels(
    raw_otc_path: Path,
    consistent_path: Path,
    pending_path: Path,
    label_dir: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict:
    """Find label candidates and measure linkage coverage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Include pending products so labels can help resolve missing information
    ndc_status = {}

    for path, status in (
        (consistent_path, "consistent"),
        (pending_path, "pending"),
    ):
        with path.open(newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                ndc = row["product_ndc"].strip()

                if ndc in ndc_status and ndc_status[ndc] != status:
                    raise ValueError(f"NDC appears in both groups: {ndc}")

                ndc_status[ndc] = status

    # Build lookup indexes from the original NDC records
    by_spl_id = defaultdict(set)
    by_set_id = defaultdict(set)

    with raw_otc_path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)
            ndc = str(record.get("product_ndc") or "").strip()

            if ndc not in ndc_status:
                continue

            metadata = record.get("openfda") or {}
            if not isinstance(metadata, dict):
                metadata = {}

            for item in _strings(record.get("spl_id")) + _strings(metadata.get("spl_id")):
                by_spl_id[item.casefold()].add(ndc)

            for item in _strings(record.get("spl_set_id")) + _strings(metadata.get("spl_set_id")):
                by_set_id[item.casefold()].add(ndc)

    label_archives = sorted(label_dir.glob("drug-label-*.json.zip"))

    if len(label_archives) != 14:
        raise ValueError(f"Expected 14 labeling archives, found {len(label_archives)}")

    candidates = defaultdict(dict)
    label_records = 0
    label_outcomes = Counter()

    # Stream the labels and retain only matches to our OTC products
    for archive_path in label_archives:
        with ZipFile(archive_path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".json")]

            if len(members) != 1:
                raise ValueError(f"Unexpected JSON members in {archive_path.name}")

            with archive.open(members[0]) as source:
                for label in ijson.items(source, "results.item"):
                    label_records += 1

                    label_id = str(label.get("id") or "").strip().casefold()

                    set_id = str(label.get("set_id") or "").strip().casefold()

                    if not label_id:
                        continue

                    metadata = label.get("openfda") or {}
                    if not isinstance(metadata, dict):
                        metadata = {}

                    evidence = defaultdict(set)

                    for ndc in by_spl_id.get(label_id, ()):
                        evidence[ndc].add("spl_id")

                    if set_id:
                        for ndc in by_set_id.get(set_id, ()):
                            evidence[ndc].add("spl_set_id")

                    for ndc in _strings(metadata.get("product_ndc")):
                        if ndc in ndc_status:
                            evidence[ndc].add("product_ndc")

                    label_outcomes["linked_otc" if evidence else "not_applicable_no_otc_link"] += 1

                    for ndc, methods in evidence.items():
                        current = candidates[ndc].setdefault(
                            label_id,
                            {
                                "set_id": set_id,
                                "effective_time": str(label.get("effective_time") or ""),
                                "version": str(label.get("version") or ""),
                                "metadata": {
                                    key: metadata.get(key, [])
                                    for key in (
                                        "product_ndc",
                                        "package_ndc",
                                        "substance_name",
                                        "route",
                                        "dosage_form",
                                        "brand_name",
                                        "generic_name",
                                        "rxcui",
                                    )
                                },
                                "active_ingredient": (label.get("active_ingredient") or []),
                                "methods": set(),
                            },
                        )

                        current["methods"].update(methods)

    # Prefer exact SPL document links over broader product NDC matches
    priority = {
        "spl_id": 0,
        "spl_set_id": 1,
        "product_ndc": 2,
    }

    counts = Counter()

    links_path = output_dir / "otc_ndc_label_links.csv"
    review_path = report_dir / "otc_label_ambiguous_review.csv"

    with (
        links_path.open("w", newline="", encoding="utf-8") as links,
        review_path.open("w", newline="", encoding="utf-8") as review,
    ):
        writer = csv.writer(links)
        writer.writerow(
            [
                "product_ndc",
                "catalog_status",
                "link_status",
                "link_method",
                "label_id",
                "label_set_id",
                "effective_time",
                "active_ingredient_json",
                "label_version",
                "label_metadata_json",
                "candidate_labels_json",
            ]
        )

        reviewer = csv.writer(review)
        reviewer.writerow(
            [
                "product_ndc",
                "label_id",
                "label_set_id",
                "match_methods",
            ]
        )

        for ndc in sorted(ndc_status):
            group = candidates.get(ndc, {})

            if not group:
                counts["unmatched"] += 1
                writer.writerow(
                    [
                        ndc,
                        ndc_status[ndc],
                        "unmatched",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )
                continue

            best_rank = min(
                min(priority[method] for method in item["methods"]) for item in group.values()
            )

            best = [
                (label_id, item)
                for label_id, item in group.items()
                if min(priority[method] for method in item["methods"]) == best_rank
            ]

            sets = {item["set_id"] for _, item in best}
            if len(best) > 1 and len(sets) == 1 and "" not in sets:
                newest = max(
                    (item["effective_time"], int(item["version"] or 0)) for _, item in best
                )
                best = [
                    (identifier, item)
                    for identifier, item in best
                    if (item["effective_time"], int(item["version"] or 0)) == newest
                ]

            # Multiple candidates at the same priority remain unresolved
            if len(best) != 1:
                counts["ambiguous"] += 1

                writer.writerow(
                    [
                        ndc,
                        ndc_status[ndc],
                        "ambiguous",
                        "",
                        "",
                        "",
                        "",
                        "",
                    ]
                )

                for label_id, item in sorted(group.items()):
                    reviewer.writerow(
                        [
                            ndc,
                            label_id,
                            item["set_id"],
                            ",".join(sorted(item["methods"])),
                        ]
                    )
                continue

            label_id, item = best[0]
            method = min(item["methods"], key=priority.get)

            counts["matched"] += 1
            counts[f"matched_by_{method}"] += 1

            if item["active_ingredient"]:
                counts["matched_with_ingredient_text"] += 1

                if ndc_status[ndc] == "pending":
                    counts["pending_with_ingredient_text"] += 1

            writer.writerow(
                [
                    ndc,
                    ndc_status[ndc],
                    "matched",
                    method,
                    label_id,
                    item["set_id"],
                    item["effective_time"],
                    json.dumps(
                        item["active_ingredient"],
                        ensure_ascii=False,
                        default=str,
                    ),
                    item["version"],
                    json.dumps(item["metadata"]),
                    json.dumps(
                        [
                            {**value, "label_id": identifier, "methods": sorted(value["methods"])}
                            for identifier, value in sorted(group.items())
                        ],
                        default=str,
                    ),
                ]
            )

    total = len(ndc_status)

    assert counts["matched"] + counts["ambiguous"] + counts["unmatched"] == total

    summary = {
        "unique_ndcs": total,
        "consistent_ndcs": sum(status == "consistent" for status in ndc_status.values()),
        "pending_ndcs": sum(status == "pending" for status in ndc_status.values()),
        "label_records_scanned": label_records,
        "label_record_outcomes": dict(label_outcomes),
        "matched_ndcs": counts["matched"],
        "ambiguous_ndcs": counts["ambiguous"],
        "unmatched_ndcs": counts["unmatched"],
        "matched_by_spl_id": counts["matched_by_spl_id"],
        "matched_by_spl_set_id": counts["matched_by_spl_set_id"],
        "matched_by_product_ndc": counts["matched_by_product_ndc"],
        "matched_with_ingredient_text": counts["matched_with_ingredient_text"],
        "pending_with_ingredient_text": counts["pending_with_ingredient_text"],
        "match_rate_pct": round(100 * counts["matched"] / total, 2),
    }

    (report_dir / "otc_label_linkage_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary


def ndc_product_key(value: str) -> str | None:
    """Convert a two-part FDA product NDC to a nine-digit key."""
    parts = str(value).strip().split("-")
    if len(parts) != 2:
        return None

    labeler, product = parts

    if (
        not labeler.isdigit()
        or not product.isdigit()
        or len(labeler) not in (4, 5)
        or len(product) not in (3, 4)
    ):
        return None

    return labeler.zfill(5) + product.zfill(4)


def rxnorm_package_product_key(value: str) -> str | None:
    """Extract the product key from a normalized 11-digit NDC."""
    ndc = value.strip()

    if len(ndc) != 11 or not ndc.isascii() or not ndc.isdigit():
        return None

    return ndc[:9]


def link_rxnorm_products(
    fda_links_path: Path,
    rxnorm_rrf_dir: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict:
    """Match OTC NDCs to active RxNorm drug-product concepts."""
    sat_path = rxnorm_rrf_dir / "RXNSAT.RRF"
    conso_path = rxnorm_rrf_dir / "RXNCONSO.RRF"

    for path in (fda_links_path, sat_path, conso_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Reuse all product identifiers from the completed FDA label audit
    with fda_links_path.open(newline="", encoding="utf-8") as file:
        products = list(csv.DictReader(file))

    product_lookup = defaultdict(set)
    seen_ndcs = set()

    for row in products:
        ndc = row["product_ndc"].strip()

        if ndc in seen_ndcs:
            raise ValueError(f"Repeated product NDC: {ndc}")

        seen_ndcs.add(ndc)
        key = ndc_product_key(ndc)

        if key:
            product_lookup[key].add(ndc)

    candidates = defaultdict(set)
    counts = Counter()

    # Use active NDC associations asserted by RxNorm
    with sat_path.open(encoding="utf-8") as file:
        for line in file:
            fields = line.rstrip("\r\n").split("|")

            if len(fields) != 14:
                raise ValueError("Unexpected RXNSAT record layout")

            if fields[8] != "NDC" or fields[9] != "RXNORM":
                continue

            if fields[11] != "N":
                continue

            counts["active_ndc_attributes"] += 1
            key = rxnorm_package_product_key(fields[10])

            if key is None:
                counts["nonstandard_ndc_values"] += 1
                continue

            if key in product_lookup and fields[0]:
                candidates[key].add(fields[0])

    required_rxcuis = set()

    for values in candidates.values():
        required_rxcuis.update(values)

    drug_concepts = {}
    drug_types = {"SCD", "SBD", "GPCK", "BPCK"}

    # Keep active normalized drug concepts, not source-only names
    with conso_path.open(encoding="utf-8") as file:
        for line in file:
            fields = line.rstrip("\r\n").split("|")

            if len(fields) != 19:
                raise ValueError("Unexpected RXNCONSO record layout")

            if (
                fields[0] in required_rxcuis
                and fields[11] == "RXNORM"
                and fields[12] in drug_types
                and fields[16] == "N"
            ):
                drug_concepts[fields[0]] = (
                    fields[12],
                    fields[14],
                )

    output_path = output_dir / "otc_rxnorm_product_links.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "product_ndc",
                "catalog_status",
                "fda_label_status",
                "rxnorm_status",
                "rxnorm_rxcui",
                "rxnorm_tty",
                "rxnorm_drug_name",
                "candidate_rxcuis_json",
            ]
        )

        for row in products:
            ndc = row["product_ndc"].strip()
            key = ndc_product_key(ndc)
            active = []

            if key is None:
                status = "invalid_product_ndc"

            elif len(product_lookup[key]) > 1:
                status = "product_key_collision"

            else:
                linked = candidates.get(key, set())
                active = sorted(linked.intersection(drug_concepts))

                if len(active) == 1:
                    status = "matched"
                elif len(active) > 1:
                    status = "multiple_drug_concepts"
                elif linked:
                    status = "no_active_drug_concept"
                else:
                    status = "no_rxnorm_ndc"

            counts[status] += 1

            if status == "matched":
                rxcui = active[0]
                tty, name = drug_concepts[rxcui]

                if row["catalog_status"] == "pending":
                    counts["pending_matched"] += 1
            else:
                rxcui, tty, name = "", "", ""

            writer.writerow(
                [
                    ndc,
                    row["catalog_status"],
                    row["link_status"],
                    status,
                    rxcui,
                    tty,
                    name,
                    json.dumps(active),
                ]
            )

    total = len(products)

    # Ensure each input product received exactly one mapping status
    statuses = [
        "matched",
        "multiple_drug_concepts",
        "no_active_drug_concept",
        "no_rxnorm_ndc",
        "invalid_product_ndc",
        "product_key_collision",
    ]

    assert sum(counts[status] for status in statuses) == total

    summary = {
        "unique_otc_ndcs": total,
        "matched_drug_concepts": counts["matched"],
        "match_rate_pct": round(100 * counts["matched"] / total, 2),
        "multiple_drug_concepts": counts["multiple_drug_concepts"],
        "no_active_drug_concept": counts["no_active_drug_concept"],
        "no_rxnorm_ndc": counts["no_rxnorm_ndc"],
        "invalid_product_ndc": counts["invalid_product_ndc"],
        "product_key_collision": counts["product_key_collision"],
        "pending_ndcs_matched": counts["pending_matched"],
        "active_rxnorm_ndc_attributes_scanned": counts["active_ndc_attributes"],
        "nonstandard_rxnorm_ndc_values": counts["nonstandard_ndc_values"],
    }

    summary_path = report_dir / "otc_rxnorm_linkage_summary.json"

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
