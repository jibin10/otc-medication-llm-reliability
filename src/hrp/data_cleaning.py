"""Prepare OTC product records for name and duplicate review."""

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path


def normalize_name(value: str | None) -> str:
    """Normalize text without removing product-specific terms."""
    if not isinstance(value, str):
        return ""

    text = unicodedata.normalize("NFKC", value)
    return " ".join(text.casefold().split())


def quantity_key(value: str) -> tuple[str, str] | None:
    """Compare quantities only within their stated physical dimensions."""
    text = normalize_name(value).replace(chr(181) + "g", "mcg").replace(chr(956) + "g", "mcg")
    number = r"(?:\d+(?:\.\d*)?|\.\d+)"
    percent = re.fullmatch(rf"({number})\s*%\s*(w/w|w/v|v/v)?", text)
    if percent:
        amount, basis = percent.groups()
        if not basis:
            return str(Fraction(amount)), "percent_unspecified"
        return str(Fraction(amount) / 100), {
            "w/w": "mass/mass",
            "w/v": "mass/volume",
            "v/v": "volume/volume",
        }[basis]
    match = re.fullmatch(
        rf"({number})\s*(mcg|ug|mg|g|kg|ml|l|iu|\[iu\]|u|%)"
        rf"(?:\s*/\s*({number})?\s*(mcg|ug|mg|g|kg|ml|l|1|actuation|dose|patch))?",
        text,
    )
    if not match:
        return None
    amount, unit, denominator, denominator_unit = match.groups()
    units = {
        "mcg": (Fraction(1, 10**6), "mass"),
        "ug": (Fraction(1, 10**6), "mass"),
        "mg": (Fraction(1, 1000), "mass"),
        "g": (Fraction(1), "mass"),
        "kg": (Fraction(1000), "mass"),
        "ml": (Fraction(1), "volume"),
        "l": (Fraction(1000), "volume"),
        "iu": (Fraction(1), "iu"),
        "[iu]": (Fraction(1), "iu"),
        "u": (Fraction(1), "unit"),
        "1": (Fraction(1), "item"),
        "dose": (Fraction(1), "dose"),
        "actuation": (Fraction(1), "actuation"),
        "patch": (Fraction(1), "patch"),
    }
    if unit not in units:
        return None
    scale, kind = units[unit]
    den_scale, den_kind = units[denominator_unit or "1"]
    den = Fraction(denominator or "1") * den_scale
    if den <= 0 or Fraction(amount) < 0:
        return None
    return str(Fraction(amount) * scale / den), kind + "/" + den_kind


def ingredient_identity(ingredient: dict) -> str:
    return ingredient.get("identity") or normalize_name(ingredient.get("name"))


def formulation_key(ingredients: list, dosage: str, route: list) -> tuple | None:
    if not ingredients or not normalize_name(dosage) or not route_signature(route):
        return None
    entries = []
    for item in ingredients:
        name = ingredient_identity(item)
        quantity = quantity_key(item.get("strength", ""))
        if not name or quantity is None:
            return None
        entries.append((name, quantity))
    return tuple(sorted(entries)), normalize_name(dosage), route_signature(route)


def ingredient_signature(ingredients: object) -> tuple | None:
    """Create a comparison key when ingredient names and strengths exist."""
    if not isinstance(ingredients, list) or not ingredients:
        return None

    entries = []

    for item in ingredients:
        if not isinstance(item, dict):
            return None

        name = ingredient_identity(item)
        strength = normalize_name(item.get("strength"))

        if not name or not strength:
            return None

        entries.append((name, strength))

    return tuple(sorted(entries))


def route_signature(route: object) -> tuple | None:
    """Preserve route information for formulation comparison."""
    values = [route] if isinstance(route, str) else route if isinstance(route, list) else []

    keys = sorted({normalize_name(value) for value in values if normalize_name(value)})

    return tuple(keys) if keys else None


def classify_repeated_ndc(rows: list[dict]) -> str:
    """Classify repeated identifiers without removing records."""
    if len({row["fingerprint"] for row in rows}) == 1:
        return "exact_duplicate"

    # Different complete ingredient, dosage or route values need review
    for field in ("ingredients", "dosage", "route"):
        known = {row[field] for row in rows if row[field] is not None}

        if len(known) > 1:
            return "review_formulation_difference"

    if any(row["ingredients"] is None or row["dosage"] is None for row in rows):
        return "incomplete_formulation_data"

    return "other_record_difference"


def prepare_otc_catalog(
    input_path: Path,
    catalog_path: Path,
    report_dir: Path,
) -> dict:
    """Standardize product names and audit repeated NDC identifiers."""
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    ndc_groups = defaultdict(list)
    name_ndcs = defaultdict(set)
    counts = Counter()

    columns = [
        "source_row",
        "product_ndc",
        "brand_name",
        "generic_name",
        "display_name",
        "normalized_name",
        "name_source",
        "dosage_form",
        "route",
        "marketing_category",
        "active_ingredients_json",
    ]

    # Read the existing OTC subset rather than the original FDA archive
    with (
        input_path.open(encoding="utf-8") as source,
        catalog_path.open("w", encoding="utf-8", newline="") as destination,
    ):
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()

        for line_no, line in enumerate(source, start=1):
            if not line.strip():
                continue

            product = json.loads(line)

            if product.get("product_type") != "HUMAN OTC DRUG":
                raise ValueError(f"Unexpected product type at line {line_no}")

            ndc = str(product.get("product_ndc") or "").strip()
            if not ndc:
                raise ValueError(f"Missing NDC at line {line_no}")

            brand = product.get("brand_name") or ""
            generic = product.get("generic_name") or ""

            # Use the generic name only when the brand is unavailable
            name_source = "brand" if normalize_name(brand) else "generic"
            display_name = brand if name_source == "brand" else generic
            name_key = normalize_name(display_name)

            if not name_key:
                counts["missing_names"] += 1

            ingredients = product.get("active_ingredients")
            ingredient_key = ingredient_signature(ingredients)
            dosage_key = normalize_name(product.get("dosage_form")) or None
            route_key = route_signature(product.get("route"))

            # Fingerprints distinguish exact copies from changed records
            payload = json.dumps(product, sort_keys=True, ensure_ascii=False)
            fingerprint = sha256(payload.encode("utf-8")).hexdigest()

            # Retain original names and ingredient descriptions
            writer.writerow(
                {
                    "source_row": line_no,
                    "product_ndc": ndc,
                    "brand_name": brand,
                    "generic_name": generic,
                    "display_name": display_name,
                    "normalized_name": name_key,
                    "name_source": name_source,
                    "dosage_form": product.get("dosage_form") or "",
                    "route": json.dumps(product.get("route")),
                    "marketing_category": (product.get("marketing_category") or ""),
                    "active_ingredients_json": json.dumps(ingredients, ensure_ascii=False),
                }
            )

            ndc_groups[ndc].append(
                {
                    "source_row": line_no,
                    "fingerprint": fingerprint,
                    "ingredients": ingredient_key,
                    "dosage": dosage_key,
                    "route": route_key,
                }
            )

            if name_key:
                name_ndcs[name_key].add(ndc)

            counts["records"] += 1

    # Save every repeated identifier for later manual inspection
    review_path = report_dir / "ndc_duplicate_review.csv"

    with review_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "product_ndc",
                "record_count",
                "review_status",
                "source_rows",
            ]
        )

        for ndc, rows in sorted(ndc_groups.items()):
            if len(rows) < 2:
                continue

            status = classify_repeated_ndc(rows)

            counts[status] += 1
            counts["repeated_ndc_groups"] += 1
            counts["repeated_occurrences"] += len(rows) - 1

            writer.writerow(
                [
                    ndc,
                    len(rows),
                    status,
                    ",".join(str(r["source_row"]) for r in rows),
                ]
            )

    summary = {
        "records": counts["records"],
        "unique_product_ndcs": len(ndc_groups),
        "unique_normalized_names": len(name_ndcs),
        "missing_names": counts["missing_names"],
        "names_used_by_multiple_ndcs": sum(len(ndcs) > 1 for ndcs in name_ndcs.values()),
        "repeated_ndc_groups": counts["repeated_ndc_groups"],
        "repeated_occurrences": counts["repeated_occurrences"],
        "exact_duplicate_groups": counts["exact_duplicate"],
        "possible_formulation_difference_groups": counts["review_formulation_difference"],
        "incomplete_formulation_groups": counts["incomplete_formulation_data"],
        "other_record_difference_groups": counts["other_record_difference"],
    }

    report_path = report_dir / "otc_name_duplicate_summary.json"
    report_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary


def build_consistent_otc_catalog(
    stage_path: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict:
    """Prepare consistent NDC records and retain uncertain ones for review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Group staged records by their original NDC identifier
    groups = defaultdict(list)

    with stage_path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])

        for row in reader:
            groups[row["product_ndc"]].append(row)

    def record_key(row: dict) -> tuple:
        ingredients = json.loads(row["active_ingredients_json"] or "null")
        route = json.loads(row["route"] or "null")

        return (
            normalize_name(row["brand_name"]),
            normalize_name(row["generic_name"]),
            normalize_name(row["dosage_form"]),
            normalize_name(row["marketing_category"]),
            route_signature(route),
            ingredient_signature(ingredients),
        )

    accepted_path = output_dir / "otc_catalog_consistent.csv"
    pending_path = output_dir / "otc_catalog_pending_review.csv"
    counts = Counter()

    with (
        accepted_path.open("w", encoding="utf-8", newline="") as accepted_file,
        pending_path.open("w", encoding="utf-8", newline="") as pending_file,
    ):
        accepted = csv.DictWriter(
            accepted_file,
            fieldnames=columns + ["source_rows", "source_record_count"],
        )
        pending = csv.DictWriter(
            pending_file,
            fieldnames=columns + ["review_reason"],
        )

        accepted.writeheader()
        pending.writeheader()

        for records in groups.values():
            keys = [record_key(row) for row in records]

            # Hold records with missing ingredients or conflicting details
            if any(key[-1] is None for key in keys):
                reason = "missing_ingredient_details"
            elif any(not row["normalized_name"] or not row["dosage_form"] for row in records):
                reason = "missing_product_identity"
            elif len(set(keys)) > 1:
                reason = "conflicting_record_fields"
            else:
                reason = None

            if reason:
                counts[f"pending_{reason}_ndcs"] += 1

                for row in records:
                    pending.writerow(
                        {
                            **row,
                            "review_reason": reason,
                        }
                    )
                    counts["pending_records"] += 1
                continue

            # Retain one candidate while preserving all source row references
            selected = min(
                records,
                key=lambda row: int(row["source_row"]),
            )

            accepted.writerow(
                {
                    **selected,
                    "source_rows": ",".join(row["source_row"] for row in records),
                    "source_record_count": len(records),
                }
            )

            counts["consistent_ndcs"] += 1
            counts["consistent_source_records"] += len(records)

    summary = {
        "source_records": sum(len(rows) for rows in groups.values()),
        "unique_ndcs": len(groups),
        "consistent_ndcs": counts["consistent_ndcs"],
        "pending_ndcs": len(groups) - counts["consistent_ndcs"],
        "consistent_source_records": counts["consistent_source_records"],
        "pending_records": counts["pending_records"],
        "pending_missing_ingredients": counts["pending_missing_ingredient_details_ndcs"],
        "pending_missing_identity": counts["pending_missing_product_identity_ndcs"],
        "pending_conflicting_fields": counts["pending_conflicting_record_fields_ndcs"],
    }

    assert (
        summary["consistent_source_records"] + summary["pending_records"]
        == summary["source_records"]
    )
    assert summary["consistent_ndcs"] + summary["pending_ndcs"] == len(groups)

    (report_dir / "otc_catalog_consistency_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
