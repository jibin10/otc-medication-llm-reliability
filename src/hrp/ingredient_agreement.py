# Compare FDA ingredient names with mapped RxNorm ingredients

import csv
import json
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

from hrp import FINAL_BENCHMARK_VERSION
from hrp.data_cleaning import formulation_key, normalize_name, quantity_key


def normalized_name(value: object) -> str:
    """Normalize case and spacing without assuming chemical synonyms."""
    if not isinstance(value, str):
        return ""

    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def fda_ingredient_names(raw_json: str) -> frozenset[str] | None:
    """Return names only when every listed ingredient has a name."""
    items = json.loads(raw_json or "null")

    if not isinstance(items, list) or not items:
        return None

    if not all(isinstance(item, dict) for item in items):
        return None

    names = [normalized_name(item.get("name")) for item in items]

    return frozenset(names) if all(names) else None


def rxnorm_names(raw_json: str) -> frozenset[str]:
    """Read RxNorm ingredient names without treating blanks as ingredients."""
    values = json.loads(raw_json or "[]")

    if not isinstance(values, list):
        raise TypeError("Expected a list of RxNorm ingredient names")

    return frozenset(name for value in values if (name := normalized_name(value)))


def compare_ingredient_names(
    fda_names: frozenset[str],
    base_names: frozenset[str],
    precise_names: frozenset[str],
) -> str:
    """Classify name agreement without inferring chemical equivalence."""
    if fda_names == base_names:
        return "exact_base_names"

    if precise_names and fda_names == precise_names and len(precise_names) == len(base_names):
        return "exact_precise_names"

    if fda_names & (base_names | precise_names):
        return "review_partial_name_overlap"

    return "review_no_exact_name_overlap"


def extract_label_ingredients(texts: list, names: list[str]) -> list[dict]:
    """Accept only a complete short ingredient list with explicit quantities."""
    text = normalize_name(" ".join(texts))
    text = re.sub(r"\((w/w|w/v|v/v)\)", r"\1", text)
    text = re.sub(r"\([^()]*\)", " ", text)
    text = re.sub(r"^active ingredients?\s*", "", text).strip()
    names = sorted(
        {normalize_name(n) for n in names if normalize_name(n)}, key=lambda n: (-len(n), n)
    )
    if not names:
        return []
    number = r"(?:\d+(?:\.\d*)?|\.\d+)"
    pattern = re.compile(
        r"(?<!\w)("
        + "|".join(re.escape(n) for n in names)
        + r")\s*[. ]*\s*("
        + number
        + r"\s*(?:mg|mcg|ug|g|ml|iu|%)(?:\s*(?:w/w|w/v|v/v)|\s*/\s*"
        + number
        + r"?\s*(?:mg|g|ml|l|1))?)(?![\w/])"
    )
    found = [{"name": m[1], "strength": m[2].strip()} for m in pattern.finditer(text)]
    remainder = pattern.sub(" ", text)
    if re.sub(r"[\s,.*\-]+", "", remainder) or not found:
        return []
    if len({x["name"] for x in found}) != len(found):
        return []
    return found


def mixed_literal_agreement(names: frozenset, options: list) -> bool:
    names = frozenset(map(normalized_name, names))
    if len(names) != len(options):
        return False
    choices = [set(map(normalized_name, values)) & names for values in options]

    def match(remaining, available):
        if not remaining:
            return not available
        return any(
            match(remaining[1:], available - {name}) for name in sorted(remaining[0] & available)
        )

    return match(sorted(choices, key=len), set(names))


def integrate_product(
    row: dict, label: dict, mapping: dict, source_conflict: bool = False, drug: dict | None = None
) -> dict:
    original = json.loads(row.get("active_ingredients_json") or "[]") or []
    metadata = json.loads(label.get("label_metadata_json") or "{}")
    label_text = json.loads(label.get("active_ingredient_json") or "[]")
    names = [x.get("name", "") for x in original if isinstance(x, dict)]
    extracted = extract_label_ingredients(label_text, names + metadata.get("substance_name", []))
    original_names = fda_ingredient_names(json.dumps(original))
    label_names = fda_ingredient_names(json.dumps(extracted))
    ingredients = original if original_names else extracted
    name_set = original_names or label_names
    reasons = []
    if source_conflict:
        reasons.append("conflicting_ndc_rows")
    if not normalize_name(row.get("display_name")):
        reasons.append("missing_identity")
    if not name_set:
        reasons.append("missing_ingredient_identity")
    if original_names and len(original_names) != len(original):
        reasons.append("repeated_ingredient_name")
    if normalize_name(row.get("dosage_form")) == "kit":
        reasons.append("kit_requires_component_identity")
    if original_names and label_names and original_names != label_names:
        reasons.append("fda_label_ingredient_conflict")
    rx_status = mapping.get("ingredient_status", "")
    rx_agreement = "unavailable"
    if name_set and rx_status == "resolved_path":
        rx_agreement = compare_ingredient_names(
            name_set,
            rxnorm_names(mapping.get("in_names_json", "[]")),
            rxnorm_names(mapping.get("pin_names_json", "[]")),
        )
        if not rx_agreement.startswith("exact_") and mixed_literal_agreement(
            name_set, json.loads(mapping.get("identity_options_json") or "[]")
        ):
            rx_agreement = "exact_mixed_literal_names"
        if not rx_agreement.startswith("exact_"):
            reasons.append("rxnorm_ingredient_names_unresolved")
    strength_notes = []
    drug_name = normalize_name((drug or {}).get("rxnorm_drug_name", ""))
    route = json.loads(row.get("route") or "[]") or []
    source_route = set(map(normalize_name, route))
    for term, allowed in {
        "rectal": {"rectal"},
        "vaginal": {"vaginal"},
        "nasal": {"nasal"},
        "ophthalmic": {"ophthalmic"},
        "otic": {"otic"},
        "oral": {"oral"},
        "topical": {"topical", "cutaneous", "percutaneous", "transdermal"},
    }.items():
        if (
            re.search(r"\b" + term + r"\b", drug_name)
            and source_route
            and not source_route & allowed
        ):
            strength_notes.append("rxnorm_route_or_form_conflict")
    dosage = normalize_name(row.get("dosage_form"))
    for form in (
        "tablet",
        "capsule",
        "cream",
        "ointment",
        "gel",
        "solution",
        "suspension",
        "patch",
    ):
        if re.search(r"\b" + form + r"\b", drug_name) and dosage and form not in dosage:
            strength_notes.append("rxnorm_route_or_form_conflict")
    for item in original:
        ingredient = re.escape(normalize_name(item.get("name")))
        match = re.search(
            r"(?<!\w)"
            + ingredient
            + r"\s+([\d.]+\s*(?:mg|mcg|g|ml)(?:\s*/\s*(?:[\d.]+\s*)?(?:g|mg|ml))?)(?![a-z/])",
            drug_name,
        )
        native = quantity_key(item.get("strength", ""))
        mapped = quantity_key(match[1]) if match else None
        if native and mapped and native != mapped:
            strength_notes.append(
                "rxnorm_strength_conflict"
                if native[1] == mapped[1]
                else "rxnorm_strength_basis_unresolved"
            )
    if original_names and label_names == original_names:
        by_name = {x["name"]: x for x in extracted}
        for item in original:
            named = by_name[normalize_name(item["name"])]
            native, stated = quantity_key(item.get("strength", "")), quantity_key(named["strength"])
            if native and stated and native[1] == stated[1] and native != stated:
                strength_notes.append("fda_label_strength_conflict")
            elif native != stated:
                strength_notes.append("label_strength_basis_unresolved")
    if not original_names and label_names:
        strength_notes.append("complete_label_list_supplements_missing_ndc_ingredients")
    accepted = not reasons
    tier = (
        "SOURCE_CONFLICT"
        if any("conflict" in r for r in reasons)
        else "NAME_NORMALIZATION_UNRESOLVED"
        if "rxnorm_ingredient_names_unresolved" in reasons
        else "UNKNOWN"
        if not name_set
        else "FDA_RXNORM_AGREEMENT"
        if rx_agreement.startswith("exact_")
        else "FDA_ONLY"
    )
    form_key = formulation_key(ingredients, row.get("dosage_form", ""), route)
    formulation_eligible = (
        accepted
        and form_key is not None
        and not any("conflict" in note or "unresolved" in note for note in strength_notes)
    )
    return {
        **row,
        "active_ingredients_json": json.dumps(ingredients),
        "original_ingredients_json": json.dumps(original),
        "label_extracted_ingredients_json": json.dumps(extracted),
        "ingredient_origin": "ndc_structured" if original_names else "label_complete_list",
        "ingredient_eligible": str(accepted),
        "formulation_eligible": str(formulation_eligible),
        "support_tier": tier,
        "rxnorm_name_agreement": rx_agreement,
        "rxnorm_product_rxcui": (drug or {}).get("rxnorm_rxcui", ""),
        "rxnorm_in_rxcuis_json": mapping.get("in_rxcuis_json", "[]"),
        "rxnorm_pin_rxcuis_json": mapping.get("pin_rxcuis_json", "[]"),
        "rxnorm_identity_options_json": mapping.get("identity_options_json", "[]"),
        "label_identity_support": "complete_list_agreement"
        if label_names == original_names and label_names
        else "complete_list_available"
        if label_names
        else "not_extractable",
        "exclusion_reasons_json": json.dumps(reasons),
        "strength_issues_json": json.dumps(
            sorted(set(strength_notes)) or ([] if form_key else ["unresolved_strength_or_form"])
        ),
        "label_id": label.get("label_id", ""),
        "label_set_id": label.get("label_set_id", ""),
        "label_effective_time": label.get("effective_time", ""),
        "label_active_ingredient_json": json.dumps(label_text),
    }


def add_fda_ingredient_identities(
    products: list[dict], output_dir: Path, mappings: dict | None = None
) -> dict:
    """Resolve source substance identities without pairing unordered UNII arrays."""
    by_name = defaultdict(set)
    supporters = defaultdict(list)
    product_uniis = {}
    for product in products:
        uniis = set()
        for source in json.loads(product["source_metadata_json"]):
            uniis.update(
                value
                for value in (source.get("openfda") or {}).get("unii", [])
                if re.fullmatch(r"[A-Z0-9]{10}", value)
            )
        product_uniis[product["product_ndc"]] = uniis
        ingredients = json.loads(product["active_ingredients_json"])
        if (
            len(ingredients) == 1
            and len(uniis) == 1
            and product["support_tier"] != "SOURCE_CONFLICT"
            and product["ingredient_origin"] == "ndc_structured"
        ):
            name, unii = normalize_name(ingredients[0]["name"]), next(iter(uniis))
            by_name[name].add(unii)
            supporters[(name, unii)].append(product["product_ndc"])
    rows = [
        {"ingredient_name": name, "unii": unii, "source_ndcs_json": json.dumps(sorted(ndcs))}
        for (name, unii), ndcs in sorted(supporters.items())
    ]
    with (output_dir / "fda_ingredient_identifiers.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["ingredient_name", "unii", "source_ndcs_json"])
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter()
    for product in products:
        ingredients = json.loads(product["active_ingredients_json"])
        own = product_uniis[product["product_ndc"]]
        names = [normalize_name(item["name"]) for item in ingredients]
        resolved = []
        if len(names) == 1 and len(own) == 1:
            resolved = [next(iter(own))]
            status = "FDA_UNII_SINGLE_INGREDIENT"
        elif names and all(len(by_name[name]) == 1 for name in names):
            proposed = [next(iter(by_name[name])) for name in names]
            if len(set(proposed)) != len(names) or (own and set(proposed) != own):
                status = "FDA_UNII_SET_UNRESOLVED"
            else:
                resolved = proposed
                status = "FDA_UNII_SET_VALIDATED" if own else "FDA_UNII_NAME_CROSSWALK"
        else:
            status = "FDA_NAME_ONLY" if not own else "FDA_UNII_SET_UNRESOLVED"
        known_links = {next(iter(by_name[name])) for name in names if len(by_name[name]) == 1}
        known_values = [next(iter(by_name[name])) for name in names if len(by_name[name]) == 1]
        complete_set = (
            bool(names)
            and len(own) == len(names)
            and known_links <= own
            and len(known_values) == len(set(known_values))
        )
        if not resolved and complete_set:
            status = "FDA_UNII_SET_WITHOUT_NAME_ALIGNMENT"
        for index, item in enumerate(ingredients):
            item["identity"] = (
                "fda_unii:" + resolved[index] if resolved else normalize_name(item["name"])
            )
        product["active_ingredients_json"] = json.dumps(ingredients)
        product["identity_status"] = status
        chemical_ids = {
            "fda_unii:" + value
            for value in (set(resolved) if resolved else own if complete_set else set())
        }
        mapping = (mappings or {}).get(product["product_ndc"], {})
        unii_options = json.loads(mapping.get("unii_options_json") or "[]")
        identifier_support = bool(chemical_ids) and mixed_literal_agreement(
            frozenset(value.removeprefix("fda_unii:") for value in chemical_ids), unii_options
        )
        product["rxnorm_identifier_status"] = (
            "FDA_UNII_PATH_SUPPORTED" if identifier_support else "NOT_CORROBORATED"
        )
        product["rxnorm_unii_options_json"] = json.dumps(unii_options)
        reasons = json.loads(product.get("exclusion_reasons_json") or "[]")
        if identifier_support and reasons == ["rxnorm_ingredient_names_unresolved"]:
            product["ingredient_eligible"] = "True"
            product["exclusion_reasons_json"] = "[]"
            product["support_tier"] = "FDA_RXNORM_IDENTIFIER_PATH"
            product["formulation_eligible"] = str(
                bool(resolved)
                and formulation_key(
                    ingredients, product["dosage_form"], json.loads(product["route"] or "[]") or []
                )
                is not None
                and not any(
                    "unresolved" in issue or "conflict" in issue
                    for issue in json.loads(product["strength_issues_json"])
                )
            )
        product["ingredient_identities_json"] = json.dumps(sorted(chemical_ids or set(names)))
        product["comparison_eligible"] = str(
            product["ingredient_eligible"] == "True" and bool(chemical_ids)
        )
        product["identity_provenance_json"] = json.dumps(
            {
                "product_uniis": sorted(own),
                "basis": status,
                "crosswalk_csv": "fda_ingredient_identifiers.csv",
                "crosswalk_evidence": [
                    {"name": name, "unii": unii, "supporting_ndc": min(supporters[(name, unii)])}
                    for name, unii in zip(names, resolved)
                    if supporters[(name, unii)]
                ],
            }
        )
        if not resolved:
            product["formulation_eligible"] = "False"
            issues = json.loads(product["strength_issues_json"]) + [
                "ingredient_identity_not_normalized"
            ]
            product["strength_issues_json"] = json.dumps(sorted(set(issues)))
        counts[status] += 1
    return {
        "statuses": dict(counts),
        "ingredient_records_recovered_by_identifier_paths": sum(
            p["support_tier"] == "FDA_RXNORM_IDENTIFIER_PATH" for p in products
        ),
        "singleton_name_unii_links": len(rows),
        "ambiguous_name_links": sum(len(values) > 1 for values in by_name.values()),
        "comparison_eligible": sum(p["comparison_eligible"] == "True" for p in products),
        "rule": "Use a product singleton UNII directly. A complete product UNII set supports unordered "
        "ingredient comparison when cardinality matches and known name links do not contradict it. "
        "Unknown name alignment still withholds formulation quantities. For multiple ingredients use unique singleton-derived "
        "name mappings only when they bijectively cover the product UNII set, when present. "
        "Never pair unordered arrays by position. Equal UNII identifiers support aliases, "
        "without inferring salt, moiety, density or clinical equivalence. Unknown identifiers "
        "remain eligible for source-name tasks but cannot establish disjoint chemical sets.",
    }


def run_integration(project_root: Path, source_dir: Path) -> dict:
    """Reconcile all OTC products and retain task-specific eligibility."""
    from hrp.data_cleaning import build_consistent_otc_catalog, prepare_otc_catalog
    from hrp.data_loading import _iter_fda_records, file_sha256, find_sources, verify_sources
    from hrp.ingredient_mapping import link_rxnorm_products, match_fda_labels, ndc_product_key
    from hrp.rxnorm_ingredients import map_rxnorm_ingredients

    interim = project_root / "data/interim"
    out = interim / FINAL_BENCHMARK_VERSION
    reports = project_root / "results" / FINAL_BENCHMARK_VERSION / "part02"
    out.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    sources = find_sources(source_dir)
    raw = interim / "otc_ndc_raw.jsonl"
    source_audit = verify_sources(source_dir, project_root / "results/final_v2/part01")
    signature = {e["path"]: e["sha256"] for e in source_audit["files"]}
    if not (reports / "coverage.json").exists():
        prior_report = project_root / "results/final_v2/part02/coverage.json"
        if prior_report.exists():
            prior_links = json.loads(prior_report.read_text())
            if prior_links.get("source_files") == signature:
                for name, digest in prior_links["files"].items():
                    source = interim / "final_v2" / name
                    if (
                        name != "catalog_evidence.csv"
                        and source.exists()
                        and file_sha256(source) == digest
                    ):
                        shutil.copyfile(source, out / name)
                for source in prior_report.parent.glob("*.json"):
                    shutil.copyfile(source, reports / source.name)
    code_signature = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in (
            "ingredient_agreement.py",
            "ingredient_mapping.py",
            "data_cleaning.py",
            "rxnorm_ingredients.py",
        )
    }
    cached_path = reports / "coverage.json"
    prior = json.loads(cached_path.read_text()) if cached_path.exists() else {}
    if (
        prior.get("source_files") == signature
        and prior.get("code") == code_signature
        and raw.exists()
        and prior.get("raw_cache_sha256") == file_sha256(raw)
        and all(
            (out / name).exists() and file_sha256(out / name) == digest
            for name, digest in prior["files"].items()
        )
    ):
        print("Reused complete catalog after source and artifact hash validation", flush=True)
        return prior
    if not raw.exists():
        with raw.open("w", encoding="utf-8") as destination:
            for source in sources.ndc:
                for record in _iter_fda_records(source):
                    if record.get("product_type") == "HUMAN OTC DRUG":
                        destination.write(
                            json.dumps(record, ensure_ascii=False, default=str) + "\n"
                        )
    original_by_ndc = defaultdict(list)
    cached = Counter()
    raw_count = 0
    with raw.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            cached[
                sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
            ] += 1
            original_by_ndc[record["product_ndc"]].append(
                {
                    k: record.get(k)
                    for k in (
                        "product_id",
                        "spl_id",
                        "openfda",
                        "labeler_name",
                        "brand_name_base",
                        "brand_name_suffix",
                        "marketing_start_date",
                        "marketing_end_date",
                        "listing_expiration_date",
                        "finished",
                        "packaging",
                    )
                }
            )
            raw_count += 1
    counts, actual = Counter(), Counter()
    packages = Counter()
    invalid_packages = []
    identifiers = Counter()
    product_ids = set()
    for path in sources.ndc:
        for record in _iter_fda_records(path):
            counts[record.get("product_type") or "missing_product_type"] += 1
            identifier = record.get("product_id")
            if not identifier:
                identifiers["missing_product_id"] += 1
            else:
                identifiers["duplicate_product_id"] += identifier in product_ids
                product_ids.add(identifier)
            expected = str(record.get("product_ndc")) + "_" + str(record.get("spl_id"))
            identifiers[
                "product_id_matches_ndc_and_spl"
                if identifier == expected
                else "product_id_relationship_unresolved"
            ] += 1
            if record.get("product_type") == "HUMAN OTC DRUG":
                actual[
                    sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()
                ] += 1
            for package in record.get("packaging", []):
                ndc = package.get("package_ndc", "")
                packages[
                    "valid_parent"
                    if ndc.rsplit("-", 1)[0] == record.get("product_ndc")
                    else "invalid_parent"
                ] += 1
                if ndc.rsplit("-", 1)[0] != record.get("product_ndc"):
                    invalid_packages.append(
                        {
                            "product_ndc": record.get("product_ndc"),
                            "package_ndc": ndc,
                            "reason": "package_parent_does_not_match_product_ndc",
                        }
                    )
    if actual != cached:
        raise ValueError("Saved OTC rows do not exactly match the original source")
    del actual, cached
    prepare_otc_catalog(raw, out / "otc_catalog_stage.csv", reports)
    consistency = build_consistent_otc_catalog(out / "otc_catalog_stage.csv", out, reports)
    old_coverage = reports / "coverage.json"
    cached_coverage = json.loads(old_coverage.read_text()) if old_coverage.exists() else {}
    label_file = out / "otc_ndc_label_links.csv"
    if (
        label_file.exists()
        and file_sha256(label_file) == cached_coverage.get("files", {}).get(label_file.name)
        and cached_coverage.get("source_files", signature) == signature
    ):
        label_summary = cached_coverage["label_linkage"]
        print("Reused hash-verified complete label linkage", flush=True)
    else:
        label_summary = match_fda_labels(
            raw,
            out / "otc_catalog_consistent.csv",
            out / "otc_catalog_pending_review.csv",
            sources.labels[0].parent,
            out,
            reports,
        )
    rxnorm_dir = sources.rxnorm[0].parent
    reuse_rxnorm = all(
        (out / name).exists()
        and file_sha256(out / name) == cached_coverage.get("files", {}).get(name)
        for name in ("otc_rxnorm_product_links.csv", "otc_rxnorm_ingredient_links.csv")
    )
    if (out / "otc_rxnorm_ingredient_links.csv").exists():
        with (out / "otc_rxnorm_ingredient_links.csv").open(encoding="utf-8") as stream:
            reuse_rxnorm = reuse_rxnorm and "unii_options_json" in stream.readline()
    if reuse_rxnorm and cached_coverage.get("source_files", signature) == signature:
        rx_summary = cached_coverage["rxnorm_linkage"]
        print("Reused hash-verified full-release RxNorm links", flush=True)
    else:
        rx_summary = link_rxnorm_products(out / "otc_ndc_label_links.csv", rxnorm_dir, out, reports)
        map_rxnorm_ingredients(out / "otc_rxnorm_product_links.csv", rxnorm_dir, out, reports)
    audit_ingredient_agreement(
        out / "otc_catalog_stage.csv",
        out / "otc_rxnorm_ingredient_links.csv",
        out / "otc_ndc_label_links.csv",
        out,
        reports,
    )

    def indexed(filename):
        with (out / filename).open(encoding="utf-8", newline="") as stream:
            return {r["product_ndc"]: r for r in csv.DictReader(stream)}

    labels = indexed("otc_ndc_label_links.csv")
    mappings = indexed("otc_rxnorm_ingredient_links.csv")
    drugs = indexed("otc_rxnorm_product_links.csv")
    staged = defaultdict(list)
    with (out / "otc_catalog_stage.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            staged[row["product_ndc"]].append(row)
    products = []
    for ndc, rows in sorted(staged.items()):
        from hrp.data_cleaning import route_signature

        keys = []
        for r in rows:
            items = json.loads(r["active_ingredients_json"] or "null") or []
            ingredients = (
                tuple(
                    sorted(
                        (normalize_name(i.get("name")), normalize_name(i.get("strength")))
                        for i in items
                    )
                )
                or None
            )
            keys.append(
                (
                    normalize_name(r["normalized_name"]) or None,
                    normalize_name(r["dosage_form"]) or None,
                    route_signature(json.loads(r["route"] or "null")),
                    ingredients,
                )
            )
        conflicts = any(len({key[i] for key in keys if key[i] is not None}) > 1 for i in range(4))
        selected = min(
            rows,
            key=lambda r: (
                not bool(fda_ingredient_names(r["active_ingredients_json"])),
                int(r["source_row"]),
            ),
        )
        selected["source_rows"] = ",".join(r["source_row"] for r in rows)
        product = integrate_product(selected, labels[ndc], mappings[ndc], conflicts, drugs[ndc])
        originals = original_by_ndc[ndc]
        product["source_metadata_json"] = json.dumps(
            [
                {
                    k: r.get(k)
                    for k in (
                        "product_id",
                        "spl_id",
                        "openfda",
                        "labeler_name",
                        "brand_name_base",
                        "brand_name_suffix",
                        "marketing_start_date",
                        "marketing_end_date",
                        "listing_expiration_date",
                        "finished",
                        "packaging",
                    )
                }
                for r in originals
            ]
        )
        if not ndc_product_key(ndc):
            reasons = json.loads(product["exclusion_reasons_json"]) + ["invalid_product_ndc"]
            product.update(
                ingredient_eligible="False",
                formulation_eligible="False",
                exclusion_reasons_json=json.dumps(reasons),
            )
        products.append(product)
    identities = add_fda_ingredient_identities(products, out, mappings)
    with (out / "catalog_evidence.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(products[0]))
        writer.writeheader()
        writer.writerows(products)
    reasons = Counter(r for p in products for r in json.loads(p["exclusion_reasons_json"]))
    summary = {
        "version": FINAL_BENCHMARK_VERSION,
        "ingredient_identity_normalization": identities,
        "ndc_records": sum(counts.values()),
        "product_types": dict(counts),
        "otc_source_records": raw_count,
        "unique_otc_ndcs": len(products),
        "legacy_consistency": consistency,
        "source_cache_exactly_verified": True,
        "package_relationships": dict(packages),
        "invalid_package_relationships": invalid_packages,
        "identifier_relationships": dict(identifiers),
        "label_linkage": label_summary,
        "rxnorm_linkage": rx_summary,
        "ingredient_support_tiers": dict(Counter(p["support_tier"] for p in products)),
        "ingredient_eligible": sum(p["ingredient_eligible"] == "True" for p in products),
        "formulation_eligible": sum(p["formulation_eligible"] == "True" for p in products),
        "exclusion_reasons_nonexclusive": dict(reasons),
        "exclusive_outcomes": dict(
            Counter(
                "accepted_ingredient"
                if p["ingredient_eligible"] == "True"
                else json.loads(p["exclusion_reasons_json"])[0]
                for p in products
            )
        ),
        "strength_issues_nonexclusive": dict(
            Counter(r for p in products for r in json.loads(p["strength_issues_json"]))
        ),
        "label_identity_support": dict(Counter(p["label_identity_support"] for p in products)),
        "ndc_record_outcomes": {
            "not_applicable_non_human_otc": sum(counts.values()) - raw_count,
            "accepted_source_rows": sum(
                len(p["source_rows"].split(","))
                for p in products
                if p["ingredient_eligible"] == "True"
            ),
            "unresolved_source_rows": sum(
                len(p["source_rows"].split(","))
                for p in products
                if p["ingredient_eligible"] != "True"
            ),
        },
        "source_policy": "Complete FDA ingredient names support ingredient tasks without RxNorm. "
        "Unresolved mapped names and submitted-source contradictions remain outside reference labels. "
        "Label text supplements only a completely parsed list. FDA sources are correlated submissions. "
        "Strength and form uncertainty does not erase known ingredient identities.",
        "human_approval_inferred": False,
        "source_files": signature,
        "code": code_signature,
        "raw_cache_sha256": file_sha256(raw),
        "files": {
            p.name: file_sha256(p)
            for p in sorted(out.glob("*.csv"))
            if p.name not in {"full_review_queue.csv", "manual_review_sample.csv"}
        },
    }
    assert sum(summary["exclusive_outcomes"].values()) == len(products)
    (reports / "coverage.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "files"}, indent=2), flush=True)
    return summary


def audit_ingredient_agreement(
    stage_path: Path,
    rxnorm_path: Path,
    label_path: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict:
    """Compare ingredient names for all OTC NDCs without dropping unknowns."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Keep all reported ingredient sets for repeated NDC identifiers
    fda_sets = defaultdict(set)

    with stage_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            ndc = row["product_ndc"].strip()

            fda_sets[ndc].add(fda_ingredient_names(row["active_ingredients_json"]))

    # FDA label text is supporting evidence, not a normalized ingredient list
    label_data = {}

    with label_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            ndc = row["product_ndc"].strip()

            if ndc in label_data:
                raise ValueError(f"Duplicate label link for {ndc}")

            label_data[ndc] = row

    output_path = output_dir / "otc_ingredient_agreement.csv"

    columns = [
        "product_ndc",
        "catalog_status",
        "agreement_status",
        "fda_status",
        "fda_names_json",
        "rxnorm_ingredient_status",
        "rxnorm_in_rxcuis_json",
        "rxnorm_in_names_json",
        "rxnorm_pin_rxcuis_json",
        "rxnorm_pin_names_json",
        "fda_label_status",
        "label_text_available",
    ]

    counts = Counter()
    seen = set()

    with (
        rxnorm_path.open(newline="", encoding="utf-8") as source,
        output_path.open("w", newline="", encoding="utf-8") as destination,
    ):
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()

        for row in csv.DictReader(source):
            ndc = row["product_ndc"].strip()

            if ndc in seen:
                raise ValueError(f"Repeated RxNorm product link for {ndc}")

            seen.add(ndc)
            source_sets = fda_sets.get(ndc)

            # Missing and inconsistent FDA records cannot establish agreement
            if not source_sets or source_sets == {None}:
                fda_status = "missing"
                fda_names = frozenset()

            elif len(source_sets) != 1:
                fda_status = "inconsistent_source_rows"
                fda_names = frozenset()

            else:
                fda_status = "usable"
                fda_names = next(iter(source_sets))

            counts[f"fda_{fda_status}"] += 1

            ingredient_status = row["ingredient_status"]

            if ingredient_status == "resolved_path":
                counts["rxnorm_resolved"] += 1

            if ingredient_status != "resolved_path":
                status = "rxnorm_unavailable"

            elif fda_status != "usable":
                status = "fda_unavailable"

            else:
                base = rxnorm_names(row["in_names_json"])
                precise = rxnorm_names(row["pin_names_json"])

                if not base:
                    raise ValueError(f"Resolved RxNorm path has no IN names: {ndc}")

                counts["both_comparable"] += 1

                status = compare_ingredient_names(fda_names, base, precise)

                if status.startswith("exact_"):
                    counts[f"exact_{row['catalog_status']}"] += 1

            counts[status] += 1

            label = label_data.get(ndc, {})
            label_text = json.loads(label.get("active_ingredient_json") or "[]")

            has_label_text = isinstance(label_text, list) and any(
                isinstance(value, str) and value.strip() for value in label_text
            )

            if has_label_text and status.startswith("exact_"):
                counts["exact_with_label_text"] += 1

            writer.writerow(
                {
                    "product_ndc": ndc,
                    "catalog_status": row["catalog_status"],
                    "agreement_status": status,
                    "fda_status": fda_status,
                    "fda_names_json": json.dumps(sorted(fda_names)),
                    "rxnorm_ingredient_status": ingredient_status,
                    "rxnorm_in_rxcuis_json": row["in_rxcuis_json"],
                    "rxnorm_in_names_json": row["in_names_json"],
                    "rxnorm_pin_rxcuis_json": row["pin_rxcuis_json"],
                    "rxnorm_pin_names_json": row["pin_names_json"],
                    "fda_label_status": label.get("link_status", "missing"),
                    "label_text_available": has_label_text,
                }
            )

    # Every source must refer to the same set of product NDCs
    if seen != set(fda_sets) or seen != set(label_data):
        raise ValueError("FDA, RxNorm and label NDC cohorts do not match")

    summary = {
        "otc_ndcs": len(seen),
        "rxnorm_resolved": counts["rxnorm_resolved"],
        "fda_usable": counts["fda_usable"],
        "fda_missing": counts["fda_missing"],
        "fda_inconsistent_source_rows": counts["fda_inconsistent_source_rows"],
        "both_comparable": counts["both_comparable"],
        "exact_base_names": counts["exact_base_names"],
        "exact_precise_names": counts["exact_precise_names"],
        "review_partial_name_overlap": counts["review_partial_name_overlap"],
        "review_no_exact_name_overlap": counts["review_no_exact_name_overlap"],
        "rxnorm_unavailable": counts["rxnorm_unavailable"],
        "fda_unavailable": counts["fda_unavailable"],
        "exact_names_consistent_catalog": counts["exact_consistent"],
        "exact_names_pending_catalog": counts["exact_pending"],
        "exact_names_with_label_text": counts["exact_with_label_text"],
    }

    comparable = counts["both_comparable"]
    exact = counts["exact_base_names"] + counts["exact_precise_names"]

    summary["exact_name_rate_among_comparable_pct"] = (
        round(100 * exact / comparable, 2) if comparable else 0.0
    )

    # Confirm that every NDC received exactly one comparison status
    assert counts["rxnorm_unavailable"] + counts["fda_unavailable"] + counts[
        "exact_base_names"
    ] + counts["exact_precise_names"] + counts["review_partial_name_overlap"] + counts[
        "review_no_exact_name_overlap"
    ] == len(seen)

    (report_dir / "otc_ingredient_agreement_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary
