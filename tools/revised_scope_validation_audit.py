"""Read-only Stage 5 audit; only train/validation cases, never sealed test data.

Run from the repository root using the existing environment:
    .venv/Scripts/python.exe tools/revised_scope_validation_audit.py --repo .

Only this script, its tests, and results/revised_scope_audit/ are new artifacts.
Name similarity and label parsing are screening signals, not human adjudication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from zipfile import ZipFile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from rapidfuzz import fuzz, process

from hrp.data_cleaning import formulation_key, normalize_name, quantity_key
from hrp.evidence_adapter import build_evidence_payload
from hrp.retrieval_baseline import (
    SOURCE_FILES,
    _descriptions,
    build_catalog,
    resolve_product,
    retrieve_evidence,
)

SEED = 42
MAX_CANDIDATES = 8
MAX_LABEL_CHARS = 1200
NEAR_NAME_THRESHOLD = 90
OUTPUT_NAMES = (
    "scope_counts.json",
    "quality_and_conflicts.json",
    "family_group_diagnostics.json",
    "validation_review_queue.PRIVATE.csv",
    "REVISED_STAGE5_FINDINGS.md",
)
# These quotas and rules are fixed before scanning the real case data or any model output.
REVIEW_QUOTAS = {
    "identical_name_different_ingredients": 20,
    "similar_name_different_ingredients": 25,
    "same_ingredient_controls": 15,
    "name_only_ambiguity": 10,
    "source_disagreement": 15,
    "conventional_oral_therapeutic": 15,
}
PRIMARY_RELATIONS = {
    "identical_normalized_name",
    "shared_distinctive_brand_variant_candidate",
    "highly_similar_name_candidate",
}
GENERIC_TOKENS = {
    "extra",
    "maximum",
    "regular",
    "strength",
    "relief",
    "pain",
    "cough",
    "cold",
    "flu",
    "allergy",
    "sinus",
    "congestion",
    "daytime",
    "nighttime",
    "night",
    "day",
    "severe",
    "multi",
    "symptom",
    "children",
    "childrens",
    "child",
    "adult",
    "adults",
    "baby",
    "infant",
    "infants",
    "junior",
    "plus",
    "pm",
    "dm",
    "pe",
    "non",
    "drowsy",
    "drowsiness",
    "free",
    "sugar",
    "dye",
    "liquid",
    "tablets",
    "tablet",
    "caplets",
    "caplet",
    "capsule",
    "capsules",
    "softgels",
    "softgel",
    "gel",
    "cream",
    "ointment",
    "spray",
    "oral",
    "topical",
    "solution",
    "suspension",
    "extended",
    "release",
    "coated",
    "chewable",
    "medicine",
    "medication",
    "first",
    "aid",
    "antiseptic",
    "antibiotic",
    "anti",
    "itch",
    "nasal",
    "decongestant",
    "rub",
    "roll",
    "on",
    "cap",
    "spf",
    "sunscreen",
    "sun",
    "screen",
    "broad",
    "spectrum",
    "lip",
    "balm",
    "skin",
    "protectant",
    "hand",
    "sanitizer",
    "alcohol",
    "homeopathic",
    "natural",
    "original",
    "formula",
    "advanced",
    "therapeutic",
    "complete",
    "total",
    "fast",
    "acting",
    "store",
    "pharmacy",
    "pharma",
    "health",
    "healthy",
    "healthcare",
    "care",
    "wellness",
    "quality",
    "value",
    "family",
    "good",
    "sense",
    "signature",
    "select",
    "choice",
    "assured",
    "daily",
    "essential",
    "essentials",
    "drug",
    "drugs",
    "brand",
    "generic",
    "with",
    "and",
    "for",
    "the",
    "of",
    "to",
    "in",
    "mg",
    "mcg",
    "ml",
    "percent",
    "usa",
    "us",
    "hcl",
    "hydrochloride",
    "acid",
    "sodium",
}
RETAILER_TOKENS = {
    "cvs",
    "walgreens",
    "walgreen",
    "equate",
    "kroger",
    "rexall",
    "rite",
    "aid",
    "dollar",
    "dollarcare",
    "family",
    "wal",
    "mart",
    "walmart",
    "target",
    "upup",
    "publix",
    "meijer",
    "walgreenshealth",
    "hannaford",
    "safeway",
    "signature",
    "care",
    "good",
    "sense",
    "health",
    "food",
    "lion",
    "dg",
    "harris",
    "teeter",
    "kirkland",
    "costco",
}
ORAL_THERAPEUTIC_INGREDIENTS = {
    "acetaminophen",
    "ibuprofen",
    "naproxen sodium",
    "aspirin",
    "cetirizine hydrochloride",
    "loratadine",
    "fexofenadine hydrochloride",
    "diphenhydramine hydrochloride",
    "chlorpheniramine maleate",
    "dextromethorphan hydrobromide",
    "guaifenesin",
    "phenylephrine hydrochloride",
    "pseudoephedrine hydrochloride",
    "doxylamine succinate",
}
SUNSCREEN_INGREDIENTS = {
    "avobenzone",
    "homosalate",
    "octisalate",
    "octocrylene",
    "octinoxate",
    "oxybenzone",
    "ecamsule",
    "ensulizole",
    "meradimate",
    "padimate o",
    "sulisobenzone",
}


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def name_tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).casefold()))


def name_key(text: str) -> str:
    return " ".join(name_tokens(text))


def metric(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def ensure_not_test(path: Path) -> None:
    """Fail BEFORE opening, hashing, statting, or resolving a forbidden test path."""
    parts = {part.casefold() for part in Path(path).parts}
    leaf = Path(path).name.casefold()
    if (
        "sealed_test" in leaf
        or leaf.startswith(("test.", "test_"))
        or parts & {"test", "heldout", "held_out", "sealed_test"}
    ):
        raise ValueError("Held-out test access is prohibited")


def sha256_file(path: Path) -> str:
    ensure_not_test(path)
    resolved = Path(path).resolve()
    ensure_not_test(resolved)
    with resolved.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Inputs:
    """Closed allowlist of input files. No recursive repository data discovery."""

    def __init__(self, repo: Path, source_dir: Path):
        self.repo = repo.resolve()
        self.source_dir = source_dir.resolve()
        self.allowed: set[Path] = set()
        self.hashes: dict[str, dict] = {}
        self.paths: dict[str, Path] = {}

    def register(self, path: Path) -> Path:
        ensure_not_test(path)
        path = path.resolve()
        ensure_not_test(path)
        if not (path.is_relative_to(self.repo) or path.is_relative_to(self.source_dir)):
            raise ValueError("Input lies outside the two authorized data locations")
        if path in self.allowed:
            return path
        if not path.is_file():
            raise FileNotFoundError(path)
        self.allowed.add(path)
        label = (
            path.relative_to(self.repo).as_posix()
            if path.is_relative_to(self.repo)
            else "external_data/" + path.relative_to(self.source_dir).as_posix()
        )
        self.hashes[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        self.paths[label] = path
        return path

    def open(self, path: Path, mode="r", **kwargs):
        ensure_not_test(path)
        path = path.resolve()
        ensure_not_test(path)
        if path not in self.allowed or mode not in {"r", "rb"}:
            raise ValueError("Input open rejected: not allowlisted or not read-only")
        return path.open(mode, **kwargs)

    def csv(self, path: Path):
        path = self.register(path)
        with self.open(path, encoding="utf-8-sig", newline="") as stream:
            yield from csv.DictReader(stream)

    def json(self, path: Path):
        path = self.register(path)
        with self.open(path, encoding="utf-8-sig") as stream:
            return json.load(stream)

    def verify_unchanged(self) -> bool:
        for label, path in self.paths.items():
            if sha256_file(path) != self.hashes[label]["sha256"]:
                raise RuntimeError(f"Input changed during audit: {label}")
        return True


def load_cases(inputs: Inputs) -> tuple[dict, dict]:
    rows_by_part, memberships = {}, {}
    seen = set()
    for partition in ("train", "validation"):
        rows = []
        for row in inputs.csv(inputs.repo / f"data/processed/final_v5/{partition}.csv"):
            if row["partition"] != partition or row["case_id"] in seen:
                raise ValueError("Unexpected partition or duplicated case ID")
            seen.add(row["case_id"])
            request = json.loads(row["input_json"])
            _descriptions(request)
            reference = json.loads(row["reference_json"])
            provenance = json.loads(row["provenance_json"])
            families = json.loads(row["family_group_ids_json"])
            targets = (
                [reference[k] for k in ("product_a", "product_b")]
                if row["task"] == "ingredient_comparison"
                else [reference["product"]]
                if "product" in reference
                else []
            )
            compact_targets = [
                {
                    "product_ndc": t.get("product_ndc"),
                    "formulation_supported": t.get("formulation_supported", False),
                    "ingredient_identities": t.get("ingredient_identities", []),
                }
                for t in targets
            ]
            item = {
                "case_id": row["case_id"],
                "partition": partition,
                "task": row["task"],
                "case_type": row["case_type"],
                "input": request,
                "families": families,
                "ndcs": provenance["product_ndcs"],
                "human_verified": provenance.get("human_verified") is True,
                "answer_status": reference["answer_status"],
                "targets": compact_targets,
            }
            rows.append(item)
            # The explicit-NDC inventory establishes membership without a test split manifest.
            if row["case_type"] == "explicit_ndc":
                ndc = request["product_ndc"]
                if len(families) != 1 or provenance["product_ndcs"] != [ndc]:
                    raise ValueError("Explicit-NDC membership must have one product and group")
                membership = {"partition": partition, "family": families[0]}
                if ndc in memberships and memberships[ndc] != membership:
                    raise ValueError("Product occurs in both training and validation")
                memberships[ndc] = membership
        rows_by_part[partition] = rows
    for partition, rows in rows_by_part.items():
        for row in rows:
            if any(
                n not in memberships or memberships[n]["partition"] != partition
                for n in row["ndcs"]
            ):
                raise ValueError("Case provenance exceeds the authorized partition membership")
            if {memberships[n]["family"] for n in row["ndcs"]} != set(row["families"]):
                raise ValueError("Case groups disagree with training/validation NDC membership")
    return rows_by_part, memberships


def load_sources(inputs: Inputs, memberships: dict) -> tuple[dict, dict, dict]:
    interim = inputs.repo / "data/interim/final_v5"
    frames = [list(inputs.csv(interim / name)) for name in SOURCE_FILES]
    catalog = build_catalog(*frames)
    metadata = {r["product_ndc"]: r for r in frames[0] if r["product_ndc"] in memberships}
    labels = {r["product_ndc"]: r for r in frames[1] if r["product_ndc"] in memberships}
    if set(memberships) - set(metadata):
        raise ValueError("Authorized products missing from source catalog")
    return catalog, metadata, labels


class NameScreen:
    def __init__(self, metadata: dict):
        names = {name_key(r["display_name"]) for r in metadata.values()}
        frequency = Counter(t for n in names for t in set(n.split()))
        ingredients = {
            t
            for r in metadata.values()
            for i in (json.loads(r["original_ingredients_json"] or "[]") or [])
            for t in name_tokens(i.get("name", ""))
        }
        self.excluded = GENERIC_TOKENS | RETAILER_TOKENS | ingredients
        self.distinctive = {
            t for t, n in frequency.items() if len(t) >= 4 and n <= 128 and t not in self.excluded
        }

    def relation(self, left: str, right: str) -> dict:
        a, b = name_key(left), name_key(right)
        at, bt = set(a.split()), set(b.split())
        shared = at & bt
        distinctive = shared & self.distinctive
        score = fuzz.ratio(a, b)
        if a == b:
            kind = "identical_normalized_name"
        elif distinctive and (
            len(distinctive) >= 2
            or at <= bt
            or bt <= at
            or next(iter(a.split()), "") == next(iter(b.split()), "")
        ):
            kind = "shared_distinctive_brand_variant_candidate"
        elif shared and not distinctive and shared <= self.excluded:
            kind = "generic_or_retailer_false_match_candidate"
        elif score >= NEAR_NAME_THRESHOLD:
            kind = "highly_similar_name_candidate"
        else:
            kind = "unrelated_co_partitioned_name_candidate"
        return {
            "class": kind,
            "character_similarity": round(score, 4),
            "shared_distinctive_tokens": sorted(distinctive),
            "retailer_tokens_shared": bool(shared & RETAILER_TOKENS),
            "verified_brand_relationship": False,
        }


def product_category(product: dict, row: dict) -> str:
    names = {normalize_name(i["name"]) for i in product["active_ingredients"]}
    marketing = normalize_name(row.get("marketing_category", ""))
    display = name_key(product["name"])
    route = {normalize_name(r) for r in product["route"] or []}
    form = normalize_name(product["dosage_form"])
    if "homeopath" in marketing:
        return "homeopathic"
    if "sunscreen" in display or re.search(r"\bspf\b", display) or names & SUNSCREEN_INGREDIENTS:
        return "sunscreen"
    if (
        route & {"dental", "periodontal"}
        or "dentifrice" in form
        or any(t in display for t in ("toothpaste", "mouthwash", "dental rinse"))
    ):
        return "dental_otc"
    if "oral" in route and names & ORAL_THERAPEUTIC_INGREDIENTS:
        return "conventional_oral_analgesic_cold_allergy"
    if route & {"topical", "cutaneous", "transdermal", "percutaneous"}:
        return "topical_otc"
    if "oral" in route:
        return "other_oral_otc"
    return "other_or_unclassified_otc"


def product_quality(product: dict, row: dict) -> dict:
    ingredients = product["active_ingredients"]
    ingredient_names = {name_key(i["name"]) for i in ingredients}
    display = name_key(product["name"])
    generic_like = (
        row.get("name_source") == "generic"
        or display == name_key(product.get("generic_name", ""))
        or any(display == n or display.startswith(n + " ") for n in ingredient_names if n)
    )
    identifiers = product["ingredient_identities"]
    return {
        "category": product_category(product, row),
        "ingredient_eligible": product.get("ingredient_eligible") == "True",
        "comparison_eligible": product.get("comparison_eligible") == "True",
        "formulation_eligible": product.get("formulation_eligible") == "True",
        "support_tier": product["support_tier"],
        "missing_strength": not ingredients or any(not i.get("strength") for i in ingredients),
        "unparsed_quantity_or_unit": not ingredients
        or any(quantity_key(i.get("strength", "")) is None for i in ingredients),
        "missing_form": not bool(product["dosage_form"]),
        "missing_route": not bool(product["route"]),
        "all_fda_unii_identities": bool(identifiers)
        and all(i.startswith("fda_unii:") for i in identifiers),
        "identity_status": product["identity_status"],
        "generic_like_name": generic_like,
        "retailer_name_token": bool(set(display.split()) & RETAILER_TOKENS),
        "label_text_available": bool(product["fda_label"]["active_ingredient_text"]),
        "label_identifiers_complete": bool(
            product["fda_label"]["label_id"] and product["fda_label"]["label_set_id"]
        ),
        "formulation_key": formulation_key(ingredients, product["dosage_form"], product["route"]),
    }


def summarize_case_set(infos: list[dict]) -> dict:
    ndcs = {n for r in infos for n in r["ndcs"]}
    groups = {g for r in infos for g in r["families"]}
    return {
        "cases": len(infos),
        "unique_ndcs": len(ndcs),
        "unique_partition_groups": len(groups),
        "cases_with_any_explicit_ndc": sum(r["any_ndc"] for r in infos),
        "cases_without_explicit_ndc": sum(not r["any_ndc"] for r in infos),
        "cases_by_category": dict(sorted(Counter(r["category"] for r in infos).items())),
    }


def analyze_partition(rows, catalog, memberships, qualities, name_screen, adapter=False):
    counts, by_type, infos = Counter(), defaultdict(Counter), []
    candidate_histogram = Counter()
    for row in rows:
        evidence = retrieve_evidence(row["input"], catalog)
        requests = evidence["products"]
        matched = {p["product_ndc"] for item in requests for p in item["candidates"]}
        # The full SOURCE catalog is permitted; only query-matched allowed source rows are analyzed.
        if matched != set(row["ndcs"]) or any(
            n not in memberships or memberships[n]["partition"] != row["partition"] for n in matched
        ):
            raise ValueError("Query evidence crosses authorized case provenance/partition")
        local = Counter(cases=1)
        any_ndc = any(item["description"].get("product_ndc") for item in requests)
        local["cases_with_any_explicit_ndc"] += int(any_ndc)
        local["cases_all_requests_explicit_ndc"] += int(
            all(item["description"].get("product_ndc") for item in requests)
        )
        local["cases_all_requests_single_exact_identifier_lookup"] += int(
            all(item["identifier_candidate_count"] == 1 for item in requests)
        )
        local["cases_all_requests_single_after_details"] += int(
            all(len(item["candidates"]) == 1 for item in requests)
        )
        local["cases_all_requests_name_only"] += int(
            all(set(item["description"]) == {"name"} for item in requests)
        )
        local["reference_targets"] += len(row["targets"])
        local["full_formulation_reference_targets"] += sum(
            t["formulation_supported"] for t in row["targets"]
        )
        payload = (
            build_evidence_payload(
                row["input"],
                evidence,
                max_candidates=MAX_CANDIDATES,
                max_label_chars=MAX_LABEL_CHARS,
            )
            if adapter
            else None
        )
        for index, item in enumerate(requests):
            description, candidates = item["description"], item["candidates"]
            local["product_requests"] += 1
            local["requests_with_explicit_ndc"] += int(bool(description.get("product_ndc")))
            local["requests_with_native_strengths"] += int("native_strengths" in description)
            local["requests_single_exact_identifier_lookup"] += int(
                item["identifier_candidate_count"] == 1
            )
            local["requests_single_name_lookup_without_ndc"] += int(
                not description.get("product_ndc") and item["identifier_candidate_count"] == 1
            )
            local["requests_single_after_details"] += int(len(candidates) == 1)
            resolution = resolve_product(item)
            local["request_resolution_" + resolution["status"]] += 1
            if set(description) == {"name"}:
                local["name_only_requests"] += 1
                supported_difference = (
                    len({tuple(sorted(p["ingredient_identities"])) for p in candidates}) > 1
                    or len(
                        {qualities[p["product_ndc"]]["formulation_key"] for p in candidates}
                        - {None}
                    )
                    > 1
                )
                ambiguous = resolution["status"] == "INSUFFICIENT_INFORMATION"
                local["name_only_ambiguity_with_identifier_or_supported_form_difference"] += int(
                    ambiguous and supported_difference
                )
                local["name_only_ambiguity_with_only_source_names_or_incomplete_forms"] += int(
                    ambiguous and not supported_difference
                )
            if adapter:
                view = payload["products"][index]
                candidate_histogram[len(candidates)] += 1
                local["candidate_records_full"] += len(candidates)
                local["candidate_records_shown"] += len(view["candidates"])
                local["requests_candidates_truncated"] += int(view["candidates_truncated"])
                local["requests_with_direct_native_ingredient_lists"] += int(
                    bool(view["candidates"])
                    and all(p["fda_active_ingredients"] for p in view["candidates"])
                )
                local["shown_candidate_records_with_ingredients"] += sum(
                    bool(p["fda_active_ingredients"]) for p in view["candidates"]
                )
                local["shown_candidate_label_text_truncated"] += sum(
                    p["fda_label_text_truncated"] for p in view["candidates"]
                )
                local["requests_any_label_text_truncated"] += int(
                    any(p["fda_label_text_truncated"] for p in view["candidates"])
                )
                if view["candidates_truncated"]:
                    full_sets = {tuple(sorted(p["ingredient_identities"])) for p in candidates}
                    shown_sets = {
                        tuple(sorted(p["ingredient_identities"]))
                        for p in candidates[:MAX_CANDIDATES]
                    }
                    local["truncated_requests_losing_ingredient_alternatives"] += int(
                        bool(full_sets - shown_sets)
                    )
                    local["truncated_requests_hiding_ingredient_ambiguity"] += int(
                        len(full_sets) > 1 and len(shown_sets) == 1
                    )
                    target_ndc = (
                        row["targets"][index]["product_ndc"]
                        if index < len(row["targets"])
                        else None
                    )
                    local["truncated_requests_omitting_exact_reference_ndc"] += int(
                        bool(target_ndc)
                        and target_ndc not in {p["product_ndc"] for p in view["candidates"]}
                    )
        if adapter:
            local["cases_candidates_truncated"] += int(
                any(item["candidates_truncated"] for item in payload["products"])
            )
            local["cases_any_label_text_truncated"] += int(
                any(
                    p["fda_label_text_truncated"]
                    for item in payload["products"]
                    for p in item["candidates"]
                )
            )
        counts.update(local)
        by_type[row["case_type"]].update(local)
        categories = {qualities[n]["category"] for n in matched}
        info = {
            k: row[k]
            for k in ("case_id", "partition", "task", "case_type", "families", "ndcs", "input")
        }
        info.update(
            any_ndc=any_ndc,
            category=next(iter(categories)) if len(categories) == 1 else "mixed_categories",
            relation="not_a_comparison",
            different=False,
            source_label_ids_complete=all(
                qualities[n]["label_identifiers_complete"] for n in matched
            ),
            max_candidates=max((len(i["candidates"]) for i in requests), default=0),
            request_candidate_ndcs=[[p["product_ndc"] for p in i["candidates"]] for i in requests],
        )
        if row["task"] == "ingredient_comparison":
            left, right = requests
            relation = name_screen.relation(
                left["description"]["name"], right["description"]["name"]
            )
            info["relation"] = relation["class"]
            info["name_screen"] = relation
            sets = [
                {tuple(sorted(p["ingredient_identities"])) for p in i["candidates"]}
                for i in requests
            ]
            if any(len(s) != 1 for s in sets):
                raise ValueError("Comparison ingredient target is not established by candidates")
            a, b = (set(next(iter(s))) for s in sets)
            info["different"] = a != b
            computed = (
                "same_ingredient_control"
                if a == b
                else "different_ingredients_shared"
                if a & b
                else "different_ingredients_disjoint"
            )
            if row["case_type"] != computed:
                raise ValueError(
                    "Saved comparison type disagrees with source ingredient identities"
                )
            info["generic_like_endpoint"] = any(qualities[n]["generic_like_name"] for n in matched)
        infos.append(info)
    totals = dict(sorted(counts.items()))
    totals.update(
        cases_by_task=dict(sorted(Counter(r["task"] for r in rows).items())),
        cases_by_type=dict(sorted(Counter(r["case_type"] for r in rows).items())),
        human_verified_cases=sum(r["human_verified"] for r in rows),
        unique_ndcs=len({n for r in rows for n in r["ndcs"]}),
        unique_partition_groups=len({g for r in rows for g in r["families"]}),
    )
    comparisons = [i for i in infos if i["task"] == "ingredient_comparison"]
    totals["comparison_name_screen"] = {
        relation: summarize_case_set([r for r in comparisons if r["relation"] == relation])
        for relation in sorted({r["relation"] for r in comparisons})
    }
    totals["comparison_name_screen_by_case_type"] = {
        kind: {
            relation: summarize_case_set(
                [r for r in comparisons if r["case_type"] == kind and r["relation"] == relation]
            )
            for relation in sorted({r["relation"] for r in comparisons})
        }
        for kind in sorted({r["case_type"] for r in comparisons})
    }
    totals["by_case_type"] = {k: dict(sorted(v.items())) for k, v in sorted(by_type.items())}
    if adapter:
        totals["candidate_count_histogram_requests"] = dict(sorted(candidate_histogram.items()))
        totals["max_candidates_for_any_request"] = max(candidate_histogram, default=0)
    return totals, infos


class IngredientLexicon:
    """Token-boundary, longest-name screening; requires amounts for extra names.

    No synonym, salt, moiety, IN/PIN or UNII equivalence is inferred. Missing
    literal names and possible extras are review signals, never incorrect-gold labels.
    """

    def __init__(self, names):
        self.by_first = defaultdict(set)
        for name in names:
            tokens = name_tokens(name)
            if tokens and len(" ".join(tokens)) >= 3:
                self.by_first[tokens[0]].add(tokens)
        self.by_first = {k: sorted(v, key=lambda x: (-len(x), x)) for k, v in self.by_first.items()}

    def screen(self, structured_names, text: str) -> dict:
        expected = {name_tokens(n) for n in structured_names if name_tokens(n)}
        if not expected or not text.strip():
            return {
                "status": "not_screenable",
                "missing_structured_names": [],
                "extra_amount_anchored_names": [],
                "amount_anchored_name_count": 0,
            }
        tokens = name_tokens(re.sub(r"<[^>]+>", " ", text))
        present, amounts = set(), set()
        index = 0
        while index < len(tokens):
            found = next(
                (
                    n
                    for n in self.by_first.get(tokens[index], [])
                    if tokens[index : index + len(n)] == n
                ),
                None,
            )
            if found:
                present.add(found)
                next_index = index + len(found)
                if next_index < len(tokens) and re.match(r"\d", tokens[next_index]):
                    amounts.add(found)
                index += len(found)
            else:
                index += 1
        # A longer matched salt must not establish presence of its shorter base name.
        missing, extras = expected - present, amounts - expected
        return {
            "status": "screen_disagreement_candidate"
            if missing or extras
            else "literal_name_screen_pass_not_verification",
            "missing_structured_names": sorted(" ".join(n) for n in missing),
            "extra_amount_anchored_names": sorted(" ".join(n) for n in extras),
            "amount_anchored_name_count": len(amounts),
            "structured_name_count": len(expected),
            "count_difference_screen": bool(amounts) and len(amounts) != len(expected),
            "human_adjudicated": False,
        }


def source_archives(source_dir: Path, kind: str) -> list[Path]:
    directory = source_dir / "raw/openfda" / kind
    paths = sorted(directory.glob(f"drug-{kind}-*.json*"))
    selected = {}
    for path in paths:
        if path.name.endswith((".json", ".json.zip")):
            key = path.name.removesuffix(".zip")
            if key not in selected or path.name.endswith(".zip"):
                selected[key] = path
    if not selected:
        raise FileNotFoundError(f"Original {kind} source archives not found at {directory}")
    return sorted(selected.values())


def fda_records(inputs: Inputs, path: Path):
    import ijson

    path = inputs.register(path)
    with inputs.open(path, "rb") as raw:
        if path.name.endswith(".zip"):
            with ZipFile(raw) as archive:
                for member in sorted(archive.namelist()):
                    if member.lower().endswith(".json"):
                        with archive.open(member) as stream:
                            for row in ijson.items(stream, "results.item"):
                                yield row, member
        else:
            for row in ijson.items(raw, "results.item"):
                yield row, path.name


def structured_signature(ingredients) -> str:
    return canonical(
        sorted(
            (normalize_name(i.get("name", "")), normalize_name(i.get("strength", "")))
            for i in ingredients
        )
    )


def verify_raw_sources(inputs, validation_ndcs, metadata, labels, progress):
    label_targets = {
        labels[n]["label_id"] for n in validation_ndcs if n in labels and labels[n].get("label_id")
    }
    raw_labels, raw_ndcs = {}, defaultdict(list)
    archive_counts = Counter()
    for path in source_archives(inputs.source_dir, "label"):
        for record, member in fda_records(inputs, path):
            archive_counts["label_records_streamed"] += 1
            if record.get("id") not in label_targets:
                continue
            selected = {
                "label_id": record.get("id", ""),
                "label_set_id": record.get("set_id", ""),
                "effective_time": record.get("effective_time", ""),
                "version": str(record.get("version", "")),
                "active_ingredient": record.get("active_ingredient") or [],
                "substance_name": (record.get("openfda") or {}).get("substance_name", []),
                "archive": str(path.relative_to(inputs.source_dir)),
                "member": member,
            }
            selected["selected_fields_sha256"] = hashlib.sha256(
                canonical(selected).encode()
            ).hexdigest()
            if record["id"] in raw_labels and raw_labels[record["id"]] != selected:
                raise ValueError("Conflicting copies of an authorized raw label version")
            raw_labels[record["id"]] = selected
        progress(
            f"Verified source shard {path.name}; authorized label versions found: {len(raw_labels)}"
        )
    for path in source_archives(inputs.source_dir, "ndc"):
        for record, member in fda_records(inputs, path):
            archive_counts["ndc_records_streamed"] += 1
            ndc = record.get("product_ndc")
            if ndc not in validation_ndcs:
                continue
            raw_ndcs[ndc].append(
                {
                    "product_id": record.get("product_id", ""),
                    "spl_id": record.get("spl_id", ""),
                    "active_ingredients": record.get("active_ingredients") or [],
                    "archive": str(path.relative_to(inputs.source_dir)),
                    "member": member,
                }
            )
        progress(f"Verified original NDC archive {path.name}")
    details, counts = {}, Counter()
    for ndc in sorted(validation_ndcs):
        label = labels.get(ndc, {})
        raw = raw_labels.get(label.get("label_id"))
        original = json.loads(metadata[ndc]["original_ingredients_json"] or "[]") or []
        same_version_ndcs = [r for r in raw_ndcs[ndc] if r["spl_id"] == label.get("label_id")]
        exact_native = [
            r
            for r in same_version_ndcs
            if structured_signature(r["active_ingredients"]) == structured_signature(original)
        ]
        cache_matches = bool(
            raw
            and raw["label_set_id"] == label.get("label_set_id")
            and raw["effective_time"] == label.get("effective_time")
            and raw["version"] == label.get("label_version")
            and raw["active_ingredient"] == json.loads(label.get("active_ingredient_json") or "[]")
        )
        details[ndc] = {
            "raw_label": raw,
            "raw_ndc": (exact_native or same_version_ndcs or raw_ndcs[ndc] or [None])[0],
            "raw_label_version_and_text_match_cache": cache_matches,
            "original_ndc_same_spl_and_ingredients_confirmed": bool(exact_native),
        }
        counts["validation_unique_ndcs"] += 1
        counts["linked_label_ids"] += int(bool(label.get("label_id")))
        counts["original_label_found"] += int(bool(raw))
        counts["cached_label_version_and_text_confirmed"] += int(cache_matches)
        counts["cached_label_discrepancy_or_missing"] += int(
            bool(label.get("label_id")) and not cache_matches
        )
        counts["original_ndc_found"] += int(bool(raw_ndcs[ndc]))
        counts["original_ndc_same_spl_and_ingredients_confirmed"] += int(bool(exact_native))
        counts["ndc_same_version_missing_or_different"] += int(not exact_native)
        counts["ndcs_without_unique_linked_label"] += int(not label.get("label_id"))
        counts["linked_ndc_same_version_or_ingredients_not_confirmed"] += int(
            bool(label.get("label_id")) and not exact_native
        )
    return details, {
        **dict(sorted(counts.items())),
        **dict(sorted(archive_counts.items())),
        "unique_original_label_versions_found": len(raw_labels),
        "scope": "Original source records matched only to validation NDCs/label IDs",
    }


def conflict_screens(catalog, metadata, labels, raw_details, validation_ndcs):
    lexicon_names = set()
    for ndc in metadata:
        p = catalog["products"][ndc]
        for field in ("active_ingredients", "original_ingredients"):
            lexicon_names.update(i["name"] for i in p[field])
        for kind in ("in_names", "pin_names"):
            lexicon_names.update(p["rxnorm_ingredients"][kind])
        lexicon_names.update(
            json.loads(labels.get(ndc, {}).get("label_metadata_json") or "{}").get(
                "substance_name", []
            )
        )
    lexicon = IngredientLexicon(lexicon_names)
    screens = {}
    for ndc in sorted(validation_ndcs):
        p = catalog["products"][ndc]
        raw = raw_details[ndc]["raw_label"]
        lines = raw["active_ingredient"] if raw else p["fda_label"]["active_ingredient_text"]
        if isinstance(lines, str):
            lines = [lines]
        names = [i["name"] for i in p["original_ingredients"]]
        screen = lexicon.screen(names, "\n".join(lines))
        screen["same_version_original_sources_confirmed"] = bool(
            raw_details[ndc]["raw_label_version_and_text_match_cache"]
            and raw_details[ndc]["original_ndc_same_spl_and_ingredients_confirmed"]
        )
        screen["structured_same_label_substance_names_match_ndc"] = bool(
            raw and {name_key(n) for n in raw["substance_name"]} == {name_key(n) for n in names}
        )
        screens[ndc] = screen
    return screens


def quality_aggregate(ndcs, qualities, catalog, screens=None):
    ndcs = sorted(set(ndcs))
    booleans = (
        "ingredient_eligible",
        "comparison_eligible",
        "formulation_eligible",
        "missing_strength",
        "unparsed_quantity_or_unit",
        "missing_form",
        "missing_route",
        "all_fda_unii_identities",
        "generic_like_name",
        "retailer_name_token",
        "label_text_available",
        "label_identifiers_complete",
    )
    result = {
        "unique_ndcs": len(ndcs),
        **{k: metric(sum(qualities[n][k] for n in ndcs), len(ndcs)) for k in booleans},
    }
    for key in ("category", "support_tier", "identity_status"):
        result[key + "_counts"] = dict(sorted(Counter(qualities[n][key] for n in ndcs).items()))
    for key in ("strength_issues", "exclusion_reasons"):
        result[key + "_counts"] = dict(
            sorted(Counter(issue for n in ndcs for issue in catalog["products"][n][key]).items())
        )
    if screens is not None:
        result["label_screen_status_counts"] = dict(
            sorted(Counter(screens[n]["status"] for n in ndcs).items())
        )
        result["source_disagreement_candidates"] = metric(
            sum(screens[n]["status"] == "screen_disagreement_candidate" for n in ndcs), len(ndcs)
        )
        screenable = [n for n in ndcs if screens[n]["status"] != "not_screenable"]
        candidates = [
            n for n in screenable if screens[n]["status"] == "screen_disagreement_candidate"
        ]
        result["screenable_unique_ndcs"] = len(screenable)
        result["disagreement_candidates_among_screenable"] = metric(
            len(candidates), len(screenable)
        )
        result["candidates_with_extra_amount_anchored_names"] = sum(
            bool(screens[n]["extra_amount_anchored_names"]) for n in candidates
        )
        result["candidates_with_missing_literal_names_only"] = sum(
            not screens[n]["extra_amount_anchored_names"] for n in candidates
        )
        result["candidates_confirmed_same_source_version"] = sum(
            screens[n]["same_version_original_sources_confirmed"] for n in candidates
        )
    return result


def adapter_pairing_diagnostics(ndcs, catalog):
    counts = Counter()
    for ndc in sorted(ndcs):
        mapping = catalog["products"][ndc]["rxnorm_ingredients"]
        affected = False
        for kind in ("in", "pin"):
            names, ids = mapping[kind + "_names"], mapping[kind + "_rxcuis"]
            if not names and not ids:
                continue
            counts[kind + "_products_with_arrays"] += 1
            if len(names) != len(ids):
                counts[kind + "_source_array_length_mismatch"] += 1
                affected = True
                continue
            original = set(zip(names, ids, strict=True))
            shown_names, shown_ids = sorted(set(names)), sorted(set(ids))
            positional = (
                set(zip(shown_names, shown_ids, strict=True))
                if len(shown_names) == len(shown_ids)
                else None
            )
            if original != positional:
                counts[kind + "_positional_pairing_would_change"] += 1
                affected = True
        counts["products_with_unsafe_positional_pairing"] += int(affected)
    return {
        "unique_ndcs": len(ndcs),
        **dict(sorted(counts.items())),
        "interpretation": "Arrays are separately sorted in the adapter. A changed positional "
        "zip is demonstrable; actual model misassociation was not measured.",
        "protocol": [
            "Freeze a versioned source-only mapping before inference; never infer from references.",
            (
                "Emit paired name/RxCUI/term_type records from original aligned source arrays; "
                "verify each pair against the pinned RxNorm release and record its SHA-256."
            ),
            (
                "Use only Unicode/case/whitespace normalization; explicit audited synonyms only. "
                "Preserve salts, PIN versus IN, strengths and unresolved UNII/name alignment."
            ),
            "Many-to-many or conflicting names remain UNKNOWN/UNRESOLVED; no closest-gold match.",
            (
                "Freeze independent name-to-identity and UNII policies, then apply identical scoring "
                "and denominators to retrieval, untuned and tuned Gemma; report unknown/invalid outputs."
            ),
        ],
    }


def family_diagnostics(inputs, memberships, metadata, qualities, name_screen, infos):
    by_family, by_part = defaultdict(list), defaultdict(list)
    for ndc, member in memberships.items():
        by_family[member["family"]].append(ndc)
        by_part[member["partition"]].append(ndc)
    family_parts = defaultdict(set)
    for member in memberships.values():
        family_parts[member["family"]].add(member["partition"])
    result = {
        "scope": "Membership inferred only from training/validation explicit-NDC cases",
        "group_id_overlap_train_validation": sum(len(p) > 1 for p in family_parts.values()),
        "groups_are_verified_marketing_families": False,
        "partitions": {},
    }
    for part in ("train", "validation"):
        groups = {g: ns for g, ns in by_family.items() if memberships[ns[0]]["partition"] == part}
        sizes = sorted(map(len, groups.values()))
        case_counts = Counter(g for i in infos[part] for g in i["families"])
        result["partitions"][part] = {
            "products": len(by_part[part]),
            "groups": len(groups),
            "group_size_histogram": dict(sorted(Counter(sizes).items())),
            "singleton_groups": sizes.count(1),
            "largest_group_products": max(sizes, default=0),
            "median_group_products": statistics.median(sizes) if sizes else None,
            "groups_over_50_products": sum(n > 50 for n in sizes),
            "products_in_groups_over_50": sum(n for n in sizes if n > 50),
            "largest_groups": [
                {
                    "partition_group_id": group,
                    "products": len(ns),
                    "case_memberships": case_counts[group],
                    "generic_like_products": sum(qualities[n]["generic_like_name"] for n in ns),
                    "retailer_token_products": sum(qualities[n]["retailer_name_token"] for n in ns),
                    "distinct_normalized_names": len(
                        {name_key(metadata[n]["display_name"]) for n in ns}
                    ),
                }
                for group, ns in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:10]
            ],
        }
    # Only names of authorized source products enter this exhaustive >=90 ratio screen.
    name_ndcs = {part: defaultdict(set) for part in ("train", "validation")}
    for part, ndcs in by_part.items():
        for ndc in ndcs:
            name_ndcs[part][name_key(metadata[ndc]["display_name"])].add(ndc)
    train_names = sorted(name_ndcs["train"])
    cross = Counter()
    counterparts = defaultdict(list)
    affected, distinctive_affected = set(), set()
    for val_name in sorted(name_ndcs["validation"]):
        for train_name, score, _ in process.extract_iter(
            val_name, train_names, scorer=fuzz.ratio, score_cutoff=NEAR_NAME_THRESHOLD
        ):
            cross["unique_name_pairs_at_ratio_90"] += 1
            cross["unique_name_pairs_at_ratio_92"] += int(score >= 92)
            cross["unique_name_pairs_at_ratio_95"] += int(score >= 95)
            cross["identical_normalized_name_pairs"] += int(val_name == train_name)
            relation = name_screen.relation(val_name, train_name)
            cross[relation["class"]] += 1
            affected.update(name_ndcs["validation"][val_name])
            if relation["class"] in PRIMARY_RELATIONS:
                distinctive_affected.update(name_ndcs["validation"][val_name])
                train_ndc = min(name_ndcs["train"][train_name])
                for val_ndc in name_ndcs["validation"][val_name]:
                    counterparts[val_ndc].append(
                        {
                            "validation_ndc": val_ndc,
                            "training_source_ndc": train_ndc,
                            "training_display_name": metadata[train_ndc]["display_name"],
                            "character_similarity": round(score, 4),
                            "relation_screen": relation["class"],
                            "human_verified": False,
                        }
                    )
    # Lower-similarity brand variants can share a rare distinctive token. Report separately.
    token_ndcs = {part: defaultdict(set) for part in ("train", "validation")}
    for part, mapping in name_ndcs.items():
        for name, ns in mapping.items():
            for token in set(name.split()) & name_screen.distinctive:
                token_ndcs[part][token].update(ns)
    cross_tokens = set(token_ndcs["train"]) & set(token_ndcs["validation"])
    token_affected = {n for t in cross_tokens for n in token_ndcs["validation"][t]}
    result["cross_split_name_screen"] = {
        **dict(sorted(cross.items())),
        "affected_validation_ndcs": len(affected),
        "affected_validation_groups": len({memberships[n]["family"] for n in affected}),
        "primary_relation_candidate_validation_ndcs": len(distinctive_affected),
        "shared_distinctive_tokens_across_splits": len(cross_tokens),
        "validation_ndcs_sharing_distinctive_token_across_splits": len(token_affected),
        "warning": "Screening candidates require brand/variant review. Zero shared algorithmic "
        "group IDs does not establish zero real-brand leakage.",
    }
    link_counts = defaultdict(Counter)
    link_path = inputs.repo / "data/processed/final_v5/family_links.csv"
    for row in inputs.csv(link_path):
        a, b = row["product_ndc_a"], row["product_ndc_b"]
        if a not in memberships or b not in memberships:
            continue  # No held-out membership, names, or reference labels are accessed.
        reason = row["reason"]
        link_counts[reason]["allowed_links"] += 1
        link_counts[reason]["cross_train_validation_links"] += int(
            memberships[a]["partition"] != memberships[b]["partition"]
        )
        relation = name_screen.relation(metadata[a]["display_name"], metadata[b]["display_name"])
        link_counts[reason][relation["class"]] += 1
        link_counts[reason]["generic_like_endpoint"] += int(
            qualities[a]["generic_like_name"] or qualities[b]["generic_like_name"]
        )
    result["allowed_link_diagnostics_by_reason"] = {
        k: dict(sorted(v.items())) for k, v in sorted(link_counts.items())
    }
    result["link_precision_is_human_verified"] = False
    return (
        result,
        distinctive_affected | token_affected,
        {
            n: sorted(values, key=lambda v: (-v["character_similarity"], v["training_source_ndc"]))[
                :3
            ]
            for n, values in counterparts.items()
        },
    )


def review_eligible_strata(info):
    if info["partition"] != "validation":
        raise ValueError("Manual queue is validation-only")
    result = set()
    if info["different"] and info["relation"] == "identical_normalized_name":
        result.add("identical_name_different_ingredients")
    if info["different"] and info["relation"] in PRIMARY_RELATIONS - {"identical_normalized_name"}:
        result.add("similar_name_different_ingredients")
    if info["case_type"] == "same_ingredient_control" and info["relation"] in PRIMARY_RELATIONS:
        result.add("same_ingredient_controls")
    if info["case_type"] == "name_only_ambiguous":
        result.add("name_only_ambiguity")
    if info.get("source_disagreement"):
        result.add("source_disagreement")
    if info["category"] == "conventional_oral_analgesic_cold_allergy":
        result.add("conventional_oral_therapeutic")
    return result


def select_review_queue(infos, seed=SEED):
    if seed != SEED:
        raise ValueError("This pre-registered audit requires seed 42")
    randomizer = random.Random(seed)
    tie_breaks = {
        r["case_id"]: randomizer.random() for r in sorted(infos, key=lambda r: r["case_id"])
    }
    pools = {s: [r for r in infos if s in review_eligible_strata(r)] for s in REVIEW_QUOTAS}
    selected, used, families, categories = [], set(), set(), Counter()
    quotas = Counter()

    def choose(stratum, force_known=False):
        available = [r for r in pools[stratum] if r["case_id"] not in used]
        if force_known:
            available = [r for r in available if "65692-0556" in r["ndcs"]]
        if not available:
            return False
        # Prefer an entirely new partition group, then manageable manual evidence and categories.
        best = min(
            available,
            key=lambda r: (
                bool(set(r["families"]) & families),
                r["max_candidates"] > 20,
                categories[r["category"]],
                tie_breaks[r["case_id"]],
                r["case_id"],
            ),
        )
        selected.append({**best, "review_stratum": stratum})
        used.add(best["case_id"])
        families.update(best["families"])
        categories[best["category"]] += 1
        quotas[stratum] += 1
        return True

    # A specifically requested, known source discrepancy is reserved as a diagnostic sentinel.
    choose("source_disagreement", force_known=True)
    for _ in range(max(REVIEW_QUOTAS.values())):
        for stratum, quota in REVIEW_QUOTAS.items():
            if quotas[stratum] < quota:
                choose(stratum)
    target = sum(REVIEW_QUOTAS.values())
    while len(selected) < target:
        if not any(choose(s) for s in REVIEW_QUOTAS):
            break
    summary = {
        "seed": seed,
        "target_cases": target,
        "selected_cases": len(selected),
        "planned_quotas": REVIEW_QUOTAS,
        "selected_by_assigned_stratum": dict(sorted(quotas.items())),
        "eligible_pool_by_stratum_overlapping": {s: len(pool) for s, pool in pools.items()},
        "stratum_shortfalls": {s: max(0, q - quotas[s]) for s, q in REVIEW_QUOTAS.items()},
        "unique_partition_groups": len(families),
        "cases_reusing_any_group": sum(
            bool(
                set(r["families"]) & {g for previous in selected[:i] for g in previous["families"]}
            )
            for i, r in enumerate(selected)
        ),
        "cases_by_category": dict(sorted(Counter(r["category"] for r in selected).items())),
        "cases_with_any_explicit_ndc": sum(r["any_ndc"] for r in selected),
        "primary_different_ingredient_cases": sum(
            r["different"] and r["relation"] in PRIMARY_RELATIONS for r in selected
        ),
        "primary_different_ingredient_cases_without_explicit_ndc": sum(
            r["different"] and r["relation"] in PRIMARY_RELATIONS and not r["any_ndc"]
            for r in selected
        ),
        "independent_human_adjudications_completed": 0,
        "model_predictions_used_for_selection": False,
        "membership_sha256": hashlib.sha256(
            canonical([[r["case_id"], r["review_stratum"]] for r in selected]).encode()
        ).hexdigest(),
        "policy": "Fixed quotas; seed-42 ID-sorted random ties; greedily prefer unused groups, "
        "<=20 candidates and underrepresented source categories. Reserve known Spearmint "
        "source sentinel. Shortfalls are redistributed in fixed stratum order. "
        "Source-derived comparison classes are strata, never expected model outcomes.",
    }
    return selected, summary


def citations_for(ndc, metadata, labels, raw_details, screens):
    row, label = metadata[ndc], labels.get(ndc, {})
    raw = raw_details[ndc]
    setid, labelid = label.get("label_set_id", ""), label.get("label_id", "")
    return {
        "product_ndc": ndc,
        "display_name": row["display_name"],
        "label_id": labelid,
        "label_set_id": setid,
        "label_version": label.get("label_version", ""),
        "effective_time": label.get("effective_time", ""),
        "dailymed_set_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=" + setid,
        "openfda_same_label_id_query": "https://api.fda.gov/drug/label.json?search=id:%22"
        + labelid
        + "%22&limit=1",
        "citations_note": "The set URL may show a newer version; review the recorded label ID/version.",
        "original_ndc_structured_ingredients": json.loads(row["original_ingredients_json"] or "[]"),
        "catalog_effective_ingredients": json.loads(row["active_ingredients_json"] or "[]"),
        "cached_label_active_text": json.loads(label.get("active_ingredient_json") or "[]"),
        "original_source_checks": raw,
        "source_disagreement_screen": screens[ndc],
        "independence_note": "FDA structured data, openFDA SPL and DailyMed can repeat the same "
        "manufacturer submission; their agreement is not independent truth.",
    }


def queue_csv(selected, metadata, labels, raw_details, screens) -> str:
    stream = io.StringIO(newline="")
    columns = [
        "case_id",
        "partition",
        "task",
        "case_type",
        "review_stratum",
        "category",
        "input_json",
        "name_relation_screen",
        "family_group_ids_json",
        "product_ndcs_json",
        "request_candidate_ndcs_json",
        "contains_explicit_ndc",
        "maximum_candidates",
        "cross_split_name_review_needed",
        "cross_split_training_source_candidates_json",
        "source_evidence_json",
        "selection_reasons_json",
        "review_status",
        "human_verified",
        "reviewer_1",
        "reviewer_1_verdict",
        "reviewer_2",
        "reviewer_2_verdict",
        "adjudicator",
        "adjudicated_verdict",
        "review_notes",
    ]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for r in selected:
        if r["partition"] != "validation":
            raise ValueError("Review queue contains an unauthorized partition")
        record = dict.fromkeys(columns, "")
        record.update(
            {
                "case_id": r["case_id"],
                "partition": "validation",
                "task": r["task"],
                "case_type": r["case_type"],
                "review_stratum": r["review_stratum"],
                "category": r["category"],
                "name_relation_screen": r["relation"],
                "input_json": canonical(r["input"]),
                "family_group_ids_json": canonical(r["families"]),
                "product_ndcs_json": canonical(r["ndcs"]),
                "request_candidate_ndcs_json": canonical(r["request_candidate_ndcs"]),
                "contains_explicit_ndc": r["any_ndc"],
                "maximum_candidates": r["max_candidates"],
                "cross_split_name_review_needed": r.get("cross_split_name_review_needed", False),
                "cross_split_training_source_candidates_json": canonical(
                    r.get("cross_split_training_source_candidates", [])
                ),
                "source_evidence_json": canonical(
                    [
                        citations_for(n, metadata, labels, raw_details, screens)
                        for n in sorted(r["ndcs"])
                    ]
                ),
                "selection_reasons_json": canonical(sorted(review_eligible_strata(r))),
                "review_status": "PENDING",
                "human_verified": False,
            }
        )
        writer.writerow(record)
    return stream.getvalue()


def independent_group_bound(infos):
    """Deterministic disjoint-group packing, a conservative proxy, not a maximum or gold count."""
    used, selected = set(), 0
    for info in sorted(infos, key=lambda r: (len(r["families"]), r["case_id"])):
        if not used.intersection(info["families"]):
            selected += 1
            used.update(info["families"])
    return selected


def protect_existing_review(output: Path) -> None:
    """Never overwrite real reviewer entries when rerunning an automated audit."""
    path = output / "validation_review_queue.PRIVATE.csv"
    if not path.exists():
        return
    if path.is_symlink():
        raise ValueError("Private review queue must not redirect through a symlink")
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            reviewed = any(
                row.get(k, "").strip()
                for k in (
                    "reviewer_1",
                    "reviewer_1_verdict",
                    "reviewer_2",
                    "reviewer_2_verdict",
                    "adjudicator",
                    "adjudicated_verdict",
                    "review_notes",
                )
            )
            if (
                reviewed
                or row.get("review_status") != "PENDING"
                or row.get("human_verified", "False").casefold() != "false"
            ):
                raise ValueError("Existing human review entries must be preserved; audit not rerun")


def render_findings(scope, quality, family):
    train, val = scope["partitions"]["train"], scope["partitions"]["validation"]
    challenge, queue = scope["challenge_screen"], scope["manual_review_queue"]
    names = challenge["different_ingredient_name_classes"]
    n = val["cases"]
    return f"""# Revised Stage 5: independent benchmark and challenge-set audit

Audit date: 2026-11-19. Seed: 42. This is a computational source/method audit and a
pre-registered **pending** independent-review queue; independent adjudication is not complete.

## Decision

Retain `final_v5` and Parts 1–4 unchanged. Introduce `challenge_v1` as a versioned ID/annotation
overlay only after manual review. No documented evidence here warrants silently rewriting the
frozen benchmark. The source discrepancy requires adjudication and an overlay exclusion/uncertainty
flag; an established reference error would warrant a separately approved versioned correction.

## Actual inputs and reproducibility

- Repository: `{scope["locations"]["repository"]}`.
- Original datasets: `{scope["locations"]["original_data"]}`.
- Derived source directory: `data/interim/final_v5/`; cases: `data/processed/final_v5/`.
- Only `train.csv` and `validation.csv` were opened. No test cases, test answers, test predictions,
  test file hashes, or test split manifest were accessed. No model inference, training, downloads,
  GPU/cloud operations, benchmark edits, or baseline-result changes occurred.
- Source archives were streamed in place, without extraction/copying. Only validation-linked original
  NDC/label records were retained for checks. RxNorm raw release files were inventoried, not parsed;
  existing hashed RxNorm link tables supply this audit's RxNorm evidence.
- Full public-source catalog indexing preserves retrieval behavior; every analyzed query candidate
  was checked against allowed train/validation provenance. Source indexing is not held-out answers.
- SHA-256 and actual absolute paths for every consumed input are in `scope_counts.json:input_files`.
  All input hashes were rechecked unchanged. The queue membership hash is
  `{queue["membership_sha256"]}`. Aggregate JSON gives all subgroup denominators.

## Verified aggregate findings

| Partition | Cases | Comparisons | Explicit-NDC case type | Name-only ambiguous | Groups |
|---|---:|---:|---:|---:|---:|
| Train | {train["cases"]:,} | {train["cases_by_task"].get("ingredient_comparison", 0):,} | {train["cases_by_type"].get("explicit_ndc", 0):,} | {train["cases_by_type"].get("name_only_ambiguous", 0):,} | {train["unique_partition_groups"]:,} |
| Validation | {n:,} | {val["cases_by_task"].get("ingredient_comparison", 0):,} | {val["cases_by_type"].get("explicit_ndc", 0):,} | {val["cases_by_type"].get("name_only_ambiguous", 0):,} | {val["unique_partition_groups"]:,} |

Validation contains {val.get("cases_with_any_explicit_ndc", 0):,}/{n:,} cases with an NDC in
at least one request, including comparison endpoints. {val.get("cases_all_requests_single_exact_identifier_lookup", 0):,}/{n:,}
cases have one candidate per request at the initial exact-name/NDC lookup, before formulation filtering.
These are source-lookup characteristics, not measurements of real-world difficulty.

Of {val["reference_targets"]:,} identifiable product targets,
{val["full_formulation_reference_targets"]:,} have full-formulation support; ingredient sets and
complete ingredient/strength/form/route formulations must have separate denominators.
All {val["human_verified_cases"]:,}/{n:,} validation provenance records marked human-verified were
counted literally; this audit supplies no new human verdicts.

Of the {val["cases_by_type"].get("name_only_ambiguous", 0)} saved name-only ambiguity cases,
{val.get("name_only_ambiguity_with_identifier_or_supported_form_difference", 0)} have different
ingredient identifier sets or multiple supported formulations; the remaining
{val.get("name_only_ambiguity_with_only_source_names_or_incomplete_forms", 0)} need source-name and
incomplete-formulation review. This is not a finding that those remaining labels are wrong.

Among the {sum(v["cases"] for v in names.values()):,} validation different-ingredient comparisons:
{names.get("identical_normalized_name", {}).get("cases", 0)} have identical normalized names,
{names.get("shared_distinctive_brand_variant_candidate", {}).get("cases", 0)} screen as distinctive
shared-brand/variant candidates, {names.get("highly_similar_name_candidate", {}).get("cases", 0)} as
high-character-similarity candidates, {names.get("generic_or_retailer_false_match_candidate", {}).get("cases", 0)}
as generic/retailer matches and {names.get("unrelated_co_partitioned_name_candidate", {}).get("cases", 0)}
as unrelated co-partitioned candidates. These are rule-based screens requiring actual relationship review.

The primary name screen finds {challenge["screen_positive_different_ingredient_cases"]["cases"]:,}
different-ingredient comparison cases in validation across
{challenge["screen_positive_different_ingredient_cases"]["unique_partition_groups"]:,} conservative
groups. Of these, {challenge["screen_positive_without_explicit_ndc"]["cases"]:,} have no explicit
NDCs, and {challenge["source_reviewable_candidates"]["cases"]:,} have complete label identifiers.
The deterministic disjoint-group packing yields {challenge["greedy_disjoint_group_case_bound"]:,}
review units. These are **screen-positive review candidates**, not verified confusing brands or
independently verified challenging cases. Independent adjudications completed: **0**.
Only {challenge["conventional_oral_screen_positive"]["cases"]} screen-positive different-active
cases are in the conventional oral analgesic/cold/allergy source category; only
{challenge["conventional_oral_screen_positive_without_explicit_ndc"]["cases"]} of these omit NDCs.
See `comparison_name_screen_by_case_type` for identical, distinctive shared-variant, high-similarity,
generic/retailer and unrelated-co-partitioned strata, each with unique products/groups and NDC use.

The actual adapter truncates {val.get("requests_candidates_truncated", 0):,}/{val["product_requests"]:,}
validation product requests in {val.get("cases_candidates_truncated", 0):,}/{n:,} cases at eight candidates.
{val.get("truncated_requests_losing_ingredient_alternatives", 0):,} truncated requests lose an ingredient-set
alternative, and {val.get("truncated_requests_hiding_ingredient_ambiguity", 0):,} display only one set
despite multiple full-catalog sets. Retrieval sees the full candidate list. Label text is separately
cut at 1,200 characters for {val.get("shown_candidate_label_text_truncated", 0):,} shown candidate records.
Direct native-ingredient lists are displayed in
{val.get("requests_with_direct_native_ingredient_lists", 0):,}/{val["product_requests"]:,} requests.
This measures evidence-grounded extraction, with substantial opportunity to copy source ingredients.

All {quality["original_source_verification"]["original_ndc_found"]:,} validation NDCs were located in
the original source. {quality["original_source_verification"]["cached_label_version_and_text_confirmed"]:,}/
{quality["original_source_verification"]["linked_label_ids"]:,} uniquely linked labels matched
their original SPL version/text; same-version structured NDC ingredients were confirmed for
{quality["original_source_verification"]["original_ndc_same_spl_and_ingredients_confirmed"]:,} products;
there were {quality["original_source_verification"]["cached_label_discrepancy_or_missing"]} cache discrepancies.
The remaining {quality["original_source_verification"]["ndcs_without_unique_linked_label"]} products lack
a unique linked label; they are not established cache errors. All
{scope["original_source_inventory_metadata_only"]["fda_label_archives"]} FDA label shards and the NDC archive
were streamed in place, with SHA-256 recorded. RxNorm raw release paths/sizes are inventoried separately.

The lexical source screen flags {quality["validation_products"]["source_disagreement_candidates"]["numerator"]:,}/
{quality["validation_products"]["screenable_unique_ndcs"]:,} screenable validation products for review,
out of {quality["validation_products"]["unique_ndcs"]:,} total products.
{quality["validation_products"]["candidates_with_extra_amount_anchored_names"]} flagged products include
extra amount-anchored names; the rest have missing literal names only. Category-specific denominators
and source-support/strength flags are in `quality_and_conflicts.json`. Missing literal
names, salts, aliases, amount anchoring, and incomplete text can create false flags; these counts
are **not** a prevalence estimate of incorrect labels. No field was rewritten.

Known requested example, NDC 65692-0556: `{canonical(quality["known_spearmint_source_conflict"])}`.
Both original source versions, strengths, label ID/set/version/date, archived evidence locators and
review-needed flags are recorded in the private queue if this validation example is present.
DailyMed set pages can change; review the recorded exact SPL version. Matching manufacturer-submitted
SPL renderings across FDA/RxNorm/DailyMed is not independent clinical verification.

There are {family["group_id_overlap_train_validation"]} shared algorithmic group IDs across train and
validation. Cross-split screens: `{canonical(family["cross_split_name_screen"])}`.
These flags require manual examination. Group ID disjointness alone does not establish real-brand
independence; oversized generic/retailer clusters and missed brand relations remain possible.
Largest allowed groups and their generic/retailer composition appear in `family_group_diagnostics.json`.

## Methodological weaknesses, ranked

1. **Critical — unadjudicated reference/source circle.** `benchmark.py:source_reference/catalog_answer`
   and `retrieval_baseline.py:build_catalog/resolve_product` share sources. Historical 100% ingredient
   and overlap results remain valid source-alignment results, not independently established truth.
   Source conflicts cannot be settled by voting among duplicate SPL renderings.
2. **High — primary construct and identifier dependence.** The broad benchmark includes easy exact
   identifiers, generic names and graph-neighbor pairs. The name screen is not proof of confusion.
   Report the focused overlay separately, with same-set controls and answerable variant descriptions.
3. **High — unequal candidate evidence and scoring.** `evidence_adapter.py:build_evidence_payload`
   caps candidates; retrieval does not. Keep full-source retrieval as a ceiling. For the principal
   matched comparison supply identical evidence/budgets to both Gemma checkpoints and an explicitly
   defined deterministic comparator, or separately report overflow; do not downgrade historical retrieval.
   `evidence_adapter.py:_source_product_view` separately sorts names/IDs. The diagnostic found
   {quality["adapter_identity_mapping"].get("products_with_unsafe_positional_pairing", 0):,}/
   {quality["adapter_identity_mapping"]["unique_ndcs"]:,} unique validation products where positional
   name/ID pairing would change. This proves a representation hazard, not an observed model error.
4. **High — uncertain family independence.** `benchmark.py:assign_catalog_families` forms conservative
   partition groups. Validate true brand/variant relations and large-cluster sensitivity; use group-aware
   uncertainty estimates and report remaining dependence, rather than treating repeated rows as independent.
5. **Moderate — incomplete formulations and category breadth.** Preserve ingredient-set targets;
   separately report full formulations and strength/route limitations. Categories here are deterministic
   source-metadata screens, not clinical judgments. Sunscreens are included as U.S. OTC drugs;
   homeopathic-labeled products form a separate stratum without efficacy claims.
6. **Moderate — parser/experiment readiness.** The smoke parser is not a validated nested task-output
   scorer. Its repair, frozen ingredient mapping and GPU compatibility checks belong to the next stage.

## Necessary fixes before the primary experiment

- Independently review this validation queue; record two reviewers where feasible, provenance,
  each actual verdict, discrepancies and an adjudicator. Ideally reviewers are blinded to source-derived
  class labels and model outputs. Confirm names/relationships, label versions, actives and answerability.
- Exclude unresolved source conflicts and unsupported ingredient identities from high-confidence
  accuracy denominators; count and report them as uncertainty/conflict items. For name-only inputs
  admitting multiple formulations, correct abstention is the target. Never guess a unique NDC.
- Pre-register primary category scope and case definitions before model outcomes. Use conventional
  oral analgesic/cold/allergy source categories as a lead stratum; retain topical, dental, sunscreen,
  homeopathic and mixed categories as explicitly reported secondary strata. If the primary scope
  lacks enough independent groups, report that limitation rather than selecting successful examples.
- Freeze the source-only name→identity protocol recorded in `quality_and_conflicts.json`, validate
  paired RxNorm names/IDs, preserve salts/PIN/IN distinctions and unresolved UNII equivalence, and
  implement nested JSON/type validation and common metric denominators. No prompt/scorer code was edited here.

## Proposed challenge_v1 overlay and adjudication

Start with the **{queue["selected_cases"]} validation-only review candidates** in the private CSV;
planned allocation: 20 identical-name/different-actives, 25 similar-name/different-actives,
15 same-ingredient controls, 10 name-only ambiguity, 15 source-disagreement diagnostics,
and 15 conventional oral therapeutic items. Strata overlap; each selected row has one assigned
stratum. Realized counts are `{canonical(queue["selected_by_assigned_stratum"])}` across
{queue["unique_partition_groups"]} groups; {queue["cases_reusing_any_group"]} rows reuse a group.
This is a deliberately stratified audit, not a random prevalence sample or an achieved final
challenge size. Report shortfalls and category/identifier cross-tabs from the aggregate JSON.
The queue includes {queue["primary_different_ingredient_cases"]} primary-screen different-active cases,
of which {queue["primary_different_ingredient_cases_without_explicit_ndc"]} omit NDCs. Controls are
stratified candidates, not yet individually matched to adjudicated cases. The private queue includes
up to three high-similarity training-source counterparts per validation product when present,
to support real-brand leakage review without any held-out case data.

After adjudication, freeze an overlay of existing case IDs, source hashes, evidence policy,
scope, source-conflict exclusions, reviewer/version annotations and permitted target identities.
Use confirmed identical/similar-name different-active cases as the primary analysis, matched
same-set controls as specificity checks, and genuine ambiguity/conflicts as separate abstention
outcomes. Exact-NDC controls and category sensitivity analyses remain supplementary.
Full formulations are secondary to active-ingredient composition. Keep the same input information
for untuned Gemma, QLoRA Gemma and the declared deterministic comparator; preserve full-source
retrieval's historical ceiling. No challenged test overlay has been created or inspected.

## Optional improvements and unchanged artifacts

Optional: matched evidence/no-evidence development ablations, independently validated partial-name
descriptions, additional reviewer replication and group-level bootstrap sensitivity. Any altered
input task is a separately versioned experiment applied equally to all compared systems.
Keep original archives, `final_v5` catalogs, references, partitioning, reference-generation code,
deterministic retrieval and Part 4 results unchanged. Archived global split proportions may be
cited from the permitted aggregate summary; this audit recomputes train/validation only.

## Unresolved questions and next stage

Reviewers must settle the actual Spearmint formulation, all other lexical flags, real commercial
brand relations, cross-split brand candidates, amount/strength bases, and the independent-evidence
status of any corroborating source. FDA/RxNorm agreement alone closes none of these questions.
The queue is PENDING throughout, with blank reviewer/verdict fields. Nothing in this audit establishes
clinical harm, independent ground truth, model performance, or a QLoRA benefit.

Next implement the **adjudication/overlay manifest and frozen parser/identity scorer**, then lock
fair evidence and denominators. After that gate, proceed to revised Stage 6 GPU load and QLoRA
micro-run, recording the actual checkpoint and environment. The independent-review component
of Stage 5 remains open until genuine review outcomes are supplied.
"""


def run(repo: Path, source_dir: Path | None = None, *, progress=print) -> dict:
    repo = repo.resolve()
    source_dir = (source_dir or repo.parent / "data").resolve()
    if not repo.is_dir() or not source_dir.is_dir():
        raise FileNotFoundError("Both repository and original dataset directory must exist")
    output = repo / "results/revised_scope_audit"
    if output.resolve() != output:
        raise ValueError("Audit output directory must not redirect through a symlink")
    protect_existing_review(output)
    csv.field_size_limit(32 * 1024 * 1024)
    inputs = Inputs(repo, source_dir)
    for name in (
        "HRP_Revised_Research_Repository_Audit_2026-11-19.md",
        "HRP_Codex_Local_Dataset_Audit_Instructions.md",
        "src/hrp/benchmark.py",
        "src/hrp/retrieval_baseline.py",
        "src/hrp/evidence_adapter.py",
        "src/hrp/rxnorm_ingredients.py",
        "src/hrp/data_cleaning.py",
        "src/hrp/data_loading.py",
        "tools/revised_scope_validation_audit.py",
    ):
        inputs.register(repo / name)
    benchmark = inputs.json(repo / "results/final_v5/part03/benchmark_summary.json")
    baseline = inputs.json(repo / "results/final_v5/part04/validation_summary.json")
    if benchmark.get("version") != "final_v5":
        raise ValueError("Expected the frozen final_v5 benchmark")
    rows, memberships = load_cases(inputs)
    progress(
        f"Loaded authorized cases: train={len(rows['train'])}, validation={len(rows['validation'])}"
    )
    catalog, metadata, labels = load_sources(inputs, memberships)
    source_matches = {
        name: inputs.hashes[name]["sha256"] == digest
        for name, digest in baseline.get("source_hashes", {}).items()
    }
    case_matches = {
        part: inputs.hashes[f"data/processed/final_v5/{part}.csv"]["sha256"]
        == baseline[part]["case_file_sha256"]
        for part in ("train", "validation")
    }
    if not all(source_matches.values()) or not all(case_matches.values()):
        raise RuntimeError("Current inputs do not match the saved Part 4 source/case hashes")
    qualities = {n: product_quality(catalog["products"][n], metadata[n]) for n in memberships}
    screen = NameScreen(metadata)
    partitions, infos = {}, {}
    for part in ("train", "validation"):
        partitions[part], infos[part] = analyze_partition(
            rows[part], catalog, memberships, qualities, screen, adapter=part == "validation"
        )
        progress(f"Audited {part}: {partitions[part]['cases']} cases")
    validation_ndcs = {n for n, m in memberships.items() if m["partition"] == "validation"}
    raw_details, raw_summary = verify_raw_sources(
        inputs, validation_ndcs, metadata, labels, progress
    )
    screens = conflict_screens(catalog, metadata, labels, raw_details, validation_ndcs)
    family, cross_split_ndcs, cross_split_counterparts = family_diagnostics(
        inputs, memberships, metadata, qualities, screen, infos
    )
    for r in infos["validation"]:
        r["source_disagreement"] = any(
            screens[n]["status"] == "screen_disagreement_candidate" for n in r["ndcs"]
        )
        r["cross_split_name_review_needed"] = bool(set(r["ndcs"]) & cross_split_ndcs)
        r["cross_split_training_source_candidates"] = []
        for n in r["ndcs"]:
            for candidate in cross_split_counterparts.get(n, []):
                training_label = labels.get(candidate["training_source_ndc"], {})
                r["cross_split_training_source_candidates"].append(
                    {
                        **candidate,
                        "training_label_id": training_label.get("label_id", ""),
                        "training_label_set_id": training_label.get("label_set_id", ""),
                        "training_label_effective_time": training_label.get("effective_time", ""),
                        "source_citation": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid="
                        + training_label.get("label_set_id", ""),
                        "note": "Public training-source relation check only; not a training answer "
                        "or an independently verified brand relation.",
                    }
                )
    selected, selection_summary = select_review_queue(infos["validation"])
    private_csv = queue_csv(selected, metadata, labels, raw_details, screens)
    selection_summary["private_csv_sha256"] = hashlib.sha256(private_csv.encode()).hexdigest()
    primary = [
        i for i in infos["validation"] if i["different"] and i["relation"] in PRIMARY_RELATIONS
    ]
    reviewable = [i for i in primary if i["source_label_ids_complete"]]
    literal_clean = [i for i in reviewable if not i["source_disagreement"]]
    known = "65692-0556"
    known_report = {
        "present_in_validation": known in validation_ndcs,
        "human_adjudicated": False,
        "correct_formulation_established": False,
    }
    if known in validation_ndcs:
        known_raw = raw_details[known]["raw_label"]
        known_report.update(
            {
                "ndc": known,
                "label_id": labels[known]["label_id"],
                "set_id": labels[known]["label_set_id"],
                "version": labels[known]["label_version"],
                "effective_time": labels[known]["effective_time"],
                "structured_names": [
                    i["name"] for i in catalog["products"][known]["original_ingredients"]
                ],
                "official_same_version_active_text": known_raw["active_ingredient"]
                if known_raw
                else [],
                "screen": screens[known],
                "review_needed": True,
            }
        )
    quality = {
        "seed": SEED,
        "screening_is_adjudication": False,
        "original_source_verification": raw_summary,
        "train_products": quality_aggregate(
            [n for n, m in memberships.items() if m["partition"] == "train"], qualities, catalog
        ),
        "validation_products": quality_aggregate(validation_ndcs, qualities, catalog, screens),
        "validation_label_screen_by_category": {
            category: quality_aggregate(
                [n for n in validation_ndcs if qualities[n]["category"] == category],
                qualities,
                catalog,
                screens,
            )
            for category in sorted({qualities[n]["category"] for n in validation_ndcs})
        },
        "validation_comparison_subgroups": {
            relation: quality_aggregate(
                {n for i in infos["validation"] if i["relation"] == relation for n in i["ndcs"]},
                qualities,
                catalog,
                screens,
            )
            for relation in sorted(
                {i["relation"] for i in infos["validation"] if i["task"] == "ingredient_comparison"}
            )
        },
        "known_spearmint_source_conflict": known_report,
        "adapter_identity_mapping": adapter_pairing_diagnostics(validation_ndcs, catalog),
        "limitations": [
            "Literal token/amount screening cannot establish clinically correct ingredients or strengths.",
            (
                "No synonym/salt merging or clinical-equivalence inference. Name/quantity parse failures "
                "are review flags. Unrecognized extra ingredient names may be missed."
            ),
            "Category rules are source-metadata screening, not clinical efficacy/safety judgments.",
            "Original-source agreement and FDA/RxNorm corroboration are not independent human truth.",
            "All human review fields remain pending. Source errors require independent adjudication.",
        ],
    }
    scope = {
        "audit": "revised-stage5-train-validation-only-v1",
        "benchmark_version": "final_v5",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "locations": {
            "repository": str(repo),
            "original_data": str(source_dir),
            "interim": str(repo / "data/interim/final_v5"),
            "processed": str(repo / "data/processed/final_v5"),
        },
        "runtime": {
            "python": sys.version,
            **{name: version(name) for name in ("pandas", "rapidfuzz", "ijson")},
        },
        "original_source_inventory_metadata_only": {
            "fda_label_archives": len(source_archives(source_dir, "label")),
            "fda_ndc_archives": len(source_archives(source_dir, "ndc")),
            "rxnorm_rrf_files_not_parsed_or_hashed": [
                {"path": str(p.resolve()), "bytes": p.stat().st_size}
                for release in sorted((source_dir / "raw/rxnorm").glob("*"))
                for p in sorted((release / "rrf").glob("*.RRF"))
            ],
        },
        "partitions": partitions,
        "policy": {
            "name_normalization": "NFKC casefold, ASCII alphanumeric tokens; punctuation to spaces",
            "similarity": "rapidfuzz.fuzz.ratio on normalized names >=90; cross-split 92/95 sensitivity",
            "distinctive_tokens": "Length >=4, <=128 distinct allowed names; source ingredient, "
            "generic and retailer tokens excluded; rule-based screen only",
            "categories": "Homeopathic marketing category first, then sunscreen names/actives, "
            "dental metadata, oral analgesic/cold/allergy ingredient list, topical, other",
            "candidate_cap": MAX_CANDIDATES,
            "label_text_cap_characters": MAX_LABEL_CHARS,
            "denominators": "Cases, product requests, identifiable reference targets, shown candidate "
            "records, unique NDCs and partition groups are distinct units",
        },
        "challenge_screen": {
            "different_ingredient_name_classes": {
                relation: summarize_case_set(
                    [i for i in infos["validation"] if i["different"] and i["relation"] == relation]
                )
                for relation in sorted(
                    {i["relation"] for i in infos["validation"] if i["different"]}
                )
            },
            "screen_positive_different_ingredient_cases": summarize_case_set(primary),
            "screen_positive_without_explicit_ndc": summarize_case_set(
                [i for i in primary if not i["any_ndc"]]
            ),
            "screen_positive_non_generic_endpoints": summarize_case_set(
                [i for i in primary if not i["generic_like_endpoint"]]
            ),
            "source_reviewable_candidates": summarize_case_set(reviewable),
            "source_reviewable_literal_screen_clean": summarize_case_set(literal_clean),
            "conventional_oral_screen_positive": summarize_case_set(
                [i for i in primary if i["category"] == "conventional_oral_analgesic_cold_allergy"]
            ),
            "conventional_oral_screen_positive_without_explicit_ndc": summarize_case_set(
                [
                    i
                    for i in primary
                    if i["category"] == "conventional_oral_analgesic_cold_allergy"
                    and not i["any_ndc"]
                ]
            ),
            "greedy_disjoint_group_case_bound": independent_group_bound(reviewable),
            "bound_interpretation": "Deterministic non-overlapping partition-group packing, "
            "not a maximum or verified brand independence count",
            "independently_adjudicated_challenging_cases": 0,
        },
        "manual_review_queue": selection_summary,
        "saved_baseline_reconciliation": {
            "source_hash_matches": source_matches,
            "case_hash_matches": case_matches,
            "retrieval_code_hash_matches": inputs.hashes["src/hrp/retrieval_baseline.py"]["sha256"]
            == baseline.get("retrieval_code_sha256"),
            "validation_ingredient_set_exact_match": baseline["validation"]["product_resolution"][
                "ingredient_identification"
            ]["set_exact_match_all_eligible"],
            "validation_comparison_accuracy": baseline["validation"]["ingredient_comparison"][
                "classification_accuracy_all_cases"
            ],
            "interpretation": "Previously saved source-consistency performance; not new model metrics",
        },
        "saved_global_partition_case_counts_aggregate_only": {
            p: data["cases"] for p, data in benchmark["partitions"].items()
        },
        "safeguards": {
            "sealed_test_opened": False,
            "sealed_test_hashed": False,
            "heldout_answers_accessed": False,
            "test_predictions_generated": False,
            "split_manifest_opened": False,
            "model_inference_run": False,
            "original_datasets_copied_or_modified": False,
            "existing_benchmark_or_baseline_modified": False,
        },
    }
    scope["input_files"] = dict(sorted(inputs.hashes.items()))
    scope["safeguards"]["all_consumed_input_sha256_unchanged_after_audit"] = (
        inputs.verify_unchanged()
    )
    output.mkdir(parents=True, exist_ok=True)
    values = (
        json.dumps(scope, indent=2, ensure_ascii=False) + "\n",
        json.dumps(quality, indent=2, ensure_ascii=False) + "\n",
        json.dumps(family, indent=2, ensure_ascii=False) + "\n",
        private_csv,
        render_findings(scope, quality, family),
    )
    for name, content in zip(OUTPUT_NAMES, values, strict=True):
        target = output / name
        if target.is_symlink():
            raise ValueError("Output file must not be a symlink")
        with target.open("w", encoding="utf-8", newline="") as stream:
            stream.write(content)
    progress(
        f"Audit complete: {len(primary)} name-screen-positive validation conflicts; "
        f"{len(selected)} pending manual-review cases; sealed test untouched"
    )
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--source-dir", type=Path)
    args = parser.parse_args()
    run(args.repo, args.source_dir)


if __name__ == "__main__":
    main()
