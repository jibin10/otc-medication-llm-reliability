"""Synthetic source/contract tests; no real benchmark or sealed test data are read."""

import builtins
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from hrp import challenge_evaluation as c

MODULE = Path(__file__).resolve().parents[1] / "tools/build_challenge_v1.py"
SPEC = importlib.util.spec_from_file_location("build_challenge_v1", MODULE)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


@pytest.fixture
def resolver():
    return c.IdentityResolver(
        [
            {"name": "alpha", "uniis": ["0000000001"]},
            {"name": "alpha synonym", "uniis": ["0000000001"], "source": "same RxNorm concept"},
            {"name": "beta", "uniis": ["0000000002"]},
            {"name": "alpha hydrochloride", "uniis": ["0000000003"]},
            {"name": "ambiguous", "uniis": ["0000000001", "0000000002"]},
        ]
    )


def candidate(ndc="1-1", name="alpha", amount="1 mg/1", text=None, **kwargs):
    return {
        "product_ndc": ndc,
        "display_name": "FictionalBrand",
        "brand_name": "FictionalBrand",
        "generic_name": name,
        "dosage_form": "TABLET",
        "route": ["ORAL"],
        "fda_active_ingredients": [{"name": name, "strength": amount}],
        "spl": {
            "label_id": "label-" + ndc,
            "set_id": "set-" + ndc,
            "version": "1",
            "effective_time": "20200101",
            "active_ingredient_text": [text or name + " " + amount],
        },
        "rxnorm": {"cached_pairs_confirmed": True, "ingredient_pairs": []},
        "provenance": {
            "original_versions_confirmed": True,
            "raw_ndc_name_matches_catalog": True,
            "product_uniis": [],
        },
        **kwargs,
    }


def payload(*products, request=None):
    return {
        "contract_version": c.VERSION,
        "products": [
            {
                "request": request or {"name": "FictionalBrand"},
                "candidate_count": len(products),
                "candidates": list(products),
            }
        ],
    }


def comparison_payload(a, b):
    return {
        "contract_version": c.VERSION,
        "products": [payload(a)["products"][0], payload(b)["products"][0]],
    }


def record(case_id, reference, role="different_without_ndc"):
    return {
        "case_id": case_id,
        "partition": "validation",
        "reference": reference,
        "scoring_role": role,
        "category": "synthetic",
    }


def test_identity_synonyms_preserve_salts_and_ambiguous_mappings(resolver):
    assert resolver.identify(" ALPHA  ") == resolver.identify("alpha synonym")
    assert resolver.identify("alpha hydrochloride") != resolver.identify("alpha")
    assert resolver.identify("ambiguous") is None
    assert resolver.identify("invented") is None
    mapped, unknown = resolver.map_output(["alpha", "invented"])
    assert "unknown:invented" in mapped and unknown == ["invented"]


def test_source_review_detects_positive_extra_not_naive_missing_name(resolver):
    extra = c.assess_candidate(candidate(text="Alpha 1 mg Beta 2 mg"), resolver)
    assert extra["status"] == "CONFLICT"
    missing = c.assess_candidate(candidate(name="ambiguous", text="Alpha 1 mg"), resolver)
    assert missing["status"] == "UNRESOLVED"
    assert "native_name_to_unii_unresolved" in missing["reasons"]


def test_unknown_amount_and_inactive_text_are_not_silently_accepted(resolver):
    assert (
        c.assess_candidate(candidate(text="Alpha 1 mg Mystery 3%"), resolver)["status"]
        == "UNRESOLVED"
    )
    assert (
        c.assess_candidate(candidate(text="Alpha 1 mg Inactive ingredients Beta 5 mg"), resolver)[
            "status"
        ]
        == "SUPPORTED"
    )


def test_strength_basis_is_not_guessed_and_disagreements_are_recorded(resolver):
    item = candidate(amount="1 mg/1", text="Alpha 2 mg")
    result = c.assess_candidate(item, resolver)
    assert result["status"] == "SUPPORTED"  # Ingredient composition remains supported.
    assert result["strength_checks"][0]["status"] == "DIFFERENT_ON_COMPARABLE_BASIS"
    assert result["formulation_supported"] is False
    assert (
        c.predict_common_evidence(
            payload(item, request={"name": "FictionalBrand", "native_strengths": ["1 mg/1"]}),
            resolver,
        )["products"][0]["status"]
        == "SOURCE_CONFLICT"
    )
    percent = c.assess_candidate(candidate(amount="1 mg/1ml", text="Alpha 0.1%"), resolver)
    assert percent["strength_checks"][0]["status"] == "BASIS_UNRESOLVED"
    assert percent["formulation_supported"] is False


def test_ingredient_set_is_not_a_unique_formulation(resolver):
    item = payload(candidate(), candidate("1-2", amount="2 mg/1"))
    result = c.predict_common_evidence(item, resolver)
    assert result["products"][0]["status"] == "INGREDIENT_SET_IDENTIFIED"
    assert result["products"][0]["product_ndc"] is None
    assert result["products"][0]["formulation"] is None


def test_missing_spl_linkage_is_unresolved_not_a_demonstrated_conflict(resolver):
    item = candidate()
    item["provenance"].update(original_versions_confirmed=False, raw_ndc_name_matches_catalog=False)
    assert c.assess_candidate(item, resolver)["status"] == "UNRESOLVED"


def test_pharmacopeia_notation_does_not_hide_a_source_ingredient(resolver):
    result = c.assess_candidate(
        candidate(text="Active ingredient (in each tablet) Alpha USP, 1 mg"), resolver
    )
    assert result["status"] == "SUPPORTED"
    assert result["strength_checks"][0]["status"] == "MATCH"


def test_explicit_spoon_volume_is_used_but_never_assumed(resolver):
    explicit = c.assess_candidate(
        candidate(amount="80 mg/ml", text="Alpha 400 mg per teaspoon (5 mL)"), resolver
    )
    assert explicit["status"] == "SUPPORTED"
    assert explicit["strength_checks"][0]["status"] == "MATCH"
    unspecified = c.assess_candidate(
        candidate(amount="80 mg/ml", text="Alpha 400 mg per teaspoon"), resolver
    )
    assert unspecified["strength_checks"][0]["status"] == "BASIS_UNRESOLVED"


def test_ion_basis_annotation_is_recorded_without_chemical_equivalence():
    resolver = c.IdentityResolver([{"name": "sodium fluoride", "uniis": ["0000000001"]}])
    result = c.assess_candidate(
        candidate(
            name="sodium fluoride",
            amount="1.1 mg/g",
            text="Sodium fluoride 0.254% (0.15% w/v fluoride ion)",
        ),
        resolver,
    )
    assert result["status"] == "SUPPORTED"
    assert result["label_observations"]["basis_annotations_not_separate_ingredients"]
    assert result["strength_checks"][0]["status"] == "BASIS_UNRESOLVED"
    assert result["formulation_supported"] is False


def test_unspecified_percent_and_rxnorm_conflicts_withhold_full_formulation(resolver):
    percent = c.assess_candidate(candidate(amount="1%", text="Alpha 1%"), resolver)
    assert percent["status"] == "SUPPORTED" and percent["formulation_supported"] is False
    item = candidate()
    item["rxnorm"]["product_name"] = "alpha 2 mg oral capsule"
    assessment = c.assess_candidate(item, resolver)
    assert assessment["status"] == "SUPPORTED"
    assert assessment["formulation_supported"] is False
    assert (
        "rxnorm_strength_disagreement_full_formulation_withheld"
        in assessment["nonblocking_source_findings"]
    )


def test_real_ingredient_ambiguity_does_not_receive_a_unique_answer(resolver):
    item = payload(candidate(), candidate("1-2", name="beta"))
    result = c.predict_common_evidence(item, resolver)
    assert result["products"][0]["status"] == "INSUFFICIENT_INFORMATION"
    assert result["products"][0]["active_ingredient_names"] == []
    assert c.parse_output(c.canonical(result), item) == result


def test_all_candidates_and_full_text_are_shared_without_old_caps(resolver):
    products = [candidate(f"1-{i:02}") for i in range(25)]
    products[-1]["spl"]["active_ingredient_text"] = ["Alpha 1 mg " + "source " * 300]
    item = payload(*products)
    messages = c.build_messages(item)
    assert "1-24" in messages[1]["content"]
    assert products[-1]["spl"]["active_ingredient_text"][0] in messages[1]["content"]
    assert len(item["products"][0]["candidates"]) == 25
    c.assert_source_only(item)


@pytest.mark.parametrize(
    "key", ["reference", "reference_json", "review_verdict", "partition", "case_id"]
)
def test_reference_and_review_metadata_never_enter_prompts(key):
    item = payload(candidate())
    item["products"][0]["candidates"][0][key] = "forbidden"
    with pytest.raises(ValueError, match="metadata"):
        c.build_messages(item)


def test_prohibited_prefix_is_rejected_recursively():
    with pytest.raises(ValueError, match="prefix"):
        c.reject_reserved_fields({"nested": [{"ai_verdict": "forbidden"}]})


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(extra="bad"),
        lambda p: p["products"][0].update(status="MADE_UP"),
        lambda p: p["products"][0].update(source_ndcs=["unseen"]),
        lambda p: p["products"][0].update(active_ingredient_names="alpha"),
        lambda p: p["products"][0].update(product_ndc=123),
        lambda p: p["products"][0].update(
            formulation={"active_ingredients": [], "dosage_form": "TABLET", "route": []}
        ),
    ],
)
def test_nested_schema_rejects_malformed_outputs(resolver, mutate):
    item = payload(candidate())
    result = c.predict_common_evidence(item, resolver)
    mutate(result)
    with pytest.raises(ValueError):
        c.parse_output(result, item)


@pytest.mark.parametrize(
    "text",
    [
        '{"products":[],"products":[],"comparison":null}',
        '{"products":NaN,"comparison":null}',
        "```json\n{}\n```",
    ],
)
def test_parser_does_not_repair_duplicate_nonfinite_or_fenced_json(text):
    with pytest.raises(ValueError):
        c.parse_output(text, payload(candidate()))


def test_comparison_booleans_and_nullability_are_strict(resolver):
    item = comparison_payload(candidate(), candidate("2-2", name="beta"))
    result = c.predict_common_evidence(item, resolver)
    result["comparison"]["same_active_ingredient_set"] = 0
    with pytest.raises(ValueError, match="boolean"):
        c.parse_output(result, item)


def test_scorer_keeps_missing_and_invalid_outputs_in_all_denominators(resolver):
    item = comparison_payload(candidate(), candidate("2-2", name="beta"))
    ref = c.predict_common_evidence(item, resolver)
    cases = [record("v1", ref), record("v2", ref)]
    result = c.score_predictions(
        cases, {"v1": ref, "v2": "invalid"}, {"v1": item, "v2": item}, resolver
    )
    assert result["overall"]["ingredient_set_accuracy"] == {
        "numerator": 2,
        "denominator": 4,
        "value": 0.5,
    }
    assert result["overall"]["comparison_accuracy"]["denominator"] == 2
    assert result["overall"]["comparison_accuracy"]["value"] == 0.5
    assert result["overall"]["counts"]["invalid_outputs"] == 1


def test_shared_name_duplicates_are_invalid_as_required_by_published_schema(resolver):
    item = comparison_payload(candidate(), candidate("2-2"))
    result = c.predict_common_evidence(item, resolver)
    result["comparison"]["shared_active_ingredient_names"] = ["alpha", "alpha"]
    with pytest.raises(ValueError, match="shared names"):
        c.parse_output(result, item)


def test_unknown_names_are_false_positives_without_reference_matching(resolver):
    item = payload(candidate())
    ref = c.predict_common_evidence(item, resolver)
    pred = copy.deepcopy(ref)
    pred["products"][0]["active_ingredient_names"] = ["invented"]
    result = c.score_predictions([record("v", ref)], {"v": pred}, {"v": item}, resolver)
    counts = result["overall"]["counts"]
    assert counts["ingredient_fp"] == 1 and counts["ingredient_fn"] == 1
    assert counts["unknown_ingredient_names"] == 1
    assert result["overall"]["ingredient_set_accuracy"]["value"] == 0


def test_invalid_output_is_not_correct_abstention(resolver):
    item = payload(candidate(), candidate("1-2", name="beta"))
    ref = c.predict_common_evidence(item, resolver)
    result = c.score_predictions([record("v", ref, "ambiguity")], {}, {"v": item}, resolver)
    assert result["overall"]["abstention_accuracy"] == {
        "numerator": 0,
        "denominator": 1,
        "value": 0.0,
    }


def test_no_test_partition_can_be_scored_or_used_for_training(resolver):
    with pytest.raises(ValueError, match="validation-only"):
        c.score_predictions([{**record("x", {}), "partition": "test"}], {}, {}, resolver)
    with pytest.raises(ValueError, match="non-training"):
        c.filter_training_rows([{"case_id": "x", "partition": "validation"}], set(), set())


def test_training_allowlist_enforces_quarantine():
    rows = [
        {"case_id": "a", "partition": "train", "families": ["blocked"]},
        {"case_id": "b", "partition": "train", "families": ["clean"]},
    ]
    assert c.filter_training_rows(rows, {"b"}, {"blocked"}) == [rows[1]]
    with pytest.raises(ValueError, match="quarantine"):
        c.filter_training_rows(rows, {"a", "b"}, {"blocked"})


def test_semantic_leakage_keys_catch_reversed_pairs_and_equivalent_units():
    first = {"name": "Brand", "route": ["ORAL", "ORAL"], "native_strengths": ["1 g/1"]}
    second = {"name": "Other"}
    alias = {"name": "brand", "route": ["oral"], "native_strengths": ["1000 mg/1"]}
    assert c.semantic_input_key({"product_a": first, "product_b": second}) == c.semantic_input_key(
        {"product_a": second, "product_b": alias}
    )


def test_review_exclusions_are_preserved_despite_source_support():
    row = {"task": "ingredient_comparison", "ndcs": ["1-1"]}
    decision = builder.disposition(
        row,
        {"review_verdict": "EXCLUDE_PRIMARY"},
        {},
        {},
        {"1-1": {"reasons": []}},
        {},
        "synthetic",
    )
    assert decision[:3] == ("EXCLUDED", "accepted_review_exclusion", "none")


def test_unresolved_source_cannot_be_promoted_from_accepted_candidate():
    row = {"task": "ingredient_comparison", "ndcs": ["1-1"]}
    result = builder.disposition(
        row,
        {"review_verdict": "PRIMARY_CANDIDATE"},
        {},
        {},
        {"1-1": {"reasons": ["native_name_to_unii_unresolved"]}},
        {},
        "synthetic",
    )
    assert result[0] == "UNRESOLVED"


def test_disputed_input_strength_stays_out_of_high_confidence_scoring():
    row = {"task": "product_identification", "ndcs": ["1-1"]}
    result = builder.disposition(
        row,
        {"review_verdict": "SUPPLEMENTARY"},
        {},
        {"products": [{"status": "SOURCE_CONFLICT"}]},
        {"1-1": {"reasons": []}},
        {},
        "synthetic",
    )
    assert result[:3] == ("UNRESOLVED", "input_depends_on_disputed_strength", "none")


def test_negative_guard_prohibits_even_hashing_a_sealed_path(monkeypatch):
    monkeypatch.setattr(Path, "open", lambda *a, **k: pytest.fail("Forbidden path was opened"))
    with pytest.raises(ValueError, match="Held-out"):
        builder.prior.sha256_file(Path("data/processed/final_v5/sealed_test.csv"))


def test_synthetic_end_to_end_freezes_and_preserves_authoritative_reviews(tmp_path, monkeypatch):
    import test_revised_scope_validation_audit as fixtures

    repo, sources = fixtures.audit_fixture.__wrapped__(tmp_path)
    interim, processed = repo / "data/interim/final_v5", repo / "data/processed/final_v5"
    with (interim / "catalog_evidence.csv").open(encoding="utf-8", newline="") as f:
        products = list(csv.DictReader(f))
    names = sorted(
        {i["name"] for p in products for i in json.loads(p["original_ingredients_json"])}
    )
    uniis = {name: f"{i:010d}" for i, name in enumerate(names, 1)}
    with (interim / "otc_ndc_label_links.csv").open(encoding="utf-8", newline="") as f:
        labels = list(csv.DictReader(f))
    # Fully synthetic original source labels, including a clean shared-name ambiguity.
    for p, label in zip(products, labels, strict=True):
        label["active_ingredient_json"] = json.dumps(
            [
                "Active ingredients "
                + " ".join(
                    i["name"] + " " + i["strength"]
                    for i in json.loads(p["original_ingredients_json"])
                )
            ]
        )
    fixtures.write_csv(interim / "otc_ndc_label_links.csv", labels)
    fixtures.write_csv(
        interim / "fda_ingredient_identifiers.csv",
        [
            {"ingredient_name": name, "unii": unii, "source_ndcs_json": "[]"}
            for name, unii in uniis.items()
        ],
    )
    raw_labels, raw_ndcs = [], []
    for p, label in zip(products, labels, strict=True):
        ingredients = json.loads(p["original_ingredients_json"])
        raw_labels.append(
            {
                "id": label["label_id"],
                "set_id": label["label_set_id"],
                "version": "2",
                "effective_time": label["effective_time"],
                "active_ingredient": json.loads(label["active_ingredient_json"]),
            }
        )
        raw_ndcs.append(
            {
                "product_ndc": p["product_ndc"],
                "spl_id": label["label_id"],
                "brand_name": p["brand_name"],
                "generic_name": p["generic_name"],
                "dosage_form": p["dosage_form"],
                "route": json.loads(p["route"]),
                "active_ingredients": ingredients,
                "openfda": {"unii": [uniis[i["name"]] for i in ingredients]},
            }
        )
    for kind, values in (("label", raw_labels), ("ndc", raw_ndcs)):
        path = sources / f"raw/openfda/{kind}/drug-{kind}-0001-of-0001.json.zip"
        with ZipFile(path, "w") as z:
            z.writestr("records.json", json.dumps({"results": values}))
    rrf = sources / "raw/rxnorm/synthetic/rrf"
    rrf.mkdir(parents=True)
    lines = []
    for i, (name, unii) in enumerate(uniis.items(), 1):
        for sab, tty, code in (("RXNORM", "IN", str(i)), ("MTHSPL", "SU", unii)):
            f = [""] * 19
            f[0], f[1], f[11], f[12], f[13], f[14], f[16] = str(i), "ENG", sab, tty, code, name, "N"
            lines.append("|".join(f))
    (rrf / "RXNCONSO.RRF").write_text("\n".join(lines), encoding="utf-8")
    (rrf / "RXNREL.RRF").write_text("", encoding="utf-8")
    for name in (
        "tools/build_challenge_v1.py",
        "src/hrp/challenge_evaluation.py",
        "src/hrp/ingredient_agreement.py",
        "pyproject.toml",
    ):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Synthetic fixture implementation marker\n")
    validation = list(
        csv.DictReader((processed / "validation.csv").open(encoding="utf-8", newline=""))
    )
    review_choices = {
        "compare-identical": "PRIMARY_CANDIDATE",
        "compare-similar": "PRIMARY_CANDIDATE",
        "compare-control": "CONTROL_CANDIDATE",
        "ambiguous": "AMBIGUITY_CANDIDATE",
    }
    pending = [
        {"case_id": r["case_id"], "input_json": r["input_json"]}
        for r in validation
        if r["case_id"] in review_choices
    ]
    reviewed = [
        {
            "case_id": r["case_id"],
            "partition": "validation",
            "input_json": r["input_json"],
            "product_ndcs_json": json.dumps(json.loads(r["provenance_json"])["product_ndcs"]),
            "review_status": "REVIEWED",
            "review_verdict": review_choices[r["case_id"]],
            "reviewer_1_verdict": review_choices[r["case_id"]],
        }
        for r in validation
        if r["case_id"] in review_choices
    ]
    pending_path = repo / "results/revised_scope_audit/validation_review_queue.PRIVATE.csv"
    reviewed_path = pending_path.with_name("validation_review_queue.REVIEWED.csv")
    fixtures.write_csv(pending_path, pending)
    fixtures.write_csv(reviewed_path, reviewed)
    original_pending, original_reviewed = pending_path.read_bytes(), reviewed_path.read_bytes()
    baseline_path = repo / "results/final_v5/part04/validation_summary.json"
    baseline = json.loads(baseline_path.read_text())
    baseline["source_hashes"] = {
        f"data/interim/final_v5/{name}": builder.prior.sha256_file(interim / name)
        for name in builder.prior.SOURCE_FILES
    }
    baseline_path.write_text(json.dumps(baseline))
    output = repo / "results/challenge_v1"
    output.mkdir(parents=True)
    (output / "checks.junit.xml").write_text(
        '<testsuites><testsuite tests="40" failures="0" errors="0" skipped="0"/></testsuites>'
    )
    monkeypatch.setattr(
        builder,
        "EXPECTED_REVIEW",
        {"PRIMARY_CANDIDATE": 2, "CONTROL_CANDIDATE": 1, "AMBIGUITY_CANDIDATE": 1},
    )
    monkeypatch.setattr(
        builder,
        "POLICY",
        {
            **builder.POLICY,
            "target_different_ingredient_cases": 2,
            "target_controls": 1,
            "target_ambiguity": 1,
            "minimum_different_ingredient_cases": 1,
            "minimum_different_groups": 1,
            "minimum_different_without_ndc": 0,
            "minimum_without_ndc_groups": 0,
            "minimum_controls": 1,
            "minimum_ambiguity": 1,
        },
    )
    real_open = io.open

    def guarded_open(path, *args, **kwargs):
        if isinstance(path, (str, Path)):
            assert "sealed_test" not in str(path).casefold(), "Sealed test opened by build"
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    manifest = builder.build(repo, sources, progress=lambda _: None)
    assert manifest["status"] == "FROZEN", manifest["quality_checks"]
    assert (
        pending_path.read_bytes() == original_pending
        and reviewed_path.read_bytes() == original_reviewed
    )
    assert (output / "accepted_review_snapshot.PRIVATE.csv").read_bytes() == original_reviewed
    builder.verify_freeze(output)
    before = (output / "freeze_manifest.json").read_bytes()
    builder.build(repo, sources, progress=lambda _: None)
    assert (output / "freeze_manifest.json").read_bytes() == before
    with (output / "summary.json").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.verify_freeze(output, check_inputs=False)
