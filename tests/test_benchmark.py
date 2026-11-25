"""Focused checks for provisional family assignment and saved review decisions."""

import json
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from hrp.benchmark import (
    _quantity_key,
    _select_shortlist,
    assess_expansion_readiness,
    assess_split_readiness,
    assign_catalog_families,
    benchmark_references,
    build_family_audit,
    build_family_review_shortlist,
    catalog_answer,
    comparison_pairs,
    confirmed_expansion_reviews,
    coverage_split_preview,
    partition_families,
    reconcile_family_reviews,
    reconcile_reviews,
    reviewed_family_assignments,
    source_training_risks,
)
from hrp.benchmark import construct_benchmark_cases as construct_catalog_cases
from hrp.data_cleaning import normalize_name


def catalog_product(ndc, name, ingredient="MENTHOL", strength="5 mg/g", form="CREAM"):
    return {
        "product_ndc": ndc,
        "display_name": name,
        "brand_name": name,
        "generic_name": ingredient,
        "dosage_form": form,
        "route": '["TOPICAL"]',
        "ingredient_eligible": "True",
        "formulation_eligible": "True",
        "active_ingredients_json": json.dumps([{"name": ingredient, "strength": strength}]),
        "source_metadata_json": "[]",
        "label_set_id": "",
        "support_tier": "FDA_ONLY",
    }


def test_complete_catalog_grouping_protects_aliases_and_avoids_retailer_word_groups():
    products = {
        p["product_ndc"]: p
        for p in [
            catalog_product("a", "Alpha Relief"),
            catalog_product("b", "Alpha-Relief"),
            catalog_product("c", "Alpha Relief Cream"),
            catalog_product("d", "Store Sleep Aid"),
            catalog_product("e", "Store Cough Syrup"),
        ]
    }
    old = [{"product_ndc": "a", "family_group_id": "Alpha", "partition": "test"}]
    assigned, _links, _ = assign_catalog_families(products, old)
    assert assigned["a"] == assigned["b"] == assigned["c"]
    assert assigned["c"]["partition"] == "test"
    assert assigned["d"]["family_group_id"] != assigned["e"]["family_group_id"]
    reversed_assignments, _, _ = assign_catalog_families(
        dict(reversed(list(products.items()))), old
    )
    assert assigned == reversed_assignments
    protected, _, _ = assign_catalog_families(products, [], {"b"})
    assert all(protected[n]["partition"] == "test" for n in ("a", "b", "c"))


def test_all_eligible_products_and_multiple_distinct_family_cases_are_constructed():
    products = {
        str(i): catalog_product(
            str(i), "Alpha Relief", "MENTHOL" if i < 5 else "LIDOCAINE", f"{i + 1} mg/g"
        )
        for i in range(12)
    }
    assigned = {n: {"partition": "train", "family_group_id": "alpha"} for n in products}
    pairs, policy = comparison_pairs(products, assigned, [])
    cases, report = construct_catalog_cases(products, assigned, pairs)
    assert cases.loc[cases.case_type.eq("explicit_ndc")].shape[0] == 12
    assert cases.task.eq("ingredient_comparison").sum() > 3
    assert cases.task.eq("product_identification").sum() > 2
    assert report["all_ingredient_eligible_products_considered"] == 12
    assert not cases[["task", "input_json"]].duplicated().any()
    assert policy["selected_unique_pairs"] < 12 * 11 // 2
    assert set(cases.partition) == {"train"}


def test_full_catalog_uniqueness_does_not_force_a_reference_ndc():
    a = catalog_product("a", "Example")
    b = catalog_product("b", "Example")
    products = {"a": a, "b": b}
    answer, _ids = catalog_answer({"name": "Example"}, products, {"example": {"a", "b"}})
    assert answer["resolution_level"] == "FORMULATION"
    assert answer["product"]["product_ndc"] is None
    products["b"] = catalog_product("b", "Example", "LIDOCAINE")
    answer, _ = catalog_answer({"name": "Example"}, products, {"example": {"a", "b"}})
    assert answer["answer_status"] == "INSUFFICIENT_INFORMATION"
    products["b"]["ingredient_eligible"] = "False"
    answer, _ = catalog_answer({"name": "Example"}, products, {"example": {"a", "b"}})
    assert answer["answer_status"] == "SOURCE_CONFLICT"


def test_uncertain_strength_cannot_eliminate_a_plausible_catalog_product():
    products = {
        "a": catalog_product("a", "Example"),
        "b": catalog_product("b", "Example", "LIDOCAINE", "10 mg/g"),
    }
    products["b"]["formulation_eligible"] = "False"
    products["b"]["strength_issues_json"] = '["label_strength_basis_unresolved"]'
    answer, ids = catalog_answer(
        {"name": "Example", "native_strengths": ["5 mg/g"]}, products, {"example": {"a", "b"}}
    )
    assert set(ids) == {"a", "b"}
    assert answer["answer_status"] == "INSUFFICIENT_INFORMATION"


def test_repeated_routes_and_equivalent_quantity_spellings_cannot_force_an_ndc():
    from hrp.benchmark import canonical_case_input

    a, b = catalog_product("a", "Example"), catalog_product("b", "Example")
    b["route"] = '["TOPICAL", "TOPICAL"]'
    request = {"name": "Example", "route": ["TOPICAL", "TOPICAL"], "native_strengths": ["5 mg/g"]}
    answer, ids = catalog_answer(request, {"a": a, "b": b}, {"example": {"a", "b"}})
    assert answer["resolution_level"] == "FORMULATION"
    assert answer["product"]["product_ndc"] is None
    assert set(ids) == {"a", "b"}
    equivalent = {"name": " EXAMPLE ", "route": ["topical"], "native_strengths": ["0.005 g/g"]}
    assert canonical_case_input(request) == canonical_case_input(equivalent)
    pair = {"product_a": request, "product_b": {"name": "Other"}}
    reverse = {"product_a": {"name": "OTHER"}, "product_b": equivalent}
    assert canonical_case_input(pair) == canonical_case_input(reverse)


def review_pair(
    left: str, right: str, decision: str = "VERIFIED", family: str = "Family One"
) -> dict[str, str]:
    return {
        "case_type": "similar_different",
        "product_ndc_a": left,
        "product_ndc_b": right,
        "name_a": "",
        "name_b": "",
        "display_name_a": "",
        "display_name_b": "",
        "review_decision": decision,
        "verified_family": family,
        "review_notes": "Keep the original review text.",
        "fda_ingredients_a": "[]",
        "fda_ingredients_b": "[]",
        "dosage_a": "TABLET",
        "dosage_b": "TABLET",
    }


def test_assignments_preserve_labels_and_do_not_accept_other_decisions():
    reviewed = pd.DataFrame(
        [
            review_pair("001-01", "001-02", family="Manual FAMILY label"),
            review_pair("002-01", "002-02", "UNCERTAIN", "Tentative label"),
            review_pair("003-01", "003-02", "REJECTED", "Rejected label"),
            review_pair("004-01", "004-02", "PENDING", "Pending label"),
        ]
    )
    original = reviewed.copy(deep=True)
    assert reviewed_family_assignments(reviewed) == {
        "001-01": ["Manual FAMILY label"],
        "001-02": ["Manual FAMILY label"],
    }
    assert reconcile_reviews(reviewed)["saved_counts"] == {
        "VERIFIED": 1,
        "UNCERTAIN": 1,
        "REJECTED": 1,
        "PENDING": 1,
    }
    pd.testing.assert_frame_equal(reviewed, original)


def test_verified_without_family_remains_unassigned():
    reviewed = pd.DataFrame([review_pair("001-01", "001-02", family="  ")])
    assert reviewed_family_assignments(reviewed) == {}
    assert reconcile_reviews(reviewed)["verified_pairs_without_family"] == 1


@pytest.mark.parametrize("decision", ["", "verified", "ACCEPTED"])
def test_unknown_decisions_are_rejected(decision: str):
    with pytest.raises(ValueError, match="Unexpected review decisions"):
        reconcile_reviews(pd.DataFrame([review_pair("001-01", "001-02", decision)]))


def test_reversed_duplicate_review_is_rejected():
    reviewed = pd.DataFrame(
        [review_pair("001-01", "001-02"), review_pair("001-02", "001-01", "REJECTED")]
    )
    with pytest.raises(ValueError, match="Duplicate reviewed pair"):
        reconcile_reviews(reviewed)


def write_inputs(
    directory: Path,
    names: list[str],
    review_rows: list[dict[str, str]],
    links: list[tuple[int, int]] | None = None,
) -> dict[str, Path]:
    """Build a small catalog with leading-zero identifiers and separate evidence."""
    ndcs = [f"001-{index:02}" for index in range(len(names))]
    ingredients = json.dumps([{"name": "Ingredient A", "strength": "1 mg/1"}])
    catalog = pd.DataFrame(
        [
            {
                "product_ndc": ndc,
                "normalized_name": normalize_name(name),
                "display_name": name,
                "name_source": "brand",
                "generic_name": "Ingredient A",
                "dosage_form": "TABLET",
                "route": '["ORAL"]',
                "active_ingredients_json": ingredients,
                "marketing_category": "OTC MONOGRAPH DRUG",
            }
            for ndc, name in zip(ndcs, names, strict=True)
        ]
    )
    agreement = pd.DataFrame(
        [
            {
                "product_ndc": ndc,
                "catalog_status": "consistent",
                "agreement_status": "exact_base_names",
                "rxnorm_ingredient_status": "resolved_path",
                "rxnorm_in_names_json": '["ingredient a"]',
                "rxnorm_pin_names_json": "[]",
                "label_text_available": "True",
            }
            for ndc in ndcs
        ]
    )
    for row in review_rows:
        for side in ("a", "b"):
            index = ndcs.index(row[f"product_ndc_{side}"])
            row[f"name_{side}"] = normalize_name(names[index])
            row[f"display_name_{side}"] = names[index]
            row[f"fda_ingredients_{side}"] = ingredients
    pair_columns = ["case_type", "product_ndc_a", "product_ndc_b", "name_a", "name_b"]
    pairs = [
        {
            "case_type": "similar_different",
            "product_ndc_a": ndcs[left],
            "product_ndc_b": ndcs[right],
            "name_a": normalize_name(names[left]),
            "name_b": normalize_name(names[right]),
        }
        for left, right in (links or [])
    ]
    pairs.extend({key: row[key] for key in pair_columns} for row in review_rows)
    frames = {
        "catalog": catalog,
        "agreement": agreement,
        "reviewed": pd.DataFrame(review_rows),
        "identical_pairs": pd.DataFrame(columns=["normalized_name", "ndc_a", "ndc_b"]),
        "similar_pairs": pd.DataFrame(pairs, columns=pair_columns),
        "controls": pd.DataFrame(columns=pair_columns),
    }
    paths = {}
    for name, frame in frames.items():
        path = directory / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[f"{name}_path"] = path
    return paths


def load_review(output: Path) -> list[dict]:
    frame = pd.read_csv(output / "family_review.csv", dtype=str, keep_default_na=False)
    return [
        {key: json.loads(value) if key.endswith("_json") else value for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def test_aliases_and_label_matches_never_propagate_assignments(tmp_path: Path):
    paths = write_inputs(
        tmp_path,
        ["Aster Plus", "Aster Plus PM", "Aster-Plus", "Aster Plus Gel", "Aster Other"],
        [review_pair("001-00", "001-01", family="ASTER PLUS")],
    )
    original_files = {key: path.read_bytes() for key, path in paths.items()}
    output = tmp_path / "output"
    summary = build_family_audit(**paths, output_dir=output)
    (row,) = load_review(output)
    assert row["manual_family_labels_json"] == ["ASTER PLUS"]
    assert row["assigned_products_json"] == {"001-00": ["ASTER PLUS"], "001-01": ["ASTER PLUS"]}
    assert row["unassigned_product_ndcs_json"] == ["001-02", "001-03"]
    assert row["assignment_status"] == "PARTIALLY_ASSIGNED"
    assert "aster other" not in row["normalized_names_json"]
    assert summary["scope"]["catalog_products_outside_audit"] == 1
    assert not summary["readiness"]["ready_for_part_3_2_split_freeze"]
    assert original_files == {key: path.read_bytes() for key, path in paths.items()}
    original_output = (output / "family_review.csv").read_bytes()
    build_family_audit(**paths, output_dir=output)
    assert (output / "family_review.csv").read_bytes() == original_output


@pytest.mark.parametrize("decision", ["UNCERTAIN", "REJECTED", "PENDING"])
def test_generic_names_and_unaccepted_reviews_remain_unresolved(tmp_path: Path, decision: str):
    paths = write_inputs(
        tmp_path,
        ["Ingredient A", "Ingredient A Gel", "Ingredient A"],
        [review_pair("001-00", "001-01", decision, "Tentative FAMILY")],
    )
    output = tmp_path / "output"
    summary = build_family_audit(**paths, output_dir=output)
    (row,) = load_review(output)
    assert row["assignment_status"] == "UNRESOLVED"
    assert row["assigned_products_json"] == {}
    assert row["manual_family_labels_json"] == []
    assert row["reviewed_pairs_json"][0]["verified_family"] == "Tentative FAMILY"
    assert row["reviewed_pairs_json"][0]["review_decision"] == decision
    assert row["generic_like_product_count"] == "3"
    assert summary["families"]["unassigned_products"] == 3
    assert summary["families"]["manually_labeled_seed_families"] == 0


def test_similarity_links_and_conflicting_manual_labels_require_review(tmp_path: Path):
    paths = write_inputs(
        tmp_path,
        ["Aster", "Aster PM", "Birch", "Birch Gel"],
        [
            review_pair("001-00", "001-01", family="ASTER"),
            review_pair("001-00", "001-02", family="BIRCH"),
        ],
        links=[(2, 3)],
    )
    output = tmp_path / "output"
    summary = build_family_audit(**paths, output_dir=output)
    (row,) = load_review(output)
    assert row["assignment_status"] == "CONFLICT"
    assert row["assigned_products_json"]["001-00"] == ["ASTER", "BIRCH"]
    assert row["conflicting_product_ndcs_json"] == ["001-00"]
    assert row["unassigned_product_ndcs_json"] == ["001-03"]
    assert "multiple_manual_labels_in_review_group" in row["review_reasons_json"]
    assert summary["families"]["conflicting_product_assignments"] == 1


def test_changed_review_snapshot_is_flagged_without_changing_decision(tmp_path: Path):
    paths = write_inputs(
        tmp_path, ["Aster", "Aster PM"], [review_pair("001-00", "001-01", family="ASTER")]
    )
    reviewed = pd.read_csv(paths["reviewed_path"], dtype=str, keep_default_na=False)
    reviewed.loc[0, "name_a"] = "Old name"
    reviewed.loc[0, "fda_ingredients_a"] = '[{"name": "Old ingredient", "strength": "2 mg/1"}]'
    reviewed.to_csv(paths["reviewed_path"], index=False)
    output = tmp_path / "output"
    summary = build_family_audit(**paths, output_dir=output)
    (row,) = load_review(output)
    assert row["assignment_status"] == "CONFLICT"
    assert row["reviewed_pairs_json"][0]["review_decision"] == "VERIFIED"
    assert summary["data_quality"]["reviewed_snapshot_conflicts_by_ndc"] == {
        "001-00": [
            "reviewed_formulation_differs_from_catalog",
            "reviewed_name_differs_from_catalog",
        ]
    }


def test_duplicate_catalog_ndc_fails_before_writing_outputs(tmp_path: Path):
    paths = write_inputs(
        tmp_path, ["Aster", "Aster PM"], [review_pair("001-00", "001-01", family="ASTER")]
    )
    catalog = pd.read_csv(paths["catalog_path"], dtype=str, keep_default_na=False)
    pd.concat([catalog, catalog.iloc[:1]]).to_csv(paths["catalog_path"], index=False)
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="duplicate product NDC"):
        build_family_audit(**paths, output_dir=output)
    assert not output.exists()


def shortlist_inputs(directory: Path) -> tuple[Path, Path]:
    """Provide reviewed seeds, new candidates and control-only groups."""
    interim = directory / "data/interim"
    interim.mkdir(parents=True)
    paths = write_inputs(
        interim,
        [
            "Aster Plus",
            "Aster Plus PM",
            "Aster-Plus",
            "Aster Plus Gel",
            "Ingredient A",
            "Ingredient A",
            "Birch",
            "Birch PM",
            "Cedar",
            "Cedar Gel",
            "Elm",
            "Elm PM",
            "Palm",
            "Palm Night",
        ],
        [review_pair("001-00", "001-01", family="ASTER PLUS")],
        links=[(2, 3), (6, 7), (10, 11)],
    )
    filenames = {
        "catalog_path": "otc_catalog_consistent.csv",
        "agreement_path": "otc_ingredient_agreement.csv",
        "reviewed_path": "feasibility_review_queue_reviewed.csv",
        "identical_pairs_path": "identical_name_candidate_pairs.csv",
        "similar_pairs_path": "similar_name_candidate_pairs.csv",
        "controls_path": "similar_name_same_ingredient_controls.csv",
    }
    for key, filename in filenames.items():
        paths[key] = paths[key].rename(interim / filename)
    catalog = pd.read_csv(paths["catalog_path"], dtype=str, keep_default_na=False)
    agreement = pd.read_csv(paths["agreement_path"], dtype=str, keep_default_na=False)
    ingredient_sets = [["Ingredient A"] for _ in catalog.index]
    for index in (1, 5, 11):
        ingredient_sets[index] = ["Ingredient B"]
    for index in (3, 7):
        ingredient_sets[index] = ["Ingredient A", "Ingredient B"]
    ingredient_ids = [[str(ord(name[-1])) for name in names] for names in ingredient_sets]
    catalog["active_ingredients_json"] = [
        json.dumps([{"name": name, "strength": "1 mg/1"} for name in names])
        for names in ingredient_sets
    ]
    catalog["source_rows"] = [str(index + 1) for index in catalog.index]
    catalog.to_csv(paths["catalog_path"], index=False)
    agreement["rxnorm_in_rxcuis_json"] = [json.dumps(ids) for ids in ingredient_ids]
    agreement["rxnorm_in_names_json"] = [json.dumps(names) for names in ingredient_sets]
    agreement.loc[3, "agreement_status"] = "review_partial_name_overlap"
    agreement.loc[7, "agreement_status"] = "rxnorm_unavailable"
    agreement.to_csv(paths["agreement_path"], index=False)
    reviewed = pd.read_csv(paths["reviewed_path"], dtype=str, keep_default_na=False)
    reviewed.loc[0, "fda_ingredients_b"] = catalog.loc[1, "active_ingredients_json"]
    reviewed.to_csv(paths["reviewed_path"], index=False)
    identical = pd.DataFrame(
        [
            {
                "normalized_name": "ingredient a",
                "ndc_a": "001-04",
                "ndc_b": "001-05",
                "in_rxcuis_a_json": json.dumps(ingredient_ids[4]),
                "in_rxcuis_b_json": json.dumps(ingredient_ids[5]),
            }
        ]
    )
    identical.to_csv(paths["identical_pairs_path"], index=False)
    similar = pd.read_csv(paths["similar_pairs_path"], dtype=str, keep_default_na=False)
    similar["ingredient_overlap"] = [
        "shared"
        if set(ingredient_ids[int(row.product_ndc_a[-2:])])
        & set(ingredient_ids[int(row.product_ndc_b[-2:])])
        else "disjoint"
        for row in similar.itertuples()
    ]
    similar.to_csv(paths["similar_pairs_path"], index=False)
    controls = pd.DataFrame(
        [
            {
                "case_type": "similar_same_control",
                "product_ndc_a": f"001-{left:02}",
                "product_ndc_b": f"001-{right:02}",
                "name_a": catalog.loc[left, "normalized_name"],
                "name_b": catalog.loc[right, "normalized_name"],
                "ingredient_overlap": "shared",
            }
            for left, right in [(8, 9), (12, 13)]
        ]
    )
    controls.to_csv(paths["controls_path"], index=False)
    pd.DataFrame(
        [
            {
                "product_ndc": "001-02",
                "link_status": "matched",
                "label_id": "saved-label-id",
                "label_set_id": "saved-set-id",
                "effective_time": "20260101",
            }
        ]
    ).to_csv(interim / "otc_ndc_label_links.csv", index=False)
    audit_dir = directory / "results/part03"
    build_family_audit(**paths, output_dir=audit_dir)
    return interim, audit_dir


def load_shortlist(audit_dir: Path) -> pd.DataFrame:
    return pd.read_csv(audit_dir / "family_review_shortlist.csv", dtype=str, keep_default_na=False)


def test_shortlist_includes_new_controls_and_preserves_review_evidence(tmp_path: Path):
    interim, audit_dir = shortlist_inputs(tmp_path)
    originals = {
        path: path.read_bytes()
        for folder in (interim, audit_dir)
        for path in folder.iterdir()
        if path.is_file()
    }
    summary = build_family_review_shortlist(interim, audit_dir)
    shortlist = load_shortlist(audit_dir)
    assert summary["shortlist_size"] == 6
    assert summary["previously_reviewed_pairs_excluded"] == 1
    assert summary["preserved_manual_family_labels"] == 1
    assert shortlist.review_group_id.nunique() == len(shortlist)
    assert set(shortlist.case_type) == {
        "identical_different",
        "similar_different",
        "similar_same_control",
    }
    assert not any(
        {row.product_ndc_a, row.product_ndc_b} == {"001-00", "001-01"}
        for row in shortlist.itertuples()
    )
    controls = shortlist.loc[shortlist.case_type.eq("similar_same_control")]
    assert len(controls) == 2
    assert all(json.loads(value) == [] for value in controls.existing_group_ids_json)
    assert (
        shortlist[["reviewer_decision", "confirmed_family_id", "reviewer_notes"]].eq("").all().all()
    )
    for row in shortlist.itertuples():
        if row.product_ndc_a == "001-02":
            assert json.loads(row.source_references_a_json)["fda_label_link"] == {
                "link_status": "matched",
                "label_id": "saved-label-id",
                "label_set_id": "saved-set-id",
                "effective_time": "20260101",
            }
            assert json.loads(row.source_references_b_json)["fda_label_link"] == {}
            assert json.loads(row.existing_family_labels_json) == ["ASTER PLUS"]
            (saved,) = json.loads(row.original_reviews_json)
            assert saved["review_decision"] == "VERIFIED"
            assert saved["verified_family"] == "ASTER PLUS"
            assert saved["review_notes"] == "Keep the original review text."
    assert originals == {path: path.read_bytes() for path in originals}
    original_output = (audit_dir / "family_review_shortlist.csv").read_bytes()
    assert build_family_review_shortlist(interim, audit_dir) == summary
    assert (audit_dir / "family_review_shortlist.csv").read_bytes() == original_output


def test_shortlist_separates_disagreement_from_missing_mapping(tmp_path: Path):
    interim, audit_dir = shortlist_inputs(tmp_path)
    build_family_review_shortlist(interim, audit_dir)
    for row in load_shortlist(audit_dir).itertuples():
        flags = json.loads(row.review_flags_json)
        if row.product_ndc_a == "001-02":
            assert "ingredient_disagreement" in flags
            supporting = json.loads(row.supporting_products_json)
            assert any(product["product_ndc"] == "001-03" for product in supporting)
        if row.product_ndc_a == "001-06":
            assert "ingredient_mapping_unavailable" in flags
            assert "ingredient_disagreement" not in flags


def test_selection_is_order_independent_and_does_not_weight_pair_counts():
    groups = [
        {
            "review_group_id": f"group_{index:02}",
            "flags": ["possible_family_alias", "potentially_ambiguous_name"],
            "pairs": [
                {
                    "case_type": case_type,
                    "ingredient_overlap": overlap,
                    "product_ndc_a": f"{index:03}-01",
                    "product_ndc_b": f"{index:03}-02",
                }
            ],
        }
        for index, (case_type, overlap) in enumerate(
            [
                (kind, overlap)
                for _ in range(4)
                for kind, overlap in [
                    ("identical_different", "shared"),
                    ("identical_different", "disjoint"),
                    ("similar_different", "shared"),
                    ("similar_different", "disjoint"),
                    ("similar_same_control", "shared"),
                ]
            ]
        )
    ]
    first = _select_shortlist(groups, 15)
    expected = [(row["group"]["review_group_id"], row["selection_category"]) for row in first]
    groups[0]["pairs"] *= 200
    second = _select_shortlist(list(reversed(groups)), 15)
    assert [
        (row["group"]["review_group_id"], row["selection_category"]) for row in second
    ] == expected
    assert len({row["group"]["review_group_id"] for row in first}) == 15
    assert {row["pair"]["case_type"] for row in first} == {
        "identical_different",
        "similar_different",
        "similar_same_control",
    }
    assert {row["pair"]["ingredient_overlap"] for row in first} == {"shared", "disjoint"}


def test_shortlist_does_not_overwrite_reviewer_entries(tmp_path: Path):
    interim, audit_dir = shortlist_inputs(tmp_path)
    build_family_review_shortlist(interim, audit_dir)
    shortlist = load_shortlist(audit_dir)
    shortlist.loc[0, "reviewer_notes"] = "Keep this manual annotation"
    path = audit_dir / "family_review_shortlist.csv"
    shortlist.to_csv(path, index=False)
    original = path.read_bytes()
    with pytest.raises(ValueError, match="has reviewer entries"):
        build_family_review_shortlist(interim, audit_dir)
    assert path.read_bytes() == original


def test_shortlist_rejects_stale_audit_inputs(tmp_path: Path):
    interim, audit_dir = shortlist_inputs(tmp_path)
    path = interim / "similar_name_candidate_pairs.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Audit input changed"):
        build_family_review_shortlist(interim, audit_dir)
    assert not (audit_dir / "family_review_shortlist.csv").exists()


@pytest.mark.parametrize("max_groups", [-1, 31, 2.5])
def test_shortlist_enforces_batch_limit(tmp_path: Path, max_groups: int):
    with pytest.raises(ValueError, match="max_groups"):
        build_family_review_shortlist(tmp_path, tmp_path, max_groups=max_groups)


def test_zero_size_shortlist_keeps_csv_schema(tmp_path: Path):
    interim, audit_dir = shortlist_inputs(tmp_path)
    summary = build_family_review_shortlist(interim, audit_dir, max_groups=0)
    shortlist = load_shortlist(audit_dir)
    assert summary["shortlist_size"] == 0
    assert shortlist.empty
    assert {"reviewer_decision", "confirmed_family_id", "reviewer_notes"} <= set(shortlist.columns)


def family_review(
    a: str,
    b: str,
    family_a: str,
    family_b: str,
    decision: str = "SAME_FAMILY",
    source_flag: str = "TRUE",
    rank: str = "1",
) -> dict[str, str]:
    return {
        "review_rank": rank,
        "product_ndc_a": a,
        "product_ndc_b": b,
        "review_decision": decision,
        "family_id_a": family_a,
        "family_id_b": family_b,
        "labels_verified": source_flag,
        "reviewer_notes": "Original family review note",
    }


def test_family_reconciliation_preserves_seeds_and_source_holds():
    seeds = pd.DataFrame([review_pair("001-01", "001-02", family="LEGACY")])
    reviewed = pd.DataFrame(
        [
            family_review("001-01", "001-03", "LEGACY", "LEGACY", source_flag="FALSE"),
            family_review("002-01", "002-02", "UNCONFIRMED", "UNCONFIRMED", "UNCERTAIN", rank="2"),
            family_review("003-01", "003-02", "EXCLUDED", "EXCLUDED", "EXCLUDE", rank="3"),
        ]
    )
    original_seeds, original_reviews = seeds.copy(deep=True), reviewed.copy(deep=True)
    result = reconcile_family_reviews(seeds, reviewed)
    assert result["assignments"] == {
        "001-01": ["LEGACY"],
        "001-02": ["LEGACY"],
        "001-03": ["LEGACY"],
    }
    assert result["source_holds"] == {"001-01", "001-03"}
    assert result["records"][("001-01", "001-02")]["decision"] == "VERIFIED"
    assert result["records"][("002-01", "002-02")]["decision"] == "UNCERTAIN"
    pd.testing.assert_frame_equal(seeds, original_seeds)
    pd.testing.assert_frame_equal(reviewed, original_reviews)


def test_possible_giltuss_alias_is_quarantined_without_merging():
    seeds = pd.DataFrame([review_pair("001-01", "001-02", family="GILTUSS ALLERGY PLUS")])
    reviewed = pd.DataFrame([family_review("001-03", "001-04", "GILTUSS", "GILTUSS")])
    result = reconcile_family_reviews(seeds, reviewed)
    assert len(result["aliases"]) == 1
    assert result["alias_holds"] == {"001-01", "001-02", "001-03", "001-04"}
    assert result["assignments"]["001-01"] == ["GILTUSS ALLERGY PLUS"]
    assert result["assignments"]["001-03"] == ["GILTUSS"]
    assert len(result["family_labels"]) == 2


def test_conflicting_reviewed_product_ids_are_not_overwritten():
    seeds = pd.DataFrame([review_pair("001-01", "001-02", family="LEGACY")])
    reviewed = pd.DataFrame([family_review("001-01", "001-03", "NEW FAMILY", "NEW FAMILY")])
    result = reconcile_family_reviews(seeds, reviewed)
    assert result["conflicts"] == {"001-01": ["LEGACY", "NEW FAMILY"]}
    assert result["assignments"]["001-01"] == ["LEGACY", "NEW FAMILY"]


@pytest.mark.parametrize(
    "decision,left,right",
    [
        ("SAME_FAMILY", "ONE", "TWO"),
        ("DIFFERENT_FAMILIES", "ONE", "ONE"),
        ("SAME_FAMILY", "", ""),
        ("ACCEPTED", "ONE", "ONE"),
    ],
)
def test_invalid_family_reviews_cannot_add_assignments(decision: str, left: str, right: str):
    seeds = pd.DataFrame([review_pair("001-01", "001-02", family="LEGACY")])
    reviewed = pd.DataFrame([family_review("002-01", "002-02", left, right, decision)])
    result = reconcile_family_reviews(seeds, reviewed)
    assert result["errors"]
    assert "002-01" not in result["assignments"]


def test_explicit_seed_identity_followup_remains_on_hold():
    row = review_pair("001-01", "001-02", family="MERIDIAN UNWRAPPED")
    row["review_notes"] = "Manufacturer/brand identity remains to be confirmed for final benchmark."
    seeds = pd.DataFrame([row])
    reviewed = pd.DataFrame(columns=family_review("a", "b", "A", "A"))
    result = reconcile_family_reviews(seeds, reviewed)
    assert result["identity_holds"] == {"001-01", "001-02"}
    assert result["assignments"]["001-01"] == ["MERIDIAN UNWRAPPED"]


def test_family_partitions_are_deterministic_and_round_by_family():
    families = [f"family_{index:02}" for index in range(24)]
    result = partition_families(families)
    assert result == partition_families(list(reversed(families)) + families[:2])
    assert set(result) == set(families)
    assert Counter(result.values()) == {"train": 17, "validation": 4, "test": 3}
    assert result == partition_families(families, seed=42)


def test_readiness_scans_full_pool_without_forcing_a_manifest(tmp_path: Path):
    interim, audit_dir = shortlist_inputs(tmp_path)
    reviewed = pd.DataFrame(
        [
            family_review("001-08", "001-09", "CEDAR", "CEDAR"),
            family_review("001-12", "001-13", "PALM", "PALM", "UNCERTAIN", rank="2"),
        ]
    )
    baseline = reviewed.copy()
    for field in [
        "review_decision",
        "family_id_a",
        "family_id_b",
        "labels_verified",
        "reviewer_notes",
    ]:
        baseline[field] = ""
    baseline.to_csv(audit_dir / "family_review_decisions.csv", index=False)
    reviewed.to_csv(audit_dir / "family_review_decisions_reviewed.csv", index=False)
    protected = {
        path: path.read_bytes()
        for folder in (interim, audit_dir)
        for path in folder.iterdir()
        if path.is_file()
    }
    summary = assess_split_readiness(tmp_path)
    assert summary["readiness"] == "NOT_READY"
    assert summary["review_validation"]["invalid_rows"] == []
    assert summary["review_validation"]["source_snapshot_matches"]
    assert summary["full_pool_coverage"]["candidate_records"] == 7
    assert (
        summary["full_pool_coverage"]["categories"]["similar_same_control"]["candidate_records"]
        == 2
    )
    assert summary["full_pool_coverage"]["review_status_counts"]["uncertain_review"] == 1
    assert summary["family_reconciliation"]["usable_core_families"] == 2
    assert summary["family_reconciliation"]["usable_core_products"] == 4
    assert "001-12" not in summary["family_reconciliation"]["assignments"]
    assert all(summary["split_policy"]["preview_checks"].values())
    assert summary["split_policy"]["preview_only"]
    assert not (audit_dir / "family_split_manifest.csv").exists()
    assert protected == {path: path.read_bytes() for path in protected}
    assert json.loads((audit_dir / "split_readiness.json").read_text())["readiness"] == "NOT_READY"


def test_native_strengths_preserve_mass_volume_and_item_bases():
    assert _quantity_key("160 mg/5 mL") == _quantity_key("32 mg/mL")
    assert _quantity_key("1 g/100 g") == _quantity_key("10 mg/g")
    assert _quantity_key("40 mg/g") != _quantity_key("40 mg/mL")
    assert _quantity_key("1.3 mg/1") != _quantity_key("1.3 mg/mL")
    assert _quantity_key("0.13%") is None
    assert _quantity_key("1 mg/0 mL") is None


@pytest.mark.parametrize(
    "strength,route,form,drug_name,expected",
    [
        ("160 mg/5 mL", "ORAL", "SUSPENSION", "Ingredient A 32 MG/ML Oral Suspension", []),
        (
            "80 mg/g",
            "TOPICAL",
            "CREAM",
            "Ingredient A 80 MG/ML Topical Cream",
            ["strength_or_basis_requires_review"],
        ),
        (
            "5 mg/mL",
            "TOPICAL",
            "SPRAY",
            "Ingredient A 5 MG/ML Rectal Spray",
            ["route_or_form_requires_review"],
        ),
        (
            "1 mg/1",
            "ORAL",
            "CAPSULE",
            "Ingredient A 1 MG Oral Tablet",
            ["route_or_form_requires_review"],
        ),
    ],
)
def test_training_screen_requires_native_strength_route_and_form(
    strength: str, route: str, form: str, drug_name: str, expected: list[str]
):
    product = {
        "catalog_status": "consistent",
        "agreement_status": "exact_base_names",
        "rxnorm_ingredient_status": "resolved_path",
        "active_ingredients_json": json.dumps([{"name": "Ingredient A", "strength": strength}]),
        "route": json.dumps([route]),
        "dosage_form": form,
    }
    drug = {"rxnorm_status": "matched", "rxnorm_drug_name": drug_name}
    label = {
        "link_status": "matched",
        "label_id": "label",
        "label_set_id": "set",
        "active_ingredient_json": '["Ingredient A"]',
    }
    assert source_training_risks(product, drug, label) == expected
    product["rxnorm_ingredient_status"] = "ambiguous"
    assert "ingredient_or_product_mapping_unresolved" in source_training_risks(product, drug, label)
    assert "label_reference_incomplete" in source_training_risks(product, drug, {})


def test_preview_is_deterministic_and_keeps_related_ids_together():
    families, cases = {}, []
    for index in range(40):
        a, b = f"{index}-a", f"{index}-b"
        families[a] = families[b] = f"family-{index:02}"
        for category in ("identical_different", "similar_different", "similar_same_control"):
            cases.append(
                {
                    "product_ndc_a": a,
                    "product_ndc_b": b,
                    "case_type": category,
                    "ingredient_overlap": "shared"
                    if category.endswith("control") or index % 2
                    else "disjoint",
                }
            )
    constraints = [["family-00", "family-01"]]
    result = coverage_split_preview(families, cases, constraints)
    assert result == coverage_split_preview(
        dict(reversed(list(families.items()))), cases[::-1], constraints
    )
    assert result["independent_family_groups"] == 39
    assert result["family_labels"] == 40
    assert result["family_partitions"]["family-00"] == result["family_partitions"]["family-01"]
    assert result["coverage_sufficient"]
    assert result["preview_only"] and not result["manifest_created"]
    assert all(value is True for value in result["checks"].values() if isinstance(value, bool))
    different_partition = next(
        ndc
        for ndc, part in result["product_partitions"].items()
        if part != result["product_partitions"]["0-a"]
    )
    cross_cases = [
        {
            "product_ndc_a": a,
            "product_ndc_b": b,
            "case_type": "similar_different",
            "ingredient_overlap": "shared",
        }
        for a, b in (("0-a", "1-a"), ("0-a", different_partition))
    ]
    cross_result = coverage_split_preview(families, cases + cross_cases, constraints)
    assert cross_result["eligible_cross_family_records"] == 1


def test_expansion_proposals_remain_separate_from_original_reviews(tmp_path: Path):
    interim, audit_dir = shortlist_inputs(tmp_path)
    reviewed = pd.DataFrame([family_review("001-08", "001-09", "CEDAR", "CEDAR")])
    baseline = reviewed.copy()
    for field in (
        "review_decision",
        "family_id_a",
        "family_id_b",
        "labels_verified",
        "reviewer_notes",
    ):
        baseline[field] = ""
    baseline.to_csv(audit_dir / "family_review_decisions.csv", index=False)
    reviewed.to_csv(audit_dir / "family_review_decisions_reviewed.csv", index=False)
    pd.DataFrame(columns=["product_ndc", "rxnorm_status", "rxnorm_drug_name"]).to_csv(
        interim / "otc_rxnorm_product_links.csv", index=False
    )
    ref = {
        "product_ndc": "001-12",
        "fda_label_link": {"label_set_id": "saved-set"},
        "live_dailymed_access": "NDC_AND_LABEL_CHECKED",
    }
    pd.DataFrame(
        [
            {
                "record_id": "E01",
                "record_type": "expansion",
                "product_ndc_a": "001-12",
                "product_ndc_b": "001-13",
                "suggested_decision": "SAME_FAMILY",
                "evidence_status": "SUPPORTED_SOURCE_RECORDS",
                "evidence_notes": "Native source records support a proposal only.",
                "proposed_family_id_a": "PALM",
                "proposed_family_id_b": "PALM",
                "use_in_conditional_preview": "TRUE",
                "related_products_json": "[]",
                "co_partition_families_json": "[]",
                "source_evidence_json": json.dumps({"a": ref}),
                "human_approved": "",
                "reviewer_id": "",
                "review_decision": "",
                "family_id_a": "",
                "family_id_b": "",
            }
        ]
    ).to_csv(audit_dir / "expansion_review.csv", index=False)
    protected = {
        path: path.read_bytes()
        for folder in (interim, audit_dir)
        for path in folder.iterdir()
        if path.is_file()
    }
    result = assess_expansion_readiness(tmp_path)
    assert result["readiness"] == "NOT_READY"
    assert result["usable_reviewed_pool"]["products"] == 4
    assert result["usable_reviewed_pool"]["new_ai_suggestions_admitted"] == 0
    assert result["proposed_evaluation_pool"]["products"] == 6
    assert result["expansion_review"]["manual_confirmation_before_inclusion"] == ["E01"]
    assert "001-12" not in result["family_reconciliation"]["assignments"]
    assert not result["training_candidates"]["admitted_to_training"]
    assert not (audit_dir / "family_split_manifest.csv").exists()
    assert protected == {path: path.read_bytes() for path in protected}


def test_fixed_preview_does_not_select_partitions_using_case_categories():
    families = {
        f"{index}-{side}": f"family-{index:02}" for index in range(20) for side in ("a", "b")
    }
    cases = [
        {
            "product_ndc_a": f"{index}-a",
            "product_ndc_b": f"{index}-b",
            "case_type": "similar_same_control",
            "ingredient_overlap": "shared",
        }
        for index in range(20)
    ]
    constraints = [["family-00", "family-01"]]
    preview = coverage_split_preview(families, cases, constraints, balance_coverage=False)
    expected = partition_families(sorted(set(families.values()) - {"family-01"}), seed=42)
    expected["family-01"] = expected["family-00"]
    assert preview["family_partitions"] == expected
    assert not preview["coverage_sufficient"]
    for row in cases:
        row["case_type"] = "similar_different"
        row["ingredient_overlap"] = "disjoint"
    changed = coverage_split_preview(
        dict(reversed(list(families.items()))), cases[::-1], constraints, balance_coverage=False
    )
    assert changed["family_partitions"] == preview["family_partitions"]
    assert changed["product_partitions"] == preview["product_partitions"]
    assert preview["preview_only"] and not preview["manifest_created"]


def test_completed_review_is_distinct_from_personal_signoff():
    baseline = pd.DataFrame(
        [
            {
                "record_id": "E01",
                "product_ndc_a": "a",
                "product_ndc_b": "b",
                "name_a": "Example",
                "evidence_status": "SUPPORTED_SOURCE_RECORDS",
                "use_in_conditional_preview": "TRUE",
                "review_decision": "",
                "labels_verified": "",
                "family_id_a": "",
                "family_id_b": "",
                "human_approved": "",
                "reviewer_id": "",
                "reviewer_notes": "",
            }
        ]
    )
    reviewed = baseline.copy()
    reviewed.loc[0, ["review_decision", "labels_verified", "family_id_a", "family_id_b"]] = [
        "SAME_FAMILY",
        "TRUE",
        "ORIGINAL",
        "ORIGINAL",
    ]
    original = reviewed.copy(deep=True)
    result = confirmed_expansion_reviews(baseline, reviewed, {"a": ["ORIGINAL"]})
    assert len(result) == 1
    assert result.human_approved.eq("").all() and result.reviewer_id.eq("").all()
    pd.testing.assert_frame_equal(original, reviewed)
    assert confirmed_expansion_reviews(baseline, baseline, {}).empty
    for field, value in [("review_decision", "UNCERTAIN"), ("labels_verified", "FALSE")]:
        held = reviewed.copy()
        held.loc[0, field] = value
        assert confirmed_expansion_reviews(baseline, held, {}).empty
    with pytest.raises(ValueError, match="overwrite an original family"):
        confirmed_expansion_reviews(baseline, reviewed, {"a": ["LEGACY"]})
    reviewed.loc[0, "name_a"] = "Changed source name"
    with pytest.raises(ValueError, match="source records changed"):
        confirmed_expansion_reviews(baseline, reviewed, {})


def test_reviewed_reference_keeps_native_units_and_separate_moiety():
    product = {
        "catalog_status": "consistent",
        "agreement_status": "exact_precise_names",
        "rxnorm_ingredient_status": "resolved_path",
        "source_rows": "7",
        "display_name": "Example",
        "dosage_form": "PASTE",
        "route": '["ORAL"]',
        "active_ingredients_json": '[{"name":"SODIUM FLUORIDE","strength":"1.15 mg/g"}]',
        "rxnorm_in_rxcuis_json": '["1"]',
        "rxnorm_in_names_json": '["sodium fluoride"]',
        "rxnorm_pin_rxcuis_json": "[]",
        "rxnorm_pin_names_json": "[]",
    }
    products = {ndc: product.copy() for ndc in ["mouthwash", "paste", "related"]}
    label = {
        "link_status": "matched",
        "label_id": "label",
        "label_set_id": "set",
        "effective_time": "20260101",
        "active_ingredient_json": '["Saved source text"]',
    }
    labels = {ndc: label.copy() for ndc in products}
    drugs = {ndc: {"rxnorm_status": "matched", "rxnorm_rxcui": "1"} for ndc in products}
    mouthwash = {
        "active_ingredients": [{"name": "SODIUM FLUORIDE", "strength": "0.5 mg/mL"}],
        "dosage_form": "LIQUID",
        "route": ["ORAL"],
    }
    paste = {
        "active_ingredients": [
            {"name": "SODIUM FLUORIDE", "strength": "0.25%", "basis": "Drug Facts"}
        ],
        "dosage_form": "PASTE",
        "route": ["ORAL"],
        "separate_active_moiety": {"name": "FLUORIDE ION", "strength": "1.15 mg/g"},
    }
    accepted = pd.DataFrame(
        [
            {
                "record_id": "B-15",
                "product_ndc_a": "mouthwash",
                "product_ndc_b": "paste",
                "proposed_reference_a_json": json.dumps(mouthwash),
                "proposed_reference_b_json": json.dumps(paste),
                "related_products_json": '[{"product_ndc":"related"}]',
            }
        ]
    )
    result = benchmark_references(
        products, labels, drugs, dict.fromkeys(products, "LEGACY"), accepted
    )
    assert result["mouthwash"]["active_ingredients"][0]["strength"] == "0.5 mg/mL"
    for ndc in ["paste", "related"]:
        assert result[ndc]["active_ingredients"] == paste["active_ingredients"]
        assert result[ndc]["separate_active_moiety"] == paste["separate_active_moiety"]
    assert products["paste"]["active_ingredients_json"] == product["active_ingredients_json"]
    labels["related"]["label_set_id"] = "different-label"
    separate = benchmark_references(
        products, labels, drugs, dict.fromkeys(products, "LEGACY"), accepted
    )
    assert "separate_active_moiety" not in separate["related"]


def case_reference(ndc: str, name: str, family: str, ingredient: str, form: str = "CREAM") -> dict:
    return {
        "product_ndc": ndc,
        "name": name,
        "family_id": family,
        "active_ingredients": [{"name": ingredient, "strength": "4 g/100g"}],
        "dosage_form": form,
        "route": ["TOPICAL"],
        "source": {"label_set_id": ndc},
    }


def test_cases_separate_answers_and_keep_all_source_supported_exact_products():
    products = {
        p["product_ndc"]: p
        for p in [
            catalog_product("a", "Ambiguous", "MENTHOL"),
            catalog_product("b", "Ambiguous", "LIDOCAINE"),
            catalog_product("c", "Control"),
            catalog_product("d", "Control Gel", form="GEL"),
        ]
    }
    assignments = {
        n: {
            "partition": "train" if n in {"a", "b"} else "test",
            "family_group_id": "A" if n in {"a", "b"} else "B",
        }
        for n in products
    }
    pairs = [("a", "b"), ("c", "d"), ("a", "c")]
    cases, report = construct_catalog_cases(products, assignments, pairs)
    assert report["construction_exclusions"]["cross_partition_description"] == 1
    assert len(cases.loc[cases.case_type.eq("explicit_ndc")]) == 4
    comparisons = cases.loc[cases.task.eq("ingredient_comparison")]
    assert set(comparisons.partition) == {"train", "test"}
    ambiguous = cases.loc[cases.case_type.eq("name_only_ambiguous")].iloc[0]
    assert json.loads(ambiguous.input_json) == {"name": "ambiguous"}
    assert json.loads(ambiguous.reference_json)["answer_status"] == "INSUFFICIENT_INFORMATION"
    assert all("reference_json" not in value for value in cases.input_json)
    reordered, _ = construct_catalog_cases(
        dict(reversed(list(products.items()))), assignments, pairs[::-1]
    )
    pd.testing.assert_frame_equal(cases, reordered)


def test_shared_rxnorm_base_does_not_equate_different_source_salts():
    products = {
        "a": catalog_product("a", "One", "LIDOCAINE"),
        "b": catalog_product("b", "Two", "LIDOCAINE HYDROCHLORIDE"),
    }
    assignments = {n: {"partition": "train", "family_group_id": "FAMILY"} for n in products}
    cases, _ = construct_catalog_cases(products, assignments, [("a", "b")])
    comparison = cases.loc[cases.task.eq("ingredient_comparison")].iloc[0]
    answer = json.loads(comparison.reference_json)
    assert answer["ingredient_overlap"] == "disjoint"
    assert answer["shared_active_ingredient_identities"] == []


def test_source_admission_does_not_invent_human_review_from_historical_scope():
    products = {
        "a": catalog_product("a", "Pain Reliever", "ACETAMINOPHEN"),
        "b": catalog_product("b", "Pain Reliever", "IBUPROFEN", form="GEL"),
    }
    assignments = {n: {"partition": "train", "family_group_id": "related"} for n in products}
    cases, _ = construct_catalog_cases(products, assignments, [("a", "b")])
    assert len(cases.loc[cases.case_type.eq("explicit_ndc")]) == 2
    assert all(json.loads(value)["human_verified"] is False for value in cases.provenance_json)
