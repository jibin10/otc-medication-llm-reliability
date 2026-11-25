"""Focused parser tests using small temporary files, not research data."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from hrp.data_loading import FdaAudit, _columns, _label_shard_check, find_sources, run


def test_fda_identifier_and_missingness_checks():
    audit = FdaAudit(source="ndc")
    audit.add({"product_ndc": "111-222", "brand_name": "Example", "active_ingredients": []})
    audit.add({"product_ndc": "111-222", "brand_name": "", "active_ingredients": None})

    assert audit.summary()["duplicate_primary_ids"] == 1
    assert audit.records == 2
    coverage = {row["field"]: row for row in audit.field_rows()}
    assert coverage["brand_name"]["missing_or_blank"] == 1


def test_rxnorm_trailing_delimiter():
    assert _columns("123|ENG|\n") == ["123", "ENG"]
    assert _columns("123||\n") == ["123", ""]


def test_shard_validation():
    paths = [Path("drug-label-0001-of-0003.json.zip"), Path("drug-label-0003-of-0003.json.zip")]
    assert _label_shard_check(paths)["missing_shards"] == [2]


def test_inventory_finds_compressed_json_and_rrf(tmp_path):
    (tmp_path / "rxnorm" / "rrf").mkdir(parents=True)
    (tmp_path / "rxnorm" / "rrf" / "RXNCONSO.RRF").write_text("", encoding="utf-8")
    for name in ("drug-ndc-0001-of-0001.json.zip", "drug-label-0001-of-0001.json.zip"):
        with ZipFile(tmp_path / name, "w") as archive:
            archive.writestr(name.removesuffix(".zip"), json.dumps({"results": [{}]}))

    found = find_sources(tmp_path)
    assert len(found.ndc) == 1
    assert len(found.labels) == 1
    assert len(found.rxnorm) == 1

    report_dir = tmp_path / "out"
    run(tmp_path, report_dir, inventory_only=True)
    assert (report_dir / "source_inventory.csv").exists()


def test_rrf_columns_and_ndc_attribute_count(tmp_path):
    from hrp.data_loading import audit_rrf

    conso = tmp_path / "RXNCONSO.RRF"
    conso_row = [""] * 18
    conso_row[0], conso_row[7], conso_row[11], conso_row[12] = ("123", "ATOM1", "RXNORM", "IN")
    conso.write_text("|".join(conso_row) + "|\n", encoding="utf-8")
    conso_summary = audit_rrf(conso)
    assert conso_summary["rows"] == 1
    assert conso_summary["wrong_column_count"] == 0

    sat = tmp_path / "RXNSAT.RRF"
    sat_row = [""] * 13
    sat_row[0], sat_row[8], sat_row[10] = "123", "NDC", "12345-6789-01"
    sat.write_text("|".join(sat_row) + "|\n", encoding="utf-8")
    sat_summary = audit_rrf(sat)
    assert sat_summary["ndc_attributes"] == 1
    assert sat_summary["wrong_column_count"] == 0


def test_full_audit_writes_only_summaries(tmp_path, monkeypatch):
    from hrp import data_loading

    for name in ("drug-ndc-0001-of-0001.json.zip", "drug-label-0001-of-0001.json.zip"):
        with ZipFile(tmp_path / name, "w") as archive:
            archive.writestr(name.removesuffix(".zip"), json.dumps({"results": [{}]}))

    conso = tmp_path / "RXNCONSO.RRF"
    conso_row = [""] * 18
    conso_row[0] = "123"
    conso.write_text("|".join(conso_row) + "|\n", encoding="utf-8")

    sat = tmp_path / "RXNSAT.RRF"
    sat_row = [""] * 13
    sat_row[0] = "123"
    sat.write_text("|".join(sat_row) + "|\n", encoding="utf-8")

    def fake_records(path):
        if "drug-ndc" in path.name:
            yield {"product_ndc": "123-45", "active_ingredients": []}
        else:
            yield {"id": "label-one", "set_id": "label-set-one"}

    monkeypatch.setattr(data_loading, "_iter_fda_records", fake_records)
    report_dir = tmp_path / "reports"
    summary = data_loading.run(tmp_path, report_dir)

    assert summary["ndc"]["records"] == 1
    assert summary["label"]["records"] == 1
    assert (report_dir / "source_audit_summary.json").exists()
    assert (report_dir / "fda_field_coverage.csv").exists()
    assert (report_dir / "rxnorm_file_summary.csv").exists()


def test_discovery_selects_one_full_release_and_never_combines_prescribe(tmp_path):
    for directory in ("release/rrf", "release/prescribe/rrf"):
        folder = tmp_path / directory
        folder.mkdir(parents=True)
        for name in ("RXNCONSO.RRF", "RXNSAT.RRF", "RXNREL.RRF"):
            (folder / name).write_text("", encoding="utf-8")
    found = find_sources(tmp_path)
    assert len(found.rxnorm) == 3
    assert all("prescribe" not in p.parts for p in found.rxnorm)
    (tmp_path / "second/rrf").mkdir(parents=True)
    (tmp_path / "second/rrf/RXNCONSO.RRF").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple full"):
        find_sources(tmp_path)


def test_prescribe_alone_is_not_a_full_release(tmp_path):
    from hrp.data_loading import verify_sources

    folder = tmp_path / "prescribe/rrf"
    folder.mkdir(parents=True)
    (folder / "RXNCONSO.RRF").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete full"):
        verify_sources(tmp_path, tmp_path / "out")


def test_cross_type_listings_keep_classification_separate_from_identity(monkeypatch):
    from hrp import data_loading

    records = [
        {
            "product_ndc": "11111-1",
            "product_type": kind,
            "dosage_form": "LIQUID",
            "route": route,
            "active_ingredients": [{"name": "Example", "strength": strength}],
        }
        for kind, route, strength in (
            ("HUMAN OTC DRUG", ["TOPICAL"], "0.3 g/30mL"),
            ("HUMAN PRESCRIPTION DRUG", ["TOPICAL"], "10 mg/mL"),
            ("BULK INGREDIENT", None, "0.1 g/10mL"),
        )
    ]
    monkeypatch.setattr(data_loading, "_iter_fda_records", lambda path: iter(records))
    result = data_loading.audit_ndc_product_types([Path("source")])
    assert result["identity_conflicts"] == 0
    assert result["relationships"][0]["classification_outcome"].startswith("unresolved")
    records[1]["active_ingredients"][0]["strength"] = "20 mg/mL"
    assert data_loading.audit_ndc_product_types([Path("source")])["identity_conflicts"] == 1
