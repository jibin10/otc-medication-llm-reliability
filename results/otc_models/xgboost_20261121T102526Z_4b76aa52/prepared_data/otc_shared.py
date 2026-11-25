import copy
import csv
import gzip
import hashlib
import json
import os
import uuid
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from hrp import challenge_evaluation as contract
from hrp.challenge_scoring_v3 import score_predictions
from hrp.data_cleaning import normalize_name, quantity_key, route_signature


STATUS = [
    "EXACT_NDC_IDENTIFIED", "FORMULATION_IDENTIFIED",
    "INGREDIENT_SET_IDENTIFIED", "INSUFFICIENT_INFORMATION",
    "NO_MATCH", "SOURCE_CONFLICT"
]
SUPPORTED = set(STATUS[:3])
QUERY_FIELDS = {"name", "dosage_form", "route", "native_strengths", "product_ndc"}
CANDIDATE_FIELDS = {
    "product_ndc", "display_name", "brand_name", "generic_name", "dosage_form",
    "route", "fda_active_ingredients", "spl", "rxnorm", "provenance"
}
BLOCKED = contract.FORBIDDEN | {
    "target", "label", "labels", "expected_status", "correctness", "families",
    "family", "family_group_id", "review_decision", "reference_answer",
    "candidate_labels", "source_assessment", "support_tier", "ingredient_eligible",
    "formulation_eligible", "comparison_eligible", "ingredient_identities",
    "status", "answer_status", "resolution_level", "expected_identification_status",
    "gold_comparison", "review_decisions", "gemma_prediction", "correctness_flag"
}

def dump_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def sha_value(value):
    return hashlib.sha256(contract.canonical(value).encode("utf-8")).hexdigest()

def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    # Fixed gzip time keeps equal bundles byte reproducible
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            for row in rows:
                stream.write((contract.canonical(row) + "\n").encode("utf-8"))


def check_source(value):
    contract.assert_source_only(value)
    if isinstance(value, dict):
        found = BLOCKED.intersection(value)
        if found:
            raise ValueError("Label or split fields reached model input " + str(sorted(found)))
        for child in value.values():
            check_source(child)
    elif isinstance(value, list):
        for child in value:
            check_source(child)

def check_payload(payload):
    check_source(payload)
    if set(payload) != {"contract_version", "source_scope", "products"}:
        raise ValueError("Unexpected evidence fields")
    if payload["contract_version"] != contract.VERSION or len(payload["products"]) not in (1, 2):
        raise ValueError("Unexpected evidence contract")
    for item in payload["products"]:
        if set(item) != {"request", "candidate_count", "candidates"}:
            raise ValueError("Unexpected request fields")
        if not item["request"] or set(item["request"]) - QUERY_FIELDS:
            raise ValueError("Unexpected query fields")
        candidates = item["candidates"]
        if item["candidate_count"] != len(candidates):
            raise ValueError("Candidate truncation is not allowed")
        if len({c["product_ndc"] for c in candidates}) != len(candidates):
            raise ValueError("Repeated source NDC")
        for candidate in candidates:
            if set(candidate) != CANDIDATE_FIELDS:
                raise ValueError("Unexpected candidate fields")
            if set(candidate["spl"]) != {
                "label_id", "set_id", "version", "effective_time", "active_ingredient_text",
                "structured_substance_names"
            }:
                raise ValueError("Unexpected SPL source fields")
            if set(candidate["rxnorm"]) != {
                "product_rxcui", "product_name", "ingredient_pairs", "cached_pairs_confirmed"
            }:
                raise ValueError("Unexpected RxNorm source fields")
            for ingredient in candidate["fda_active_ingredients"]:
                if set(ingredient) != {"name", "strength"}:
                    raise ValueError("Unexpected native ingredient fields")
            for pair in candidate["rxnorm"]["ingredient_pairs"]:
                if set(pair) != {"name", "rxcui", "term_type", "uniis"}:
                    raise ValueError("Unexpected source identity fields")


def ingredient_key(candidate, resolver):
    names = [i["name"] for i in candidate["fda_active_ingredients"]]
    values = [resolver.identify(name) for name in names]
    return tuple(sorted(set(values))) if names and all(values) else None

def form_key(candidate, resolver):
    form = {
        "active_ingredients": candidate["fda_active_ingredients"],
        "dosage_form": candidate["dosage_form"],
        "route": candidate["route"]
    }
    return contract._form_key(form, resolver)

def copy_form(candidate):
    return {
        "active_ingredients": copy.deepcopy(candidate["fda_active_ingredients"]),
        "dosage_form": candidate["dosage_form"],
        "route": list(candidate["route"])
    }

def compare_products(products, resolver):
    if len(products) == 1:
        return None
    out = {
        "same_active_ingredient_set": None, "ingredient_overlap": None,
        "shared_active_ingredient_names": []
    }
    if all(p["status"] in SUPPORTED for p in products):
        a, b = [resolver.map_output(p["active_ingredient_names"])[0] for p in products]
        shared = a & b
        out.update(
            same_active_ingredient_set=a == b,
            ingredient_overlap="shared" if shared else "disjoint",
            shared_active_ingredient_names=sorted({
                name for name in products[0]["active_ingredient_names"]
                if resolver.identify(name) in shared
            })
        )
    return out


def adapt_product(item, decision, resolver, thresholds, guard=True):
    candidates = item["candidates"]
    scores = np.asarray(decision["candidate"], dtype=float)
    status_scores = np.asarray(decision["status"], dtype=float)
    if scores.shape != (len(candidates),) or status_scores.shape != (len(STATUS),):
        raise ValueError("Model scores have the wrong shape")
    if not np.isfinite(scores).all() or not np.isfinite(status_scores).all():
        raise ValueError("Model scores must be finite")
    raw_status = STATUS[int(status_scores.argmax())]
    status = raw_status
    selected = candidates[int(scores.argmax())] if len(scores) else None
    raw_form = bool(selected and raw_status in SUPPORTED and decision["form"] >= thresholds["form"])
    form, status_reasons, form_reasons = None, [], []
    if guard and status in SUPPORTED:
        if decision["sufficiency"] < thresholds["sufficiency"]:
            status_reasons.append("learned sufficiency below threshold")
        if not len(scores) or float(scores.max()) < thresholds["candidate"]:
            status_reasons.append("learned source support below threshold")
        keys = [ingredient_key(candidate, resolver) for candidate in candidates]
        if not keys or None in keys or len(set(keys)) != 1:
            status_reasons.append("source ingredients cannot support one set")
        if status == STATUS[0] and len(candidates) != 1:
            status_reasons.append("exact NDC requires one source candidate")
    if raw_form:
        keys = [form_key(candidate, resolver) for candidate in candidates]
        strengths = [quantity_key(item.get("strength", ""))
                     for candidate in candidates for item in candidate["fda_active_ingredients"]]
        can_copy = (
            bool(keys) and None not in keys and len(set(keys)) == 1 and bool(strengths)
            and all(key and key[1] != "percent_unspecified" for key in strengths)
        )
        if can_copy or not guard:
            form = copy_form(selected)
        else:
            form_reasons.append("formulation withheld because source fields do not agree")
    if guard and status == STATUS[1] and form is None:
        status_reasons.append("learned formulation claim cannot be copied")
    if guard and status == "NO_MATCH" and candidates:
        status_reasons.append("nonempty source evidence")
    if status_reasons:
        status, form = "INSUFFICIENT_INFORMATION", None
    if guard and not candidates:
        status, form = "NO_MATCH", None
        status_reasons.append("empty source candidate set")
    names = sorted({item["name"] for item in selected["fda_active_ingredients"]}) if selected else []
    product = {
        "status": status,
        "product_ndc": selected["product_ndc"] if selected and status == STATUS[0] else None,
        "active_ingredient_names": names if status in SUPPORTED else [],
        "source_ndcs": [candidate["product_ndc"] for candidate in candidates],
        "formulation": form if status in SUPPORTED else None
    }
    change = {
        "raw_status": raw_status, "final_status": status,
        "raw_form_asserted": raw_form, "final_form_asserted": product["formulation"] is not None,
        "status_changed": raw_status != status,
        "form_changed": raw_form != (product["formulation"] is not None),
        "structural_no_match": bool(guard and not candidates),
        "reasons": status_reasons + form_reasons
    }
    return product, change


def adapt_case(payload, decisions, resolver, thresholds, guard=True):
    check_payload(payload)
    if len(decisions) != len(payload["products"]):
        raise ValueError("Missing model decisions")
    results = [adapt_product(item, decision, resolver, thresholds, guard)
               for item, decision in zip(payload["products"], decisions)]
    products = [product for product, _ in results]
    output = {"products": products, "comparison": compare_products(products, resolver)}
    if guard:
        contract.parse_output(output, payload)
    return output, [change for _, change in results]

def flatten(rows):
    items, labels = [], []
    for row in rows:
        check_payload(row["payload"])
        for index, item in enumerate(row["payload"]["products"]):
            items.append(item)
            labels.append({
                "status": STATUS.index(row["target"]["products"][index]["status"]),
                "sufficiency": int(row["target"]["products"][index]["status"] in SUPPORTED),
                "form": int(row["target"]["products"][index]["formulation"] is not None),
                "candidate": row["candidate_labels"][index]
            })
    return items, labels


def metric(numerator, denominator):
    return {
        "numerator": int(numerator), "denominator": int(denominator),
        "value": float(numerator / denominator) if denominator else None
    }

def binary_metrics(truth, pred):
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    return {
        "precision": metric(tp, tp + fp), "recall": metric(tp, tp + fn),
        "f1": metric(2 * tp, 2 * tp + fp + fn)
    }

def choose_threshold(truth, scores):
    if len(set(truth)) < 2:
        return 0.5
    grid = [0.25, 0.4, 0.5, 0.6, 0.75]
    values = [binary_metrics(truth, np.asarray(scores) >= t)["f1"]["value"] or 0 for t in grid]
    return grid[max(range(len(grid)), key=lambda i: (values[i], -abs(grid[i] - 0.5)))]

def calibrate(labels, decisions):
    # Calibration uses only its reserved development families
    out = {}
    for key in ("sufficiency", "form"):
        out[key] = choose_threshold([y[key] for y in labels], [d[key] for d in decisions])
    ys, ps = [], []
    for label, decision in zip(labels, decisions):
        for y, p in zip(label["candidate"], decision["candidate"]):
            if y >= 0:
                ys.append(y)
                ps.append(p)
    out["candidate"] = choose_threshold(ys, ps)
    return out


def raw_metrics(labels, decisions, thresholds):
    truth = np.array([y["status"] for y in labels])
    pred = np.array([int(np.argmax(d["status"])) for d in decisions])
    good = np.isin(truth, [0, 1, 2])
    chosen = np.isin(pred, [0, 1, 2])
    counts = np.zeros((6, 6), dtype=int)
    for a, b in zip(truth, pred):
        counts[a, b] += 1
    eligible, selected, nontrivial = 0, 0, 0
    cy, cp = [], []
    for y, d in zip(labels, decisions):
        ys = y["candidate"]
        if any(v == 1 for v in ys):
            eligible += 1
            selected += int(ys[int(np.argmax(d["candidate"]))] == 1)
            nontrivial += int(any(v == 0 for v in ys))
        for a, b in zip(ys, d["candidate"]):
            if a >= 0:
                cy.append(a)
                cp.append(b)
    return {
        "status_accuracy": metric((truth == pred).sum(), len(truth)),
        "status_confusion": counts.tolist(),
        "top_record_source_support": metric(selected, eligible),
        "mixed_source_support_requests": nontrivial,
        "source_support": binary_metrics(cy, np.asarray(cp) >= thresholds["candidate"]),
        "source_support_target": "source assessment support rather than query intent",
        "sufficiency": binary_metrics([y["sufficiency"] for y in labels],
                                    [d["sufficiency"] >= thresholds["sufficiency"] for d in decisions]),
        "form": binary_metrics([y["form"] for y in labels],
                              [d["form"] >= thresholds["form"] for d in decisions]),
        "abstention": binary_metrics(~good, ~chosen),
        "false_identification": metric((chosen & ~good).sum(), (~good).sum()),
        "false_abstention": metric((~chosen & good).sum(), good.sum())
    }


def adapter_metrics(diagnostics):
    changes = [change for row in diagnostics for change in row["changes"]]
    status_changes = sum(change["status_changed"] for change in changes)
    return {
        "adapter_changed_requests": status_changes,
        "adapter_status_changes": status_changes,
        "adapter_form_changes": sum(change["form_changed"] for change in changes),
        "adapter_intervened_requests": sum(
            change["status_changed"] or change["form_changed"] for change in changes
        ),
        "adapter_guard_requests": sum(bool(change["reasons"]) for change in changes),
        "structural_no_match_requests": sum(change["structural_no_match"] for change in changes),
        "adapter_reason_counts": dict(Counter(reason for change in changes for reason in change["reasons"]))
    }


def evaluate(rows, decisions, resolver, thresholds):
    predictions, raw_predictions, payloads, cases, diagnostics = {}, {}, {}, [], []
    raw_valid = 0
    cursor = 0
    for row in rows:
        size = len(row["payload"]["products"])
        local = decisions[cursor:cursor + size]
        cursor += size
        cid = row["case_id"]
        pred, changes = adapt_case(row["payload"], local, resolver, thresholds)
        raw_pred, _ = adapt_case(row["payload"], local, resolver, thresholds, guard=False)
        try:
            contract.parse_output(raw_pred, row["payload"])
            raw_valid += 1
        except (ValueError, TypeError, KeyError):
            pass
        predictions[cid], raw_predictions[cid], payloads[cid] = pred, raw_pred, row["payload"]
        # The frozen scoring function expects the validation partition name
        cases.append({
            "case_id": cid, "partition": "validation", "reference": row["target"],
            "scoring_role": row["meta"]["task"], "category": row["meta"]["category"]
        })
        diagnostics.append({"case_id": cid, "changes": changes})
    if cursor != len(decisions):
        raise ValueError("Extra model decisions")
    scores = score_predictions(cases, predictions, payloads, resolver)
    _, labels = flatten(rows)
    scores["raw"] = raw_metrics(labels, decisions, thresholds)
    scores["raw_contract_validity"] = metric(raw_valid, len(rows))
    scores.update(adapter_metrics(diagnostics))
    scores["raw_pipeline"] = score_predictions(cases, raw_predictions, payloads, resolver)
    final_truth, final_pred = [], []
    for row in rows:
        final_truth.extend(p["status"] not in SUPPORTED for p in row["target"]["products"])
        final_pred.extend(p["status"] not in SUPPORTED for p in predictions[row["case_id"]]["products"])
    scores["final_abstention"] = binary_metrics(final_truth, final_pred)
    a, b = np.asarray(final_truth), np.asarray(final_pred)
    scores["final_false_identification"] = metric((a & ~b).sum(), a.sum())
    scores["final_false_abstention"] = metric((~a & b).sum(), (~a).sum())
    return scores, predictions, diagnostics


def plot_confusion(matrix, path, title):
    short = ["NDC", "Form", "Ingredients", "Insufficient", "No match", "Conflict"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(np.asarray(matrix), display_labels=short).plot(
        ax=ax, colorbar=False, cmap="Blues", xticks_rotation=35
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()


def status_statistics(truth, prediction):
    counts = np.zeros((len(STATUS), len(STATUS)), dtype=int)
    for actual, predicted in zip(truth, prediction):
        counts[int(actual), int(predicted)] += 1
    classes = []
    for i, name in enumerate(STATUS):
        tp, support, selected = int(counts[i, i]), int(counts[i].sum()), int(counts[:, i].sum())
        classes.append({"label": name, "support": support,
                        "precision": tp / selected if selected else 0.0,
                        "recall": tp / support if support else 0.0,
                        "f1": 2 * tp / (support + selected) if support + selected else 0.0})
    return {"labels": STATUS, "counts": counts.tolist(), "per_class": classes,
            "accuracy": metric(np.trace(counts), counts.sum()),
            "orientation": "rows=true, columns=predicted", "zero_division": 0}

def save_development_artifacts(root, part, rows, decisions, predictions, report, thresholds, seconds):
    # Private request IDs, labels and probabilities; no copied source records.
    root = Path(root)
    _, labels = flatten(rows)
    truth = [y["status"] for y in labels]
    raw = [int(np.argmax(d["status"])) for d in decisions]
    adapted = [STATUS.index(p["status"]) for row in rows
               for p in predictions[row["case_id"]]["products"]]
    status = {"raw": status_statistics(truth, raw),
              "adapted": status_statistics(truth, adapted)}
    dump_json(root / (part + "_status_counts.json"), status)
    with (root / (part + "_per_class.csv")).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["pipeline", "label", "support", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows({"pipeline": key, **row} for key, info in status.items() for row in info["per_class"])
    truth_sufficient = np.array([y["sufficiency"] for y in labels], bool)
    raw_sufficient = np.array([d["sufficiency"] >= thresholds["sufficiency"] for d in decisions])
    adapted_sufficient = np.isin(adapted, [0, 1, 2])
    dump_json(root / (part + "_evaluation_summary.json"), {
        "thresholds": thresholds, "status": status,
        "evidence_sufficiency_accuracy": {
            "raw": metric((truth_sufficient == raw_sufficient).sum(), len(labels)),
            "adapted": metric((truth_sufficient == adapted_sufficient).sum(), len(labels))},
        "prediction_seconds": seconds, "requests": len(decisions),
        "requests_per_second": len(decisions) / seconds if seconds > 0 else None,
        "metrics_file": part + "_metrics.json"
    })
    def probability_rows():
        cursor = 0
        for row in rows:
            for product_index in range(len(row["payload"]["products"])):
                yield {"case_id": row["case_id"], "product_index": product_index,
                       "labels": labels[cursor], "probabilities": decisions[cursor]}
                cursor += 1
    write_jsonl(root / (part + "_probabilities.PRIVATE.jsonl.gz"), probability_rows())

def regenerate_status_figures(root, part, title):
    # Requires only saved counts, Matplotlib and scikit-learn; no model or data.
    root = Path(root)
    saved = json.loads((root / (part + "_status_counts.json")).read_text(encoding="utf-8"))
    for pipeline, info in saved.items():
        if info["labels"] != STATUS:
            raise ValueError("Unexpected saved status labels")
        suffix = "_status.png" if pipeline == "raw" else "_adapted_status.png"
        plot_confusion(info["counts"], root / (part + suffix), title + " (" + pipeline + ")")
