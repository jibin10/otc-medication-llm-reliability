"""Cold-kernel restoration checks; never run challenge inference or scoring."""

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/06_XGBoost_OTC_Identification.ipynb"
RUN = ROOT / "results/otc_models/xgboost_20261121T102526Z_4b76aa52"


def cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}


def enable_source(*, confirm):
    source = cells()["final-eval-only-enable"]
    source = source.replace("ENABLE_FINAL_COMPARISON = False", "ENABLE_FINAL_COMPARISON = True")
    return source.replace(
        "CONFIRM_MODEL_SETTINGS_FROZEN = False",
        f"CONFIRM_MODEL_SETTINGS_FROZEN = {confirm}",
    )


def test_optional_cells_preserve_normal_training_state():
    source = cells()
    sentinel = object()
    namespace = {"RUN_ROOT": sentinel, "RUN_FINAL_CHALLENGE_EVAL": True,
                 "MODEL_SETTINGS_FROZEN": True}
    for cell_id in ("final-eval-only-settings", "final-eval-only-restore", "final-eval-only-enable"):
        exec(source[cell_id], namespace)
    assert namespace["RUN_ROOT"] is sentinel
    assert namespace["RUN_FINAL_CHALLENGE_EVAL"] is True
    assert namespace["MODEL_SETTINGS_FROZEN"] is True
    assert namespace["FINAL_EVALUATION_ONLY"] is False
    assert namespace["ENABLE_FINAL_COMPARISON"] is False


@pytest.mark.parametrize(
    ("restored", "confirm", "error"),
    [(False, True, RuntimeError), (True, False, RuntimeError), (True, True, FileNotFoundError)],
)
def test_enablement_fails_closed(restored, confirm, error):
    namespace = {"FINAL_EVALUATION_ONLY": True, "FINAL_EVAL_RESTORED": restored,
                 "RUN_FINAL_CHALLENGE_EVAL": True, "MODEL_SETTINGS_FROZEN": True,
                 "SET_PREDICTIONS": None, "Path": Path}
    with pytest.raises(error):
        exec(enable_source(confirm=confirm), namespace)
    assert namespace["RUN_FINAL_CHALLENGE_EVAL"] is False
    assert namespace["MODEL_SETTINGS_FROZEN"] is False


def test_enablement_requires_all_three_returned_artifacts(tmp_path):
    paths = {key: tmp_path / key for key in
             ("SET_PREDICTIONS", "SET_RAW_DECISIONS", "SET_FREEZE_FILE")}
    namespace = {"FINAL_EVALUATION_ONLY": True, "FINAL_EVAL_RESTORED": True,
                 "Path": Path, **paths}
    for key, path in paths.items():
        with pytest.raises(FileNotFoundError, match=key):
            exec(enable_source(confirm=True), namespace)
        assert namespace["RUN_FINAL_CHALLENGE_EVAL"] is False
        path.touch()
    exec(enable_source(confirm=True), namespace)
    assert namespace["RUN_FINAL_CHALLENGE_EVAL"] is True
    assert namespace["MODEL_SETTINGS_FROZEN"] is True


def test_notebook_schema_and_all_code_cells_compile():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
    for cell_id, source in cells().items():
        if cell_id in {c["id"] for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
                       if c["cell_type"] == "code"}:
            compile(source, cell_id, "exec")
    for cell_id in ("review-095", "review-097", "review-099"):
        assert ast.unparse(ast.parse(cells()[cell_id]).body[0].test) == "RUN_FINAL_CHALLENGE_EVAL"


def test_cold_kernel_restores_frozen_run_without_training_or_challenge_access():
    if not (RUN / "model/frozen.json").is_file():
        pytest.skip("The completed local XGBoost run is required for this bounded regression")
    for package in ("xgboost", "joblib", "numpy", "pandas", "sklearn", "nbformat"):
        pytest.importorskip(package)
    script = r'''
import builtins, gzip, hashlib, itertools, json, sys
from pathlib import Path

root = Path.cwd().parent
run = root / "results/otc_models/xgboost_20261121T102526Z_4b76aa52"
notebook = json.loads((root / "notebooks/06_XGBoost_OTC_Identification.ipynb").read_text())
sources = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

before = {p.relative_to(run).as_posix(): digest(p) for p in run.rglob("*") if p.is_file()}

def forbidden(*args, **kwargs):
    raise AssertionError("Training, source preparation or scoring was called")

import xgboost as xgb
xgb.XGBClassifier.fit = forbidden
xgb.train = forbidden
sys.path.insert(0, str(root / "src"))
from hrp import challenge_evaluation, challenge_scoring_v3
challenge_evaluation.score_predictions = forbidden
challenge_evaluation.predict_common_evidence = forbidden
challenge_scoring_v3.score_predictions = forbidden

real_path_open, real_gzip_open = Path.open, gzip.open
def guarded_open(path, mode="r", *args, **kwargs):
    if path.name in {"benchmark.PRIVATE.jsonl", "common_evidence.PRIVATE.jsonl",
                     "deterministic_predictions.PRIVATE.jsonl", "validated_outputs.PRIVATE.jsonl"}:
        assert "b" in mode, "Challenge contents must not be parsed during restoration"
    return real_path_open(path, mode, *args, **kwargs)
def guarded_gzip_open(path, *args, **kwargs):
    assert Path(path).name == "ingredient_identities.jsonl.gz", "Dataset preparation was attempted"
    return real_gzip_open(path, *args, **kwargs)
Path.open, gzip.open = guarded_open, guarded_gzip_open

namespace = {"__name__": "__main__"}
exec(sources["review-004"], namespace)
exec(sources["final-eval-only-settings"], namespace)
namespace["FINAL_EVALUATION_ONLY"] = True
exec(sources["final-eval-only-restore"], namespace)
assert namespace["FINAL_EVAL_RESTORED"] is True
assert namespace["RUN_FINAL_CHALLENGE_EVAL"] is False
assert namespace["MODEL_SETTINGS_FROZEN"] is False
assert not hasattr(namespace["TreeHead"], "fit")
assert not hasattr(namespace["TreeHead"], "save")
assert not {"fit_heads", "prepare_all", "train_rows", "matrices", "cal_decisions"} & namespace.keys()
assert set(namespace["loaded_heads"]) == {"candidate", "sufficiency", "status", "form"}
assert namespace["thresholds"] == {"sufficiency": 0.6, "form": 0.75, "candidate": 0.6}
assert len(namespace["challenge_ids"]) == 230
assert "\n\n".join(namespace["inspect"].getsource(f) for f in namespace["FEATURE_FUNCTIONS"]) == (
    run / "model/feature_source.py").read_text()
exec(sources["final-eval-only-enable"], namespace)
for cell_id in ("review-093", "review-095", "review-097", "review-099"):
    exec(sources[cell_id], namespace)
assert not {"source_cases", "final_decisions", "final_reports", "final_folder"} & namespace.keys()

# Restore only eight development cases; never score or access challenge inputs.
Path.open, gzip.open = real_path_open, real_gzip_open
with gzip.open(run / "prepared_data/calibration.jsonl.gz", "rt", encoding="utf-8") as stream:
    rows = [json.loads(line) for line in itertools.islice(stream, 8)]
items, _ = namespace["flatten"](rows)
request, candidate = namespace["feature_rows"](items, namespace["resolver"])
encoders, np = namespace["saved_encoders"], namespace["np"]
decisions = namespace["predict_heads"](
    namespace["loaded_heads"], items,
    encoders["request"].transform(request).astype(np.float32),
    encoders["candidate"].transform(candidate).astype(np.float32))
evaluation = run / "evaluations/development_d535889f"
with gzip.open(evaluation / "calibration_probabilities.PRIVATE.jsonl.gz", "rt", encoding="utf-8") as stream:
    recorded = [json.loads(line)["probabilities"] for line in itertools.islice(stream, len(items))]
assert decisions == recorded
with gzip.open(evaluation / "calibration_predictions.jsonl.gz", "rt", encoding="utf-8") as stream:
    predictions = [json.loads(line) for line in itertools.islice(stream, len(rows))]
cursor = 0
for row, expected in zip(rows, predictions):
    count = len(row["payload"]["products"])
    output, _ = namespace["adapt_case"](
        row["payload"], decisions[cursor:cursor + count], namespace["resolver"], namespace["thresholds"])
    assert row["case_id"] == expected["case_id"]
    assert output == expected["output"]
    cursor += count
assert {p.relative_to(run).as_posix(): digest(p) for p in run.rglob("*") if p.is_file()} == before
print("Cold restoration and eight development prediction replays passed; all saved files unchanged.")
'''
    result = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=ROOT / "notebooks",
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
