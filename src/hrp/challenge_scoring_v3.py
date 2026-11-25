"""Scoring-only amendment for the frozen challenge_v1_contract_2 benchmark.

Stage 6 callers should import score_predictions from this module. Evidence,
references, parser, identity mapping and all non-joint metrics retain contract 2.
The historical module remains byte-for-byte unchanged for reproducibility.
"""

from hrp import challenge_evaluation as frozen

SCORING_CONTRACT_VERSION = "challenge_v1_scoring_contract_3"
BENCHMARK_CONTRACT_VERSION = "challenge_v1_contract_2"


def score_predictions(cases, predictions, payloads, resolver):
    """Require all applicable formulations for joint success; preserve other metrics.

    The frozen scorer's per-case correct_full_formulation count already uses
    _form_key(). Reusing that count preserves its exact normalization and error
    handling without duplicating the scorer or patching its module globals.
    A case loses at most one joint credit, even if both product formulations fail.
    """
    if frozen.VERSION != BENCHMARK_CONTRACT_VERSION:
        raise RuntimeError("This scoring amendment requires challenge_v1_contract_2")
    result = frozen.score_predictions(cases, predictions, payloads, resolver)
    for row in cases:
        if not any(target["formulation"] is not None for target in row["reference"]["products"]):
            continue
        case_id = row["case_id"]
        single_predictions = {case_id: predictions[case_id]} if case_id in predictions else {}
        single = frozen.score_predictions([row], single_predictions, payloads, resolver)
        counts = single["overall"]["counts"]
        if (
            counts["joint_correct_cases"]
            and counts["correct_full_formulation"] < counts["full_formulation_target_requests"]
        ):
            for summary in (
                result["overall"],
                result["by_role"][row["scoring_role"]],
                result["by_category"][row.get("category", "unclassified")],
            ):
                summary["counts"]["joint_correct_cases"] -= 1
                joint = summary["joint_case_accuracy"]
                joint["numerator"] -= 1
                joint["value"] = joint["numerator"] / joint["denominator"]
    result["scoring_contract_version"] = SCORING_CONTRACT_VERSION
    result["benchmark_contract_version"] = BENCHMARK_CONTRACT_VERSION
    return result
