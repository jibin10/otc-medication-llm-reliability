# Stage 5C challenge_v1 source review and finalization

Decision: **GO**. Benchmark status: **FROZEN**.
This is a validation-development overlay, source-reviewed and not independently clinically verified.
No additional human review is requested or required by this implementation.

## Accepted review and immutable boundaries

The authoritative input is `validation_review_queue.REVIEWED.csv`, with exactly
17 PRIMARY_CANDIDATE, 14 CONTROL_CANDIDATE, 4 AMBIGUITY_CANDIDATE, 14 SUPPLEMENTARY,
13 EXCLUDE_PRIMARY and 38 HOLD_SOURCE_OR_IDENTITY verdicts. All 100 REVIEWED markers and both
verdict fields were checked. The accepted input is preserved byte-for-byte in
`accepted_review_snapshot.PRIVATE.csv`; both original reviewed and pending CSVs remain unchanged.
No verdict was invented or overwritten. The per-case decision ledger records the original verdict
separately from automated eligibility, reasons and membership. Reconciliation:
`{"AMBIGUITY_CANDIDATE":{"selected_ambiguity":2,"unresolved":2},"CONTROL_CANDIDATE":{"selected_same_ingredient_control":11,"unresolved":3},"EXCLUDE_PRIMARY":{"excluded":13},"HOLD_SOURCE_OR_IDENTITY":{"eligible_not_selected":1,"selected_different_with_ndc":2,"unresolved":35},"PRIMARY_CANDIDATE":{"selected_different_with_ndc":10,"selected_different_without_ndc":2,"unresolved":5},"SUPPLEMENTARY":{"selected_supplementary":8,"unresolved":6}}`.

Only existing train/validation cases and original public-source FDA/RxNorm records were used.
The sealed test was never opened, decoded, hashed or evaluated. No final_v5 files, historical
baseline results, source downloads or original review files were modified. No Gemma inference,
training, Runpod operation, source download or new clinical verification was performed.

## Automated source review

Original source paths and actual SHA-256 hashes are in `evidence_manifest.json` and the freeze
manifest. The exact SPL ID, set ID, version, effective date and active-ingredient text are
matched to the NDC SPL ID and cached source. Raw NDC product names and ingredient records are
cross-checked. FDA Drug Labeling/openFDA and DailyMed may display the same manufacturer SPL;
the saved DailyMed set links are provenance links, not independently fetched clinical evidence.
The local exact-version SPL is authoritative for version matching; no newer webpage is substituted.

The original RxNorm RXNCONSO and RXNREL release is read and hashed. Cached IN/PIN name/RxCUI
pairs and ingredient paths are checked against it. Names and identifiers stay paired. UNII
links come from original MTHSPL SU atoms and the source-only singleton FDA crosswalk, never
from validation references. Same-concept RxNorm aliases are allowed only when the UNII link
is unique. Salts, PIN and IN are not collapsed. Ambiguous, absent or conflicting mappings
remain unresolved. Percentages retain explicit or unspecified basis; no density conversion,
active-moiety equivalence or unsupported strength denominator is inferred.

Structured and amount-anchored active-text ingredient identities are compared. Missing names,
unrecognized text, uncovered quantities or unsupported mappings are unresolved, not automatic
proof of an incorrect label. Strength checks distinguish matches, comparable-basis differences
and unresolved basis. Ingredient-set support is distinct from full formulation support.
USP/NF grades, explicitly stated teaspoon/tablepoon mL denominators and fluoride-ion basis
annotations are recognized as source notation. No spoon volume is assumed, no fluoride-ion
strength is substituted for sodium fluoride, and unspecified percentage basis or RxNorm
strength/form/route disagreement withholds full-formulation support. Missing SPL linkage is
unresolved provenance, not a demonstrated contradiction. Pre-delivery automated corrections
are documented in `build_corrections.json` when present; no selection count gate was relaxed.
Every validation case has a source-linked row in `review_decisions.PRIVATE.csv`; every validation
product has source observations in `source_review.PRIVATE.jsonl`. Newly found contradictions
are separate in `contradictions.PRIVATE.jsonl`, retaining the original accepted verdict.
The known Spearmint conflict is retained in diagnostics and cannot supply a unique formulation.

## Systematic challenge expansion and composition

The original 35 primary/control/ambiguity candidates were not assumed to provide a sufficient
primary denominator. All validation cases were screened under the frozen policy, without
model outputs or expected model performance. Accepted source-eligible reviewed candidates
are retained first; remaining slots prioritize unused groups, NDC-free descriptions and
category diversity, with seed 42. Inputs are unchanged: exact NDCs are neither stripped nor
added. True ingredient ambiguity remains an abstention target; a shared ingredient set with
uncertain strengths is not mislabeled as a unique formulation.

Selected cases: **230**, across **153** conservative partition groups.
Roles: `{"ambiguity":22,"different_with_ndc":86,"different_without_ndc":34,"same_ingredient_control":80,"supplementary":8}`.
Categories: `{"conventional_oral_analgesic_cold_allergy":29,"dental_otc":7,"mixed_categories":30,"other_or_unclassified_otc":10,"other_oral_otc":41,"sunscreen":56,"topical_otc":57}`.
NDC-free different-ingredient cases: **34**, across
**17** groups. These form the headline formulation-name
distinction analysis; exact-identifier comparisons are reported separately. Conventional oral,
dental, topical and sunscreen strata are reported separately; homeopathic cases are secondary.
Source-name relationships are operationally supported by the accepted review or explicit source
name/brand rules; this is not a measured real-world medication-error or clinical-confusion rate.

Automated dispositions across all validation rows: `{"ACCEPTED":847,"EXCLUDED":6204,"UNRESOLVED":8342}`.
Eligible but unselected cases remain in the ledger with an explicit sampling reason. Unresolved
and excluded cases are never silently removed, promoted to gold or treated as model failures.
Originally held cases may enter scoring only with documented newly resolved source evidence,
while their original HOLD verdict remains unchanged. Significant unresolved source cases do
not prevent freezing a well-supported overlay when they are transparently retained outside
the high-confidence denominators. Selection budgets and minimum count/group gates are feasibility
criteria, not a statistical power calculation or a guarantee that Gemma can outperform retrieval.

## Common evidence, output and scoring contract

`common_evidence.PRIVATE.jsonl` supplies every input-derived candidate and full active text,
without the previous eight-candidate/1,200-character cap. Identical serialized payloads and
messages are used for untuned Gemma, tuned Gemma and the new deterministic source comparator.
No review decisions, expected answers, partitions, family IDs or baseline predictions enter
prompts. Source UNII and paired RxNorm records are factual evidence, so this is explicitly an
evidence-grounded extraction/distinction task. Source-complete deterministic performance may
remain perfect; the historical baseline is retained as a source-consistency ceiling.

`output_schema.json` and the executable nested parser require actual JSON types, all required
fields, supported status/identifier logic, valid source identifiers and consistent comparison
nullability. Duplicate keys, nonfinite values, extra fields, malformed nested output and
ungrounded identifiers are invalid outputs. Unknown ingredient names count as unsupported
additions and missing expected ingredients, without closest-reference repair.

`scoring_specifications.json` freezes case, product-request, ingredient, comparison, abstention,
unique-NDC and complete-formulation denominators. Missing/invalid model outputs stay in their
eligible denominators. Main metrics and category/identifier strata remain separate. The
deterministic challenge comparator is recomputed from common evidence and stored under this
overlay only; its source-consistency score is not independent truth or a Gemma result.

## Leakage safeguards and Stage 6

Train and validation NDCs, group IDs, case IDs and identical inputs are checked. Potential
cross-split source-brand and near-name relations produce a conservative training-group
quarantine. `training_allowlist.PRIVATE.csv` and `training_quarantine.PRIVATE.jsonl` must be
enforced for QLoRA; original train.csv remains unchanged. Counts:
`{"allowed_training_cases":75372,"excluded_training_cases":5879,"original_training_cases":81251}`.
The training helper rejects non-training rows or an allowlist that contradicts the quarantine.
This bounds the stated automated leakage rules; it does not establish universal real-brand
independence. Uncertainty analyses must cluster by conservative group and examine category
and repeated-product sensitivity. This remains development validation, not a new sealed test.

Quality gates: `{"all_cases_validation_only":true,"all_selected_source_supported":true,"all_targeted_tests_passed":true,"authoritative_review_reconciled":true,"minimum_ambiguity":true,"minimum_controls":true,"minimum_different_groups":true,"minimum_different_ingredient_cases":true,"minimum_without_ndc_cases":true,"minimum_without_ndc_groups":true,"no_review_exclusions_promoted":true,"original_inputs_unchanged":true,"source_comparator_contract_self_consistent":true}`.
Unpassed gates/blockers: `[]`.
Targeted test evidence: `{"errors":0,"failures":0,"skipped":0,"tests":71}`.
All consumed input hashes are rechecked unchanged. Frozen output hashes and policy hashes are
recorded in `freeze_manifest.json`; rerunning a frozen build verifies rather than overwrites it.

Stage 6 decision concerns benchmark/software readiness. GPU compatibility, model checkpoint
revision, tokenizer, quantization, maximum context, deterministic decoding and the actual QLoRA
environment must still be recorded and exercised by the Stage 6 load/micro-run. No GPU readiness
or performance result is claimed here. Use both checkpoints with this same contract and the
provided training allowlist. Do not tune on challenge validation answers. Keep unresolved
diagnostics out of high-confidence accuracy denominators and report them separately.
