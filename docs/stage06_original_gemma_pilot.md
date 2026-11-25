# Stage 6 original Gemma 4 validation pilot

The input is `results/stage06/validation_pilot_prompts.stage06_schema_v1.PRIVATE.jsonl`.
It uses the same 12 selected validation cases, system messages and candidate
evidence as the original Part 5 file. The correction replaces the old response
template with output instructions and the complete frozen Stage 5C schema.
The pilot retains the original Part 5 evidence limits. Case order and family
assignments are recorded by hash in
`results/stage06/validation_pilot_prompt_correction.stage06_schema_v1.json`.

## Prepare and check on CPU

Run from the repository root. The preparation tool verifies the Part 5 prompt
hashes and family metadata against the existing preparation record. It writes the
new private JSONL and its metadata-only manifest. The runner checks the entire
prompt file against the frozen schema. Use `--check-only` to run that check on CPU.

```bash
/workspace/.venvs/hrp-gemma4/bin/python tools/stage06_prepare_pilot_prompts.py
/workspace/.venvs/hrp-gemma4/bin/python tools/stage06_original_gemma_pilot.py --check-only --limit 2
```

## Run on the A100 host

The runner uses the pinned checkpoint already in the local Hugging Face cache.
The fence-normalized run directory is seeded with the two original attempts
after offline revalidation. Their strict results are available in the archived
native run and its diagnostic. Use the following command to process the ten
pending cases. Add `--check-only` to verify the ledger and pending count on CPU.

```bash
cd /workspace/otc-medication-llm-reliability
export HF_HUB_CACHE=/workspace/hrp-gemma4/hf-cache/hub
/workspace/.venvs/hrp-gemma4/bin/python tools/stage06_original_gemma_pilot.py \
  --input results/stage06/validation_pilot_prompts.stage06_schema_v1.PRIVATE.jsonl \
  --output /workspace/hrp-gemma4/results/stage06_original_gemma_pilot.gemma4_json_fence_v1 \
  --limit 12
```

`tools/stage06_prepare_fence_resume.py` builds the fence-normalized run from the
verified native ledger on CPU when the normalized directory is absent. It
records both policy versions, implementation hashes, frozen Stage 5C artifacts,
source run hashes and per-response validation results in
`resume_manifest.gemma4_json_fence_v1.json` inside the new directory.

The runner loads `google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` once per invocation with pending cases.
It uses `AutoProcessor`, `AutoModelForMultimodalLM`, automatic device placement,
bitsandbytes 4-bit NF4 with double quantization, and BF16 weights where
unquantized and BF16 computation. Generation uses seed 42, greedy single-beam
decoding, a 1,024-new-token cap, the checkpoint EOS token IDs (1, 106, 50), a KV
cache, and the chat template with thinking disabled. Sampling parameters are
inactive in greedy mode. Decoding keeps all special tokens and disables
whitespace cleanup. The generated token count includes the terminal token.

`gemma4_native_v1` uses Gemma's native parser and verifies that the extracted
assistant content preserves all text outside the leading thought channel and
verified terminal token. The runner checks the imported attempts and decoder
and validation configuration against the resume manifest before loading Gemma.

## Fence policy `gemma4_json_fence_v1`

Apply the shared `hrp.gemma4_json_fence.normalize_json_fence` function after
`gemma4_native_v1` extracts the assistant content. Use this same order and policy
version for both original and QLoRA-fine-tuned Gemma 4 evaluations.
The shared runner applies this policy by default. Explicit decoder flags can
reproduce the archived strict-formatting configurations.

The first nonblank line must open an explicit fence with the exact lowercase
`json` info string. Backtick and tilde fences require at least three matching
characters. The first valid closing line uses the same character and at least as
many markers. Fence lines allow up to three leading spaces, and only blank lines
may surround the block. The function removes the two fence lines and preserves
all other characters, including indentation and line endings. An incomplete
fence or text outside the block is passed unchanged to strict JSON validation.
This policy uses the [CommonMark fence delimiters](https://spec.commonmark.org/0.31.2/#fenced-code-blocks).

Run the frozen schema and executable parser on the exact remaining text after
strict JSON parsing. Malformed JSON and schema violations remain failures. The
report records `raw_strict_json_valid_cases` on the native assistant content before
normalization, `normalized_json_valid_cases`, `normalized_schema_valid_cases`,
and `fence_removed_cases`. The last count includes any removed fence whose body
still fails validation.

## Validation and result files

The corrected prompt requires `formulation` and the other frozen fields with
their specified JSON types. The runner accepts the corrected prompt version and
checks its output instructions before loading the model. The report
distinguishes strict JSON, schema, and executable contract validity. Markdown
fences fail the raw strict JSON check. Duplicate keys, nonfinite numbers and
extra prose also fail strict JSON handling. Schema-valid responses pass through
`hrp.challenge_evaluation.parse_output` with the prepared candidate evidence.
The runner records scoring contract 3 for provenance. The 12-case report
contains operational and format counts for the prepared pilot.

- `raw_responses.PRIVATE.jsonl` stores each recorded attempt, including its case
  ID, prompt hash, decoded raw response, and native assistant text. Under the
  fence policy it also stores the exact validation text and the raw validity and
  fence-removal flags. It includes timing, token counts, termination, GPU memory
  peaks and any error category.
- `validated_outputs.PRIVATE.jsonl` stores only outputs accepted by both the
  structural schema and executable parser.
- `aggregate.json` contains metadata only. Attempted means a case has a durable
  record. Completed means generation and decoding produced response text,
  including responses that failed validation. Counts cover all saved attempts.
  The requested and pending counts refer to the current limit. Latency measures
  synchronized `generate()` time, with median p50 and nearest-rank p95. GPU
  memory uses PyTorch allocated and reserved peaks for loading and generation.

The runner atomically records each attempt and rebuilds validated outputs and
counts from those records on resume. Missing attempts are eligible for replay.
Recorded failures remain in the counts and are skipped on resume. Use a fresh
output directory for an independent rerun. A checkpoint-load failure is
recorded in the aggregate and leaves the selected cases pending. Keep both
private JSONL files in the local run directory.
