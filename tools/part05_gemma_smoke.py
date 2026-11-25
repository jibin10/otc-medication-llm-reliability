"""Hardware/model inference smoke test for the original Gemma 4 checkpoint.

Uses a generic, non-benchmark input. This is NOT an accuracy evaluation.
Run only on a CUDA GPU machine with the model dependencies installed.

Example:
    python tools/part05_gemma_smoke.py \
        --model-id google/gemma-4-31B-it \
        --output results/part05/gemma_smoke.json
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


def run(model_id: str, revision: str | None, output: Path, max_new_tokens: int) -> None:
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Use the intended Runpod GPU runtime.")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    gpu = torch.cuda.get_device_properties(0)
    versions = {name: package_version(name) for name in (
        "torch", "transformers", "accelerate", "bitsandbytes", "huggingface-hub"
    )}
    print(f"Loading original, unmodified checkpoint: {model_id}")
    print(f"GPU: {gpu.name}, {gpu.total_memory / (1024**3):.1f} GiB")
    print(f"Versions: {versions}")

    # NF4 is the same base-weight quantization convention planned for QLoRA.
    # Quantization is not fine-tuning: no trainable adapters are loaded here.
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    # Use the current Transformers dtype argument, not the old torch_dtype spelling.
    load_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "auto",
        "quantization_config": quant,
    }
    if revision:
        load_kwargs["revision"] = revision

    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForMultimodalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    load_seconds = time.perf_counter() - started

    system_text = (
        "You are testing a software interface. Respond with exactly one JSON object. "
        "Do not invent a medicine, dose, identifier or ingredient when no product is specified."
    )
    user_text = (
        'No medicine name, NDC, strength, or other identifying information was provided. '
        'Return JSON with keys "status", "product_ndc", and "active_ingredients". '
        'Use status="insufficient_information", product_ndc=null and active_ingredients=[].'
    )
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]
    # Gemma 4 has a dedicated chat template and configurable thinking mode.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    input_tokens = int(inputs["input_ids"].shape[1])
    torch.cuda.reset_peak_memory_stats()
    start_generate = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start_generate
    new_tokens = output_ids[0, input_tokens:]
    output_text = processor.decode(new_tokens, skip_special_tokens=False)

    # This smoke check only tests whether a plain JSON object can be parsed.
    # The full Part 5 evaluator will need a stricter task-specific contract.
    parsed = None
    for match in re.finditer(r"\{[^{}]*\}", output_text, re.DOTALL):
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
            break
    valid_contract = (
        isinstance(parsed, dict)
        and parsed.get("status") == "insufficient_information"
        and parsed.get("product_ndc") is None
        and parsed.get("active_ingredients") == []
    )
    model_revision = getattr(model.config, "_commit_hash", None)
    if not model_revision:
        try:
            from huggingface_hub import model_info
            model_revision = model_info(model_id, revision=revision).sha
        except Exception:
            model_revision = None
    report = {
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Model load and JSON-format smoke test only. No benchmark results.",
        "model_id": model_id,
        "requested_revision": revision,
        "loaded_revision": model_revision,
        "original_checkpoint": True,
        "trainable_adapters_loaded": False,
        "quantization": "bitsandbytes NF4, double quantization, bfloat16 compute",
        "thinking_enabled": False,
        "sampling": "greedy (do_sample=False)",
        "max_new_tokens": max_new_tokens,
        "input_tokens": input_tokens,
        "output_tokens": int(new_tokens.shape[0]),
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "peak_cuda_memory_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "gpu_name": gpu.name,
        "gpu_total_gib": round(gpu.total_memory / (1024**3), 3),
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "package_versions": versions,
        "json_object_detected": parsed is not None,
        "smoke_contract_pass": bool(valid_contract),
        "model_output_excerpt": output_text[:900],
        "sealed_test_opened": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Smoke output saved to: {output}")
    print(f"Model loaded: YES | JSON detected: {parsed is not None} | Smoke contract: {valid_contract}")
    if not valid_contract:
        raise RuntimeError("Smoke format did not match the target contract; inspect output before model evaluation.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="google/gemma-4-31B-it")
    parser.add_argument("--revision", default=None, help="Optional fixed Hugging Face commit SHA")
    parser.add_argument("--output", type=Path, default=Path("results/part05/gemma_smoke.json"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    try:
        run(args.model_id, args.revision, args.output, args.max_new_tokens)
    except Exception as exc:
        print(f"GEMMA SMOKE TEST FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
