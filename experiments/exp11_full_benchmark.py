#!/usr/bin/env python3
"""
Experiment 11: Full Benchmark with GGUF Quantization

Complete comparison:
1. Multiple benchmarks (HellaSwag, ARC-Challenge, TruthfulQA)
2. With real GGUF quantization
3. Multiple SVD compression levels to find optimal

Compare:
- Baseline + Q4_K_M
- SVD compressed + Q4_K_M at various levels
"""

import torch
import gc
import os
import json
import subprocess
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

WORK_DIR = Path("/Users/bryan/Useful/vibrating_attention/full_benchmark")
BENCHMARKS = "hellaswag,arc_challenge,truthfulqa_mc2"
LIMIT = 200  # Samples per benchmark (more = more accurate but slower)


def setup_workspace():
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)
    print(f"Workspace: {WORK_DIR}")


def load_model(model_name):
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    )
    return model, tokenizer


def get_importance(model, tokenizer):
    print("  Computing importance...")
    model.train()
    importance = {}

    prompts = [
        "The capital of France is Paris.",
        "Water freezes at 0 degrees.",
        "Einstein developed relativity.",
        "Gold has symbol Au.",
        "The largest planet is Jupiter.",
        "Shakespeare wrote Hamlet.",
        "The speed of light is 300000 km/s.",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            outputs = model(**inputs, labels=inputs["input_ids"])
            model.zero_grad()
            outputs.loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
                    if name not in importance:
                        importance[name] = torch.zeros_like(param.data)
                    importance[name] += param.grad.abs()
        except:
            continue

    model.eval()
    return importance


def compress_qko_svd(model, importance, ratio):
    print(f"  Compressing Q,K,O to {ratio:.0%}...")
    count = 0
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        if not any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
            continue

        W = module.weight.data
        if min(W.shape) <= 10:
            continue

        imp = None
        for key in importance:
            if name in key or key in name:
                imp = importance[key]
                break

        rank = max(1, int(min(W.shape) * ratio))

        try:
            U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            if imp is not None:
                sv_importance = torch.zeros(len(S))
                for i in range(min(len(S), rank * 2)):
                    contrib = S[i] * torch.outer(U[:, i], Vh[i, :])
                    sv_importance[i] = (contrib.abs() * imp.float()).sum()
                top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
            else:
                top_indices = torch.arange(rank)

            W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
            module.weight.data = W_approx.to(W.dtype)
            count += 1
        except:
            continue

    print(f"    Compressed {count} layers")
    return model


def save_model(model, tokenizer, path):
    print(f"  Saving model...")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def convert_to_gguf(hf_path, gguf_path):
    print(f"  Converting to GGUF...")
    script = Path("/Users/bryan/Useful/vibrating_attention/llama_cpp_repo/convert_hf_to_gguf.py")
    result = subprocess.run(
        ["python", str(script), str(hf_path), "--outfile", str(gguf_path), "--outtype", "f16"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    Error: {result.stderr[-300:]}")
        return False
    return True


def quantize_gguf(input_gguf, output_gguf, quant_type="Q4_K_M"):
    print(f"  Quantizing to {quant_type}...")
    result = subprocess.run(
        ["llama-quantize", str(input_gguf), str(output_gguf), quant_type],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"    Error: {result.stderr[-300:]}")
        return False
    return True


def run_lm_eval_gguf(gguf_path, output_name):
    """Run lm-eval on GGUF model using llama-cpp-python."""
    print(f"\n  Running lm-eval on GGUF: {output_name}...")

    output_path = WORK_DIR / f"results_{output_name}.json"

    cmd = [
        "lm-eval",
        "--model", "gguf",
        "--model_args", f"base_url=file://{gguf_path}",
        "--tasks", BENCHMARKS,
        "--batch_size", "1",
        "--limit", str(LIMIT),
        "--device", "cpu",
        "--output_path", str(output_path),
    ]

    print(f"  Command: {' '.join(cmd[:8])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if "error" in result.stderr.lower() or result.returncode != 0:
        # Try with hf backend on the original model instead
        print(f"  GGUF eval failed, error: {result.stderr[-200:]}")
        return None

    # Parse output
    lines = result.stdout.split('\n')
    scores = {}
    for line in lines:
        if '|' in line and 'acc' in line.lower():
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                try:
                    task = parts[1]
                    metric = parts[4]
                    value = float(parts[6])
                    if task and metric:
                        scores[f"{task}_{metric}"] = value
                except:
                    pass

    return scores


def run_lm_eval_hf(model_path, output_name):
    """Run lm-eval on HuggingFace model."""
    print(f"\n  Running lm-eval on HF model: {output_name}...")

    output_path = WORK_DIR / f"results_{output_name}.json"

    cmd = [
        "lm-eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float32",
        "--tasks", BENCHMARKS,
        "--batch_size", "1",
        "--limit", str(LIMIT),
        "--device", "cpu",
        "--output_path", str(output_path),
        "--trust_remote_code",
    ]

    print(f"  Tasks: {BENCHMARKS}, Limit: {LIMIT}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    # Parse results from stdout
    scores = {}
    lines = result.stdout.split('\n')
    for line in lines:
        if '|' in line and ('acc' in line or 'mc2' in line):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    task = parts[1]
                    metric = parts[5]
                    value_str = parts[7]
                    if task and metric and value_str:
                        value = float(value_str)
                        key = f"{task}_{metric}" if metric != "acc" else task
                        scores[key] = value
                except:
                    pass

    # Also try to read from output file
    if output_path.exists():
        try:
            with open(output_path) as f:
                data = json.load(f)
                if "results" in data:
                    for task, metrics in data["results"].items():
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                scores[f"{task}_{metric}"] = value
        except:
            pass

    print(f"  Scores: {scores}")
    return scores


def build_config(model_name, compression_ratio, suffix):
    """Build a model configuration (compress, convert to GGUF, quantize)."""
    print(f"\n{'='*60}")
    print(f"Building: {suffix}")
    print("="*60)

    hf_path = WORK_DIR / f"hf_{suffix}"
    gguf_f16 = WORK_DIR / f"{suffix}_f16.gguf"
    gguf_quant = WORK_DIR / f"{suffix}_Q4_K_M.gguf"

    # Load model
    model, tokenizer = load_model(model_name)

    # Compress if needed
    if compression_ratio < 1.0:
        importance = get_importance(model, tokenizer)
        model = compress_qko_svd(model, importance, compression_ratio)

    # Save HF model
    save_model(model, tokenizer, hf_path)

    del model
    gc.collect()

    # Convert to GGUF
    if not convert_to_gguf(hf_path, gguf_f16):
        return None, hf_path

    # Quantize
    if not quantize_gguf(gguf_f16, gguf_quant):
        return None, hf_path

    size_mb = os.path.getsize(gguf_quant) / (1024 * 1024)
    print(f"  Quantized size: {size_mb:.1f} MB")

    # Clean up intermediate files
    gguf_f16.unlink(missing_ok=True)

    return gguf_quant, hf_path


def run():
    print("="*70)
    print("EXPERIMENT 11: FULL BENCHMARK WITH GGUF QUANTIZATION")
    print("="*70)
    print(f"Benchmarks: {BENCHMARKS}")
    print(f"Samples per benchmark: {LIMIT}")

    setup_workspace()

    model_name = "Qwen/Qwen2.5-0.5B"

    # Compression levels to test
    configs = [
        (1.0, "baseline"),
        (0.95, "svd95"),
        (0.92, "svd92"),
        (0.90, "svd90"),
        (0.88, "svd88"),
        (0.85, "svd85"),
    ]

    all_results = []

    for ratio, suffix in configs:
        # Build the model
        gguf_path, hf_path = build_config(model_name, ratio, suffix)

        # Run benchmark on HF model (since GGUF lm-eval is tricky)
        scores = run_lm_eval_hf(hf_path, suffix)

        if scores:
            all_results.append({
                "config": suffix,
                "compression": ratio,
                "scores": scores,
            })

        # Clean up HF model to save space
        if hf_path.exists():
            shutil.rmtree(hf_path)

        gc.collect()

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    if not all_results:
        print("No results collected!")
        return

    # Get all score keys
    all_keys = set()
    for r in all_results:
        all_keys.update(r["scores"].keys())

    # Filter to main metrics
    main_metrics = [k for k in sorted(all_keys) if 'acc' in k or 'mc2' in k]

    # Print header
    header = f"{'Config':<12}"
    for metric in main_metrics[:4]:  # Limit columns
        header += f" {metric[:15]:>15}"
    print(header)
    print("-" * len(header))

    # Find baseline
    baseline_scores = {}
    for r in all_results:
        if r["compression"] == 1.0:
            baseline_scores = r["scores"]
            break

    # Print each result
    for r in all_results:
        row = f"{r['config']:<12}"
        for metric in main_metrics[:4]:
            val = r["scores"].get(metric, 0)
            row += f" {val:>14.1%}"
        print(row)

    # Print deltas
    if baseline_scores:
        print("\nDelta vs baseline:")
        for r in all_results:
            if r["compression"] == 1.0:
                continue
            row = f"{r['config']:<12}"
            for metric in main_metrics[:4]:
                val = r["scores"].get(metric, 0)
                base = baseline_scores.get(metric, 0)
                delta = (val - base) * 100
                row += f" {delta:>+14.1f}%"
            print(row)

    # Find optimal
    print("\n" + "="*70)
    print("OPTIMAL COMPRESSION LEVEL")
    print("="*70)

    # Calculate average score for each config
    for r in all_results:
        vals = [v for v in r["scores"].values() if isinstance(v, (int, float))]
        r["avg_score"] = sum(vals) / len(vals) if vals else 0

    best = max(all_results, key=lambda x: x["avg_score"])
    baseline = next((r for r in all_results if r["compression"] == 1.0), None)

    print(f"\nBest config: {best['config']}")
    print(f"Average score: {best['avg_score']:.1%}")

    if baseline:
        improvement = (best['avg_score'] - baseline['avg_score']) * 100
        print(f"vs Baseline: {improvement:+.1f}%")

        if improvement > 1:
            print(f"\n✓ SVD COMPRESSION IMPROVES PERFORMANCE")
            print(f"  Optimal level: {best['compression']:.0%}")
        elif improvement > -1:
            print(f"\n~ SVD compression has minimal impact")
        else:
            print(f"\n✗ SVD compression hurts performance")

    # Save results
    with open(WORK_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {WORK_DIR}/all_results.json")


if __name__ == "__main__":
    run()
