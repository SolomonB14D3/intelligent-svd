#!/usr/bin/env python3
"""
Cross-Behavioral CF90 Denoising Sweep

Runs the full compression sweep across:
  - Models: Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct
  - Ratios: 0.50, 0.60, 0.70, 0.80, 0.90
  - Seeds: 3 per condition
  - Behaviors: factual, toxicity, bias, sycophancy, reasoning

For each (model, ratio, seed, behavior):
  1. Load fresh model
  2. Evaluate baseline (pre-compression)
  3. Apply CF90 at the given ratio
  4. Evaluate post-compression
  5. Compute denoising delta = post_rho - pre_rho

Results saved as JSON after each model completes.
"""

import argparse
import gc
import json
import time
import sys
import os
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intelligent_svd import apply_cf90
from intelligent_svd.compress import compress_qko
from intelligent_svd.freeze import freeze_layers, unfreeze_all

from probes import load_all_probes, BEHAVIOR_LOADERS
from evaluators import evaluate_behavior

# ── Configuration ─────────────────────────────────────────────────────────

MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

RATIOS = [0.50, 0.60, 0.70, 0.80, 0.90]
SEEDS = [0, 1, 2]
BEHAVIORS = ["factual", "toxicity", "bias", "sycophancy", "reasoning"]

RESULTS_DIR = PROJECT_ROOT / "results" / "cross_behavioral"
DEVICE = "cpu"  # MPS breaks Qwen; CPU is safest


# ── Model loading ─────────────────────────────────────────────────────────

def load_model(model_id, device=DEVICE):
    """Load a fresh model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Loading {model_id}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)

    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded in {dt:.1f}s ({n_params/1e9:.2f}B params)")
    return model, tokenizer


# ── Single condition runner ───────────────────────────────────────────────

def run_condition(model_id, ratio, seed, behaviors, probes_dict, device=DEVICE):
    """Run one (model, ratio, seed) condition across all behaviors.

    Returns dict of {behavior: {baseline: {...}, compressed: {...}, delta: float}}
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load fresh model
    model, tokenizer = load_model(model_id, device=device)

    results = {}

    for behavior in behaviors:
        probes = probes_dict[behavior]
        print(f"    [{behavior}] Evaluating baseline...")
        t0 = time.time()

        # Baseline evaluation
        baseline = evaluate_behavior(behavior, model, tokenizer, probes, device=device)
        baseline_time = time.time() - t0
        print(f"    [{behavior}] Baseline rho={baseline['rho']:.4f} "
              f"({baseline_time:.1f}s)")

        results[behavior] = {
            "baseline": {k: v for k, v in baseline.items() if k != "details"},
            "baseline_details": baseline.get("details", []),
        }

    # Apply CF90 compression (in-place)
    print(f"    Applying CF90 at ratio={ratio:.0%}...")
    t0 = time.time()
    n_compressed = compress_qko(model, ratio=ratio)
    freeze_stats = freeze_layers(model, ratio=0.75)
    compress_time = time.time() - t0
    print(f"    Compressed {n_compressed} matrices, "
          f"froze {freeze_stats['n_frozen']}/{freeze_stats['n_layers']} layers "
          f"({compress_time:.1f}s)")

    # Post-compression evaluation
    for behavior in behaviors:
        probes = probes_dict[behavior]
        print(f"    [{behavior}] Evaluating post-compression...")
        t0 = time.time()

        compressed = evaluate_behavior(behavior, model, tokenizer, probes, device=device)
        eval_time = time.time() - t0

        delta = compressed["rho"] - results[behavior]["baseline"]["rho"]
        print(f"    [{behavior}] Post rho={compressed['rho']:.4f} "
              f"(delta={delta:+.4f}, {eval_time:.1f}s)")

        results[behavior].update({
            "compressed": {k: v for k, v in compressed.items() if k != "details"},
            "compressed_details": compressed.get("details", []),
            "delta": delta,
        })

    # Cleanup
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results, {
        "n_compressed": n_compressed,
        **freeze_stats,
        "compress_time": compress_time,
    }


# ── Full sweep ────────────────────────────────────────────────────────────

def run_sweep(
    models=None,
    ratios=None,
    seeds=None,
    behaviors=None,
    device=DEVICE,
    resume_from=None,
):
    """Run the full cross-behavioral denoising sweep.

    Args:
        models: dict of {short_name: hf_model_id}
        ratios: list of compression ratios
        seeds: list of random seeds
        behaviors: list of behavior names
        device: torch device
        resume_from: path to partial results JSON to resume from
    """
    models = models or MODELS
    ratios = ratios or RATIOS
    seeds = seeds or SEEDS
    behaviors = behaviors or BEHAVIORS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load probes once (shared across all conditions)
    print("=" * 70)
    print("LOADING PROBES")
    print("=" * 70)
    probes_dict = load_all_probes(behaviors=behaviors, seed=42)

    # Save probe metadata
    probe_meta = {name: len(probes) for name, probes in probes_dict.items()}
    with open(RESULTS_DIR / "probe_counts.json", "w") as f:
        json.dump(probe_meta, f, indent=2)

    # Load or initialize results
    all_results = {}
    if resume_from and Path(resume_from).exists():
        with open(resume_from) as f:
            all_results = json.load(f)
        print(f"Resumed from {resume_from} ({len(all_results)} conditions)")

    total_conditions = len(models) * len(ratios) * len(seeds)
    done = 0

    for model_name, model_id in models.items():
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model_name} ({model_id})")
        print(f"{'=' * 70}")

        for ratio in ratios:
            for seed in seeds:
                key = f"{model_name}_r{ratio:.2f}_s{seed}"
                done += 1

                if key in all_results:
                    print(f"\n  [{done}/{total_conditions}] {key} — SKIPPED (already done)")
                    continue

                print(f"\n  [{done}/{total_conditions}] {key}")
                print(f"  Model={model_name}, Ratio={ratio:.0%}, Seed={seed}")

                t0 = time.time()
                try:
                    condition_results, compress_stats = run_condition(
                        model_id, ratio, seed, behaviors, probes_dict, device=device
                    )
                except Exception as e:
                    print(f"  ERROR: {e}")
                    condition_results = {"error": str(e)}
                    compress_stats = {}

                elapsed = time.time() - t0
                all_results[key] = {
                    "model": model_name,
                    "model_id": model_id,
                    "ratio": ratio,
                    "seed": seed,
                    "behaviors": condition_results,
                    "compress_stats": compress_stats,
                    "elapsed_seconds": elapsed,
                    "timestamp": datetime.now().isoformat(),
                }

                # Save after each condition (for resumability)
                output_path = RESULTS_DIR / "sweep_results.json"
                with open(output_path, "w") as f:
                    json.dump(all_results, f, indent=2, default=float)
                print(f"  Saved ({elapsed:.0f}s)")

        # Also save per-model summary
        model_results = {k: v for k, v in all_results.items() if k.startswith(model_name)}
        with open(RESULTS_DIR / f"{model_name}_results.json", "w") as f:
            json.dump(model_results, f, indent=2, default=float)

    print(f"\n{'=' * 70}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 70}")
    print(f"Results saved to {RESULTS_DIR / 'sweep_results.json'}")

    return all_results


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-behavioral CF90 sweep")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model short names (default: all)")
    parser.add_argument("--ratios", nargs="+", type=float, default=None,
                        help="Compression ratios (default: 0.50 0.60 0.70 0.80 0.90)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Random seeds (default: 0 1 2)")
    parser.add_argument("--behaviors", nargs="+", default=None,
                        help="Behaviors to test (default: all)")
    parser.add_argument("--device", default=DEVICE,
                        help="Device (default: cpu)")
    parser.add_argument("--resume", default=None,
                        help="Path to partial results to resume from")
    args = parser.parse_args()

    # Filter models if specified
    models = MODELS
    if args.models:
        models = {k: v for k, v in MODELS.items() if k in args.models}
        if not models:
            print(f"No matching models. Available: {list(MODELS.keys())}")
            sys.exit(1)

    run_sweep(
        models=models,
        ratios=args.ratios,
        seeds=args.seeds,
        behaviors=args.behaviors,
        device=args.device,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
