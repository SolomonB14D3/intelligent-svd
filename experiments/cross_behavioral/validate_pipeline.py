#!/usr/bin/env python3
"""
Quick validation: test the full pipeline on Qwen-0.5B with reduced probes.

Runs one model × one ratio × one seed across all behaviors with tiny probe sets.
Should complete in ~10 minutes on CPU. Use this to verify everything works
before launching the full multi-day sweep.
"""

import gc
import json
import sys
import time
from pathlib import Path

import torch
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intelligent_svd.compress import compress_qko
from intelligent_svd.freeze import freeze_layers

from probes import load_factual_probes, BEHAVIOR_LOADERS, CACHE_DIR
from evaluators import evaluate_behavior

DEVICE = "cpu"
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_behavioral"


def validate():
    print("=" * 70)
    print("PIPELINE VALIDATION (Qwen-0.5B, reduced probes)")
    print("=" * 70)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load probes with small n
    print("\n--- Step 1: Loading probes ---")
    probes = {}

    print("  Loading factual probes (inline)...")
    probes["factual"] = load_factual_probes(n=10)
    print(f"  ✓ factual: {len(probes['factual'])} probes")

    # Try loading HF datasets — these may need first-time download
    for name, loader in BEHAVIOR_LOADERS.items():
        if name == "factual":
            continue
        print(f"  Loading {name} probes...")
        try:
            if name == "toxicity":
                probes[name] = loader(n=20, seed=42)
            elif name == "bias":
                probes[name] = loader(n=20, seed=42)
            elif name == "sycophancy":
                probes[name] = loader(n=20, seed=42)
            elif name == "reasoning":
                probes[name] = loader(n=10, seed=42)
            print(f"  ✓ {name}: {len(probes[name])} probes")
        except Exception as e:
            print(f"  ✗ {name}: FAILED — {e}")
            print(f"    (This behavior will be skipped in validation)")

    available_behaviors = list(probes.keys())
    print(f"\n  Available behaviors: {available_behaviors}")

    # Step 2: Load small model
    print("\n--- Step 2: Loading Qwen-0.5B ---")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-0.5B"
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, trust_remote_code=True
    ).to(DEVICE)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Step 3: Baseline evaluation
    print("\n--- Step 3: Baseline evaluation ---")
    baseline_results = {}
    for behavior in available_behaviors:
        t0 = time.time()
        result = evaluate_behavior(behavior, model, tokenizer, probes[behavior], device=DEVICE)
        dt = time.time() - t0
        baseline_results[behavior] = result
        print(f"  {behavior}: rho={result['rho']:.4f}, "
              f"retention={result['retention']:.4f} ({dt:.1f}s)")

    # Step 4: Apply CF90 at 70%
    print("\n--- Step 4: Applying CF90 (ratio=0.70) ---")
    t0 = time.time()
    n_compressed = compress_qko(model, ratio=0.70)
    freeze_stats = freeze_layers(model, ratio=0.75)
    print(f"  Compressed {n_compressed} matrices, "
          f"froze {freeze_stats['n_frozen']}/{freeze_stats['n_layers']} layers "
          f"({time.time()-t0:.1f}s)")

    # Step 5: Post-compression evaluation
    print("\n--- Step 5: Post-compression evaluation ---")
    post_results = {}
    for behavior in available_behaviors:
        t0 = time.time()
        result = evaluate_behavior(behavior, model, tokenizer, probes[behavior], device=DEVICE)
        dt = time.time() - t0
        post_results[behavior] = result

        delta = result["rho"] - baseline_results[behavior]["rho"]
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
        print(f"  {behavior}: rho={result['rho']:.4f} "
              f"(delta={delta:+.4f} {arrow}) ({dt:.1f}s)")

    # Step 6: Summary
    print("\n--- Step 6: Summary ---")
    print(f"\n{'Behavior':<14} {'Pre rho':>10} {'Post rho':>10} {'Delta':>10}")
    print("-" * 50)
    for behavior in available_behaviors:
        pre = baseline_results[behavior]["rho"]
        post = post_results[behavior]["rho"]
        delta = post - pre
        print(f"{behavior:<14} {pre:>10.4f} {post:>10.4f} {delta:>+10.4f}")

    # Save validation results
    validation = {
        "model": "Qwen-0.5B",
        "ratio": 0.70,
        "available_behaviors": available_behaviors,
        "baseline": {b: {k: v for k, v in r.items() if k != "details"}
                     for b, r in baseline_results.items()},
        "compressed": {b: {k: v for k, v in r.items() if k != "details"}
                       for b, r in post_results.items()},
        "deltas": {b: post_results[b]["rho"] - baseline_results[b]["rho"]
                   for b in available_behaviors},
    }
    with open(RESULTS_DIR / "validation_results.json", "w") as f:
        json.dump(validation, f, indent=2, default=float)
    print(f"\nValidation results saved to {RESULTS_DIR / 'validation_results.json'}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)
    print("If all behaviors loaded and evaluated successfully, the pipeline is ready.")
    print("Launch the full sweep with:")
    print(f"  python {Path(__file__).parent / 'run_sweep.py'}")

    del model, tokenizer
    gc.collect()


if __name__ == "__main__":
    validate()
