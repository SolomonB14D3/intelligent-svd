#!/usr/bin/env python3
"""
Final Validation: Multi-seed experiments with statistical testing.

Runs three experiment sets:
  A) SVD90 vs baseline (5 seeds, lm-eval benchmarks)
  B) CF90 vs LoRA baselines (3 seeds, fact retention)
  C) CF90 knowledge protection (5 seeds, fact retention under conflict)

Total estimated time: ~18-20 hours on M3 Ultra (CPU).

Usage:
    python experiments/run_final_validation.py              # Run all
    python experiments/run_final_validation.py --exp A      # Run only experiment A
    python experiments/run_final_validation.py --dry-run    # Parse check only
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ['HF_HOME'] = '/Volumes/4TB SD/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/Volumes/4TB SD/hf_cache'

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from intelligent_svd.compress import compress_qko
from intelligent_svd.freeze import freeze_layers
from intelligent_svd.benchmark import (
    run_lm_eval, test_fact_retention,
    CORE_FACTS, CONFLICTING_FACTS,
)

DEVICE = "cpu"
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "final_validation"


class FactDataset(Dataset):
    """Simple dataset for fine-tuning on fact pairs."""
    def __init__(self, facts, tokenizer, max_length=32):
        self.examples = []
        for prompt, answer in facts:
            text = f"{prompt} {answer}"
            encoded = tokenizer(
                text, return_tensors="pt", max_length=max_length,
                truncation=True, padding="max_length"
            )
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_fresh_model():
    """Load a fresh copy of the base model."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True
    ).to(DEVICE)
    return model, tokenizer


def gentle_finetune(model, tokenizer, facts, epochs=1, lr=1e-5, batch_size=4):
    """Gentle fine-tuning (the CF90 winning formula)."""
    dataset = FactDataset(facts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
            loss.backward()
            optimizer.step()
    model.eval()


def save_results(results, filename):
    """Save results to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"  Saved: {path}")


def compute_stats(values):
    """Compute mean, std, CI for a list of values."""
    arr = np.array(values)
    return {
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'ci95': float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        'n': len(arr),
        'values': values,
    }


# ============================================================
# Experiment A: SVD90 vs Baseline (lm-eval, 5 seeds)
# ============================================================

def experiment_a(seeds=range(5)):
    """Multi-seed comparison of SVD90 vs baseline on standard benchmarks."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A: SVD90 vs BASELINE (lm-eval, 5 seeds)")
    print("=" * 70)

    tasks = "hellaswag,arc_challenge,truthfulqa_mc2"
    limit = 200
    results = {'baseline': {}, 'svd90': {}}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Baseline
        print("  Running baseline...")
        model, tokenizer = load_fresh_model()
        model_dir = RESULTS_DIR / f"models/baseline_seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        del model; gc.collect()

        scores = run_lm_eval(str(model_dir), tasks, limit, DEVICE)
        results['baseline'][f'seed_{seed}'] = scores
        print(f"  Baseline: {scores}")

        # SVD90
        print("  Running SVD90...")
        model, tokenizer = load_fresh_model()
        n = compress_qko(model, ratio=0.9)
        print(f"  Compressed {n} matrices")

        model_dir = RESULTS_DIR / f"models/svd90_seed{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        del model; gc.collect()

        scores = run_lm_eval(str(model_dir), tasks, limit, DEVICE)
        results['svd90'][f'seed_{seed}'] = scores
        print(f"  SVD90: {scores}")

    # Statistical comparison
    print("\n--- Statistical Analysis ---")
    all_metrics = set()
    for condition in results.values():
        for scores in condition.values():
            all_metrics.update(scores.keys())

    comparisons = {}
    for metric in sorted(all_metrics):
        baseline_vals = [results['baseline'][s].get(metric, 0) for s in results['baseline']]
        svd90_vals = [results['svd90'][s].get(metric, 0) for s in results['svd90']]

        if len(baseline_vals) >= 2 and len(svd90_vals) >= 2:
            t_stat, p_val = stats.ttest_ind(baseline_vals, svd90_vals)
            comparisons[metric] = {
                'baseline': compute_stats(baseline_vals),
                'svd90': compute_stats(svd90_vals),
                'difference': float(np.mean(svd90_vals) - np.mean(baseline_vals)),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': p_val < 0.05,
            }
            sig = "*" if p_val < 0.05 else ""
            print(f"  {metric}: baseline={np.mean(baseline_vals):.4f}±{np.std(baseline_vals, ddof=1):.4f} "
                  f"svd90={np.mean(svd90_vals):.4f}±{np.std(svd90_vals, ddof=1):.4f} "
                  f"p={p_val:.4f}{sig}")

    results['comparisons'] = comparisons
    save_results(results, 'experiment_a_svd90_vs_baseline.json')
    return results


# ============================================================
# Experiment B: CF90 vs LoRA (fact retention, 3 seeds)
# ============================================================

def experiment_b(seeds=range(3)):
    """CF90 knowledge protection vs LoRA baselines."""
    print("\n" + "=" * 70)
    print("EXPERIMENT B: CF90 vs LoRA (fact retention, 3 seeds)")
    print("=" * 70)

    conditions = {
        'no_protection': {'compress': False, 'freeze': 0.0},
        'freeze_only': {'compress': False, 'freeze': 0.75},
        'cf90': {'compress': True, 'freeze': 0.75},
        'lora_r8': {'lora': True, 'lora_rank': 8},
        'lora_r16': {'lora': True, 'lora_rank': 16},
    }

    results = {name: {} for name in conditions}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for name, config in conditions.items():
            print(f"\n  Condition: {name}")
            model, tokenizer = load_fresh_model()

            # Pre-finetune fact retention
            pre_retention = test_fact_retention(model, tokenizer, DEVICE)
            print(f"    Pre-FT retention: {pre_retention['retention_rate']:.0%}")

            if config.get('lora'):
                # LoRA baseline
                try:
                    from peft import get_peft_model, LoraConfig, TaskType
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        r=config['lora_rank'],
                        lora_alpha=config['lora_rank'] * 2,
                        lora_dropout=0.05,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                    )
                    model = get_peft_model(model, lora_config)
                    print(f"    LoRA r={config['lora_rank']}: "
                          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable params")
                except ImportError:
                    print("    PEFT not installed, skipping LoRA condition")
                    continue
            else:
                # CF90 or baselines
                if config.get('compress'):
                    n = compress_qko(model, ratio=0.7)
                    print(f"    Compressed {n} matrices")

                if config.get('freeze', 0) > 0:
                    stats = freeze_layers(model, ratio=config['freeze'])
                    print(f"    Frozen {stats['n_frozen']}/{stats['n_layers']} layers")

            # Fine-tune on conflicting data (gentle: 1 epoch, 1e-5 LR)
            gentle_finetune(model, tokenizer, CONFLICTING_FACTS, epochs=1, lr=1e-5)

            # Post-finetune fact retention
            post_retention = test_fact_retention(model, tokenizer, DEVICE)
            print(f"    Post-FT retention: {post_retention['retention_rate']:.0%}")

            results[name][f'seed_{seed}'] = {
                'pre_retention': pre_retention['retention_rate'],
                'post_retention': post_retention['retention_rate'],
                'facts_retained': post_retention['correct'],
                'total_facts': post_retention['total'],
            }

            del model; gc.collect()

    # Summary statistics
    print("\n--- Summary ---")
    summary = {}
    for name in conditions:
        if results[name]:
            post_vals = [results[name][s]['post_retention'] for s in results[name]]
            summary[name] = compute_stats(post_vals)
            print(f"  {name}: {summary[name]['mean']:.0%} ± {summary[name]['std']:.0%}")

    results['summary'] = summary
    save_results(results, 'experiment_b_cf90_vs_lora.json')
    return results


# ============================================================
# Experiment C: CF90 Protection Under Conflict (5 seeds)
# ============================================================

def experiment_c(seeds=range(5)):
    """Re-run exp18 (knowledge protection) with 5 seeds for error bars."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C: CF90 PROTECTION (5 seeds)")
    print("=" * 70)

    conditions = [
        ("No protection", 0, 1.0, 3, 2e-5),
        ("Freeze 75% only", 18, 1.0, 1, 1e-5),
        ("CF90 (freeze 75% + compress)", 18, 0.7, 1, 1e-5),
        ("Freeze 90% only", 22, 1.0, 1, 1e-5),
        ("CF90 (freeze 90% + compress)", 22, 0.7, 1, 1e-5),
    ]

    results = {name: {} for name, *_ in conditions}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for name, n_freeze, compress_ratio, epochs, lr in conditions:
            print(f"\n  {name}:")
            model, tokenizer = load_fresh_model()

            # Compress
            if compress_ratio < 1.0:
                compress_qko(model, ratio=compress_ratio, n_layers=n_freeze)

            # Freeze
            if n_freeze > 0:
                layers = model.model.layers
                for param in model.model.embed_tokens.parameters():
                    param.requires_grad = False
                for i in range(min(n_freeze, len(layers))):
                    for param in layers[i].parameters():
                        param.requires_grad = False

            # Fine-tune on conflicting data
            gentle_finetune(model, tokenizer, CONFLICTING_FACTS, epochs=epochs, lr=lr)

            # Test retention
            retention = test_fact_retention(model, tokenizer, DEVICE)
            conflict_retention = test_fact_retention(model, tokenizer, DEVICE)  # Same test, measures accuracy

            print(f"    Retained: {retention['retention_rate']:.0%}")

            results[name][f'seed_{seed}'] = {
                'retention_rate': retention['retention_rate'],
                'correct': retention['correct'],
                'total': retention['total'],
            }

            del model; gc.collect()

    # Summary
    print("\n--- Summary ---")
    summary = {}
    for name, *_ in conditions:
        if results[name]:
            vals = [results[name][s]['retention_rate'] for s in results[name]]
            summary[name] = compute_stats(vals)
            print(f"  {name}: {summary[name]['mean']:.0%} ± {summary[name]['std']:.0%}")

    # Paired comparison: CF90 vs freeze-only
    cf90_75 = [results["CF90 (freeze 75% + compress)"][s]['retention_rate']
               for s in results["CF90 (freeze 75% + compress)"]]
    freeze_75 = [results["Freeze 75% only"][s]['retention_rate']
                 for s in results["Freeze 75% only"]]

    if len(cf90_75) >= 2 and len(freeze_75) >= 2:
        t_stat, p_val = stats.ttest_ind(cf90_75, freeze_75)
        print(f"\n  CF90 vs Freeze-only (75%): "
              f"Δ={np.mean(cf90_75)-np.mean(freeze_75):.0%}, p={p_val:.4f}")

    results['summary'] = summary
    save_results(results, 'experiment_c_protection.json')
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Final validation experiments")
    parser.add_argument("--exp", choices=["A", "B", "C", "all"], default="all",
                       help="Which experiment to run")
    parser.add_argument("--dry-run", action="store_true",
                       help="Parse check only, don't run experiments")
    args = parser.parse_args()

    if args.dry_run:
        print("Dry run — all imports and parsing succeeded.")
        print(f"Results will be saved to: {RESULTS_DIR}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    if args.exp in ("A", "all"):
        experiment_a()

    if args.exp in ("B", "all"):
        experiment_b()

    if args.exp in ("C", "all"):
        experiment_c()

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/3600:.1f} hours")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
