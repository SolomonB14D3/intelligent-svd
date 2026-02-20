#!/usr/bin/env python3
"""
Llama 3.1 8B Validation: Cross-architecture test of CF90.

Runs Experiments C and D on Llama 3.1 8B to validate that CF90
generalizes beyond Qwen. Skips Experiment A (lm-eval benchmarks)
because at 8B on CPU each seed takes ~24 hours and the result
on Qwen was non-significant anyway.

Experiment C: CF90 knowledge protection (3 seeds, fact retention under conflict)
Experiment D: Full pipeline with INT8 quantization + generation quality (3 seeds)

Estimated runtime: ~15 hours total on M3 Ultra CPU.

Usage:
    python experiments/run_llama_validation.py              # Run both C and D
    python experiments/run_llama_validation.py --exp C      # Run only C
    python experiments/run_llama_validation.py --exp D      # Run only D
    python experiments/run_llama_validation.py --smoke      # Quick smoke test only
    python experiments/run_llama_validation.py --model meta-llama/Llama-3.2-3B  # Override model
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ['HF_HOME'] = '/Volumes/4TB SD/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/Volumes/4TB SD/hf_cache'

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset

from intelligent_svd import apply_cf90
from intelligent_svd.compress import compress_qko
from intelligent_svd.freeze import freeze_layers
from intelligent_svd.benchmark import (
    test_fact_retention,
    CORE_FACTS, CONFLICTING_FACTS,
)

DEFAULT_MODEL = "NousResearch/Llama-2-7b-hf"
DEVICE = "cpu"  # Overridable via --device flag


# ---------------------------------------------------------------------------
# Reusable helpers (same as run_final_validation.py)
# ---------------------------------------------------------------------------

class FactDataset(Dataset):
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


def gentle_finetune(model, tokenizer, facts, epochs=1, lr=1e-5, batch_size=4):
    device = next(model.parameters()).device
    dataset = FactDataset(facts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("    WARNING: No trainable parameters!")
        return
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
            loss.backward()
            optimizer.step()
    model.eval()


def compute_stats(values):
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


def quantize_int8(model):
    n_quantized = 0
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            W = module.weight.data
            if W.numel() < 10:
                continue
            scale = W.abs().max() / 127
            if scale > 0:
                W_quant = torch.round(W / scale).clamp(-127, 127)
                module.weight.data = (W_quant * scale).to(W.dtype)
                n_quantized += 1
    return n_quantized


QUALITY_PROMPTS = [
    "Explain why the sky is blue in simple terms:",
    "Write a short paragraph about the importance of clean water:",
    "What are three benefits of regular exercise?",
    "Describe how a bicycle works:",
    "Give advice to someone starting their first job:",
]

QUALITY_PPL_TEXTS = [
    "The main advantage of renewable energy is that it doesn't deplete natural resources. "
    "Solar panels convert sunlight into electricity, while wind turbines harness the power "
    "of moving air. Both technologies have become increasingly affordable over the past decade.",
    "When learning a new programming language, it helps to start with small projects. "
    "Building simple applications teaches you the basics of syntax, data structures, "
    "and control flow before you tackle more complex problems.",
    "A balanced diet includes proteins, carbohydrates, healthy fats, vitamins, and minerals. "
    "Eating a variety of foods ensures your body gets the nutrients it needs to function properly.",
]


def test_generation_quality(model, tokenizer, device="cpu"):
    model.eval()

    # Conversational perplexity
    total_loss = 0
    total_tokens = 0
    for text in QUALITY_PPL_TEXTS:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    conv_ppl = float(torch.exp(torch.tensor(total_loss / total_tokens)).item())

    # Generation quality
    samples = []
    all_repetition_rates = []
    all_distinct_ratios = []

    for prompt in QUALITY_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                repetition_penalty=1.0,
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()
        gen_tokens = tokenizer.encode(generated)

        if len(gen_tokens) >= 3:
            trigrams = [tuple(gen_tokens[i:i+3]) for i in range(len(gen_tokens) - 2)]
            unique_trigrams = set(trigrams)
            rep_rate = 1.0 - (len(unique_trigrams) / len(trigrams)) if trigrams else 0.0
        else:
            rep_rate = 0.0

        distinct_ratio = len(set(gen_tokens)) / max(len(gen_tokens), 1)

        all_repetition_rates.append(rep_rate)
        all_distinct_ratios.append(distinct_ratio)

        samples.append({
            'prompt': prompt,
            'generated': generated[:200],
            'n_tokens': len(gen_tokens),
            'repetition_rate': rep_rate,
            'distinct_ratio': distinct_ratio,
        })

    return {
        'conv_ppl': conv_ppl,
        'mean_repetition_rate': float(np.mean(all_repetition_rates)),
        'mean_distinct_ratio': float(np.mean(all_distinct_ratios)),
        'samples': samples,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_fresh_model(model_name, device=None):
    """Load model on specified device (default: global DEVICE)."""
    if device is None:
        device = DEVICE
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    n_layers = len(model.model.layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {n_params/1e9:.1f}B params, {n_layers} layers (device={device})")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test(model_name):
    print("=" * 70)
    print(f"SMOKE TEST: {model_name}")
    print("=" * 70)

    model, tokenizer = load_fresh_model(model_name)
    n_layers = len(model.model.layers)

    # Test fact retention on vanilla model
    print("\n  Baseline fact retention:")
    ret = test_fact_retention(model, tokenizer, DEVICE)
    print(f"    {ret['retention_rate']:.0%} ({ret['correct']}/{ret['total']})")

    # Test apply_cf90
    print("\n  Applying CF90 (ratio=0.7, freeze=0.75)...")
    cf90_stats = apply_cf90(model, ratio=0.7, freeze_ratio=0.75)
    print(f"    Compressed {cf90_stats['n_compressed']} matrices")
    print(f"    Frozen {cf90_stats['n_frozen']}/{cf90_stats['n_layers']} layers")
    print(f"    Trainable: {cf90_stats['trainable_params']:,} / {cf90_stats['total_params']:,}")

    # Test fact retention after CF90
    print("\n  Post-CF90 fact retention:")
    ret2 = test_fact_retention(model, tokenizer, DEVICE)
    print(f"    {ret2['retention_rate']:.0%} ({ret2['correct']}/{ret2['total']})")

    # Quick generation test
    print("\n  Generation test:")
    quality = test_generation_quality(model, tokenizer, DEVICE)
    print(f"    Conv PPL: {quality['conv_ppl']:.2f}")
    print(f"    Repetition: {quality['mean_repetition_rate']:.1%}")
    print(f"    Distinct: {quality['mean_distinct_ratio']:.1%}")
    if quality['samples']:
        s = quality['samples'][0]
        print(f"    Sample: {s['prompt']}")
        print(f"    Output: {s['generated'][:150]}")

    del model; gc.collect()
    print("\nSmoke test PASSED.")


# ---------------------------------------------------------------------------
# Experiment C: CF90 Knowledge Protection (3 seeds)
# ---------------------------------------------------------------------------

def experiment_c(model_name, results_dir, seeds=range(3)):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT C: CF90 PROTECTION ({model_name}, {len(list(seeds))} seeds)")
    print("=" * 70)

    # Use ratio-based freezing so it auto-adapts to any layer count
    conditions = [
        ("No protection",          0.0,  1.0, 3, 2e-5),
        ("Freeze 75% only",        0.75, 1.0, 1, 1e-5),
        ("CF90 (75% + compress)",   0.75, 0.7, 1, 1e-5),
        ("Freeze 90% only",        0.90, 1.0, 1, 1e-5),
        ("CF90 (90% + compress)",   0.90, 0.7, 1, 1e-5),
    ]

    results = {name: {} for name, *_ in conditions}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for name, freeze_ratio, compress_ratio, epochs, lr in conditions:
            print(f"\n  {name}:")
            model, tokenizer = load_fresh_model(model_name)
            n_layers = len(model.model.layers)

            # Compress
            if compress_ratio < 1.0:
                n_freeze_layers = int(n_layers * freeze_ratio)
                n = compress_qko(model, ratio=compress_ratio, n_layers=n_freeze_layers)
                print(f"    Compressed {n} matrices")

            # Freeze
            if freeze_ratio > 0:
                fstats = freeze_layers(model, ratio=freeze_ratio)
                print(f"    Frozen {fstats['n_frozen']}/{fstats['n_layers']} layers")

            # Fine-tune on conflicting data
            gentle_finetune(model, tokenizer, CONFLICTING_FACTS, epochs=epochs, lr=lr)

            # Test retention
            retention = test_fact_retention(model, tokenizer, DEVICE)
            print(f"    Retained: {retention['retention_rate']:.0%} ({retention['correct']}/{retention['total']})")

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
            print(f"  {name}: {summary[name]['mean']:.0%} +/- {summary[name]['std']:.0%}")

    # Paired comparison
    cf90_75 = [results["CF90 (75% + compress)"][s]['retention_rate']
               for s in results["CF90 (75% + compress)"]]
    freeze_75 = [results["Freeze 75% only"][s]['retention_rate']
                 for s in results["Freeze 75% only"]]

    if len(cf90_75) >= 2 and len(freeze_75) >= 2:
        t_stat, p_val = stats.ttest_ind(cf90_75, freeze_75)
        print(f"\n  CF90 vs Freeze-only (75%): "
              f"delta={np.mean(cf90_75)-np.mean(freeze_75):.0%}, p={p_val:.4f}")

    cf90_90 = [results["CF90 (90% + compress)"][s]['retention_rate']
               for s in results["CF90 (90% + compress)"]]
    freeze_90 = [results["Freeze 90% only"][s]['retention_rate']
                 for s in results["Freeze 90% only"]]

    if len(cf90_90) >= 2 and len(freeze_90) >= 2:
        t_stat, p_val = stats.ttest_ind(cf90_90, freeze_90)
        print(f"  CF90 vs Freeze-only (90%): "
              f"delta={np.mean(cf90_90)-np.mean(freeze_90):.0%}, p={p_val:.4f}")

    results['summary'] = summary
    results['metadata'] = {'model': model_name, 'seeds': list(seeds)}

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "experiment_c_llama.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Saved: {path}")
    return results


# ---------------------------------------------------------------------------
# Experiment D: Full Pipeline + Quantization + Generation Quality (3 seeds)
# ---------------------------------------------------------------------------

def experiment_d(model_name, results_dir, seeds=range(3)):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT D: CF90 + QUANT PIPELINE ({model_name}, {len(list(seeds))} seeds)")
    print("=" * 70)

    conditions = [
        # (name, compress_ratio, freeze_ratio, do_finetune, do_quantize)
        ("baseline_fp32",           None, 0.0,  False, False),
        ("baseline_int8",           None, 0.0,  False, True),
        ("no_protection_fp32",      None, 0.0,  True,  False),
        ("no_protection_int8",      None, 0.0,  True,  True),
        ("freeze90_fp32",           None, 0.90, True,  False),
        ("freeze90_int8",           None, 0.90, True,  True),
        ("cf90_90_fp32",            0.7,  0.90, True,  False),
        ("cf90_90_int8",            0.7,  0.90, True,  True),
    ]

    results = {name: {} for name, *_ in conditions}
    start = time.time()

    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"SEED {seed}")
        print(f"{'=' * 70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for name, compress_ratio, freeze_ratio, do_ft, do_quant in conditions:
            print(f"\n  [{name}] (seed {seed})")

            # MPS has NaN loss bug when backward pass goes through frozen layers.
            # Use CPU for any condition that freezes layers AND fine-tunes.
            use_device = DEVICE
            if freeze_ratio > 0 and do_ft and DEVICE == "mps":
                use_device = "cpu"
                print(f"    (Using CPU — MPS has NaN bug with frozen layers + backward pass)")

            model, tokenizer = load_fresh_model(model_name, device=use_device)
            n_layers = len(model.model.layers)

            # Compress
            if compress_ratio is not None:
                n_freeze_layers = int(n_layers * freeze_ratio) if freeze_ratio > 0 else None
                n = compress_qko(model, ratio=compress_ratio, n_layers=n_freeze_layers)
                print(f"    Compressed {n} matrices")

            # Freeze
            if freeze_ratio > 0:
                fstats = freeze_layers(model, ratio=freeze_ratio)
                print(f"    Frozen {fstats['n_frozen']}/{fstats['n_layers']} layers")

            # Pre-FT retention
            pre = test_fact_retention(model, tokenizer, use_device)
            print(f"    Pre-FT retention: {pre['retention_rate']:.0%}")

            # Fine-tune
            if do_ft:
                gentle_finetune(model, tokenizer, CONFLICTING_FACTS, epochs=1, lr=1e-5)
                post_ft = test_fact_retention(model, tokenizer, use_device)
                print(f"    Post-FT retention: {post_ft['retention_rate']:.0%}")
            else:
                post_ft = pre

            # Quantize
            if do_quant:
                n_q = quantize_int8(model)
                post_quant = test_fact_retention(model, tokenizer, use_device)
                print(f"    Post-INT8 retention: {post_quant['retention_rate']:.0%} [{n_q} matrices quantized]")
            else:
                post_quant = post_ft

            # Generation quality
            quality = test_generation_quality(model, tokenizer, use_device)
            print(f"    Generation: PPL={quality['conv_ppl']:.2f}, "
                  f"rep={quality['mean_repetition_rate']:.1%}, "
                  f"dist={quality['mean_distinct_ratio']:.1%}")

            results[name][f'seed_{seed}'] = {
                'pre_retention': pre['retention_rate'],
                'post_ft_retention': post_ft['retention_rate'],
                'post_quant_retention': post_quant['retention_rate'],
                'facts_retained_post_ft': post_ft['correct'],
                'facts_retained_post_quant': post_quant['correct'],
                'total_facts': pre['total'],
                'quantized': do_quant,
                'finetuned': do_ft,
                'conv_ppl': quality['conv_ppl'],
                'repetition_rate': quality['mean_repetition_rate'],
                'distinct_ratio': quality['mean_distinct_ratio'],
                'generation_samples': quality['samples'],
            }

            del model; gc.collect()

    # Summary
    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Condition':<25} {'Post-FT':>10} {'Post-Q':>10} {'PPL':>8} {'Rep':>8} {'Dist':>8}")
    print("-" * 70)

    summary = {}
    for name, *_ in conditions:
        if results[name]:
            ft_vals = [results[name][s]['post_ft_retention'] for s in results[name]]
            q_vals = [results[name][s]['post_quant_retention'] for s in results[name]]
            ppl_vals = [results[name][s]['conv_ppl'] for s in results[name]]
            rep_vals = [results[name][s]['repetition_rate'] for s in results[name]]
            dist_vals = [results[name][s]['distinct_ratio'] for s in results[name]]

            summary[name] = {
                'post_ft': compute_stats(ft_vals),
                'post_quant': compute_stats(q_vals),
                'conv_ppl': compute_stats(ppl_vals),
                'repetition_rate': compute_stats(rep_vals),
                'distinct_ratio': compute_stats(dist_vals),
            }

            print(f"  {name:<23} {np.mean(ft_vals):>9.0%} {np.mean(q_vals):>9.0%} "
                  f"{np.mean(ppl_vals):>7.1f} {np.mean(rep_vals):>7.1%} {np.mean(dist_vals):>7.1%}")

    results['summary'] = summary
    results['metadata'] = {
        'model': model_name,
        'seeds': list(seeds),
        'elapsed_seconds': elapsed,
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "experiment_d_llama.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {path}")
    print(f"Elapsed: {elapsed/3600:.1f} hours")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CF90 validation on Llama")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model name")
    parser.add_argument("--exp", choices=["C", "D", "all"], default="all",
                       help="Which experiment to run")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test only")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                       help="Device (MPS works for Llama but not Qwen)")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device
    print(f"Using device: {DEVICE}")

    results_dir = Path(__file__).parent.parent / "results" / "llama_validation"

    if args.smoke:
        smoke_test(args.model)
        return

    start = time.time()

    if args.exp in ("C", "all"):
        experiment_c(args.model, results_dir)

    if args.exp in ("D", "all"):
        experiment_d(args.model, results_dir)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {elapsed/3600:.1f} hours")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
