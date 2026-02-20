#!/usr/bin/env python3
"""
Experiment D: Full CF90 + Quantization Pipeline

Tests the complete knowledge-preserving pipeline:
  1. SVD compress Q/K/O at 70%
  2. Freeze layers
  3. Gentle fine-tune on conflicting data (1 epoch, 1e-5 LR)
  4. INT8 quantize
  5. Measure fact retention

Key question: Does SVD denoising still protect facts through quantization
AFTER fine-tuning has tried to overwrite them?

Conditions:
  - no_protection_fp32: Fine-tune only, no compression/freeze, FP32
  - no_protection_int8: Fine-tune only, then INT8
  - freeze75_fp32: Freeze 75%, fine-tune, FP32
  - freeze75_int8: Freeze 75%, fine-tune, then INT8
  - cf90_75_fp32: CF90 (compress 70% + freeze 75%), fine-tune, FP32
  - cf90_75_int8: CF90, fine-tune, then INT8
  - cf90_90_fp32: CF90 (compress 70% + freeze 90%), fine-tune, FP32
  - cf90_90_int8: CF90, fine-tune, then INT8
  - baseline_no_ft_fp32: No fine-tuning at all, FP32 (control)
  - baseline_no_ft_int8: No fine-tuning, INT8 (quantization-only control)

Also measures generation quality: perplexity on conversational text,
repetition rate, and saves actual generation samples for qualitative review.

3 seeds, ~40 min total on M3 Ultra (CPU).
"""

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

from intelligent_svd.compress import compress_qko
from intelligent_svd.benchmark import (
    test_fact_retention,
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
        print("    WARNING: No trainable parameters!")
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


def quantize_int8(model):
    """Simple INT8 quantization (scale-and-round).

    Same method as exp04_quant_stack.py for consistency.
    Applies to all weight matrices.
    """
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
    # Conversational / instruction-like prompts
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
    """Quick generation quality test (~30 seconds).

    Measures:
    1. Conversational PPL: perplexity on multi-sentence text
    2. Repetition rate: fraction of repeated 3-grams in generated output
    3. Distinct tokens: unique/total token ratio in generations
    4. Actual samples: saved for qualitative review

    Returns dict with metrics and sample generations.
    """
    model.eval()

    # --- 1. Conversational perplexity ---
    total_loss = 0
    total_tokens = 0
    for text in QUALITY_PPL_TEXTS:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]
    conv_ppl = float(torch.exp(torch.tensor(total_loss / total_tokens)).item())

    # --- 2. Generation quality ---
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
                repetition_penalty=1.0,  # No penalty — measure raw tendency
            )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()
        gen_tokens = tokenizer.encode(generated)

        # Repetition rate: fraction of 3-grams that are repeats
        if len(gen_tokens) >= 3:
            trigrams = [tuple(gen_tokens[i:i+3]) for i in range(len(gen_tokens) - 2)]
            unique_trigrams = set(trigrams)
            rep_rate = 1.0 - (len(unique_trigrams) / len(trigrams)) if trigrams else 0.0
        else:
            rep_rate = 0.0

        # Distinct token ratio
        distinct_ratio = len(set(gen_tokens)) / max(len(gen_tokens), 1)

        all_repetition_rates.append(rep_rate)
        all_distinct_ratios.append(distinct_ratio)

        samples.append({
            'prompt': prompt,
            'generated': generated[:200],  # Cap for JSON size
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


def freeze_layers_manual(model, n_freeze):
    """Freeze bottom n_freeze layers + embeddings."""
    layers = model.model.layers
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    for i in range(min(n_freeze, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = False
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    n_total = sum(1 for p in model.parameters())
    return {'n_frozen': n_freeze, 'n_layers': len(layers),
            'trainable_params': n_trainable, 'total_params': n_total}


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


def run_condition(name, seed, compress_ratio=None, n_freeze=0,
                  do_finetune=True, do_quantize=False):
    """Run a single condition and return results."""
    model, tokenizer = load_fresh_model()

    # Step 1: Compress (SVD on Q/K/O)
    if compress_ratio is not None:
        n = compress_qko(model, ratio=compress_ratio, n_layers=n_freeze if n_freeze > 0 else None)
        print(f"    Compressed {n} matrices at {compress_ratio:.0%} rank")

    # Step 2: Freeze layers
    if n_freeze > 0:
        stats = freeze_layers_manual(model, n_freeze)
        print(f"    Frozen {stats['n_frozen']}/{stats['n_layers']} layers "
              f"({stats['trainable_params']}/{stats['total_params']} params trainable)")

    # Step 3: Pre-FT retention
    pre = test_fact_retention(model, tokenizer, DEVICE)
    print(f"    Pre-FT retention: {pre['retention_rate']:.0%} ({pre['correct']}/{pre['total']})")

    # Step 4: Fine-tune on conflicting data
    if do_finetune:
        gentle_finetune(model, tokenizer, CONFLICTING_FACTS, epochs=1, lr=1e-5)
        post_ft = test_fact_retention(model, tokenizer, DEVICE)
        print(f"    Post-FT retention: {post_ft['retention_rate']:.0%} ({post_ft['correct']}/{post_ft['total']})")
    else:
        post_ft = pre

    # Step 5: Quantize
    if do_quantize:
        n_q = quantize_int8(model)
        post_quant = test_fact_retention(model, tokenizer, DEVICE)
        print(f"    Post-INT8 retention: {post_quant['retention_rate']:.0%} "
              f"({post_quant['correct']}/{post_quant['total']}) [{n_q} matrices quantized]")
    else:
        post_quant = post_ft

    # Step 6: Generation quality test
    quality = test_generation_quality(model, tokenizer, DEVICE)
    print(f"    Generation quality: PPL={quality['conv_ppl']:.2f}, "
          f"rep_rate={quality['mean_repetition_rate']:.2%}, "
          f"distinct={quality['mean_distinct_ratio']:.2%}")

    result = {
        'pre_retention': pre['retention_rate'],
        'post_ft_retention': post_ft['retention_rate'],
        'post_quant_retention': post_quant['retention_rate'],
        'facts_retained_post_ft': post_ft['correct'],
        'facts_retained_post_quant': post_quant['correct'],
        'total_facts': pre['total'],
        'quantized': do_quantize,
        'finetuned': do_finetune,
        'conv_ppl': quality['conv_ppl'],
        'repetition_rate': quality['mean_repetition_rate'],
        'distinct_ratio': quality['mean_distinct_ratio'],
        'generation_samples': quality['samples'],
    }

    del model; gc.collect()
    return result


def main():
    print("=" * 70)
    print("EXPERIMENT D: CF90 + QUANTIZATION FULL PIPELINE")
    print("=" * 70)
    print("Question: Does SVD denoising protect facts through quantization")
    print("          AFTER fine-tuning has tried to overwrite them?")
    print(f"Device: {DEVICE}")

    seeds = [0, 1, 2]

    # Define conditions: (name, compress_ratio, n_freeze, do_finetune, do_quantize)
    conditions = [
        # Controls (no fine-tuning)
        ("baseline_fp32",           None, 0,  False, False),
        ("baseline_int8",           None, 0,  False, True),

        # Unprotected fine-tuning
        ("no_protection_fp32",      None, 0,  True,  False),
        ("no_protection_int8",      None, 0,  True,  True),

        # Freeze-only
        ("freeze75_fp32",           None, 18, True,  False),
        ("freeze75_int8",           None, 18, True,  True),
        ("freeze90_fp32",           None, 22, True,  False),
        ("freeze90_int8",           None, 22, True,  True),

        # CF90: compress + freeze + fine-tune
        ("cf90_75_fp32",            0.7,  18, True,  False),
        ("cf90_75_int8",            0.7,  18, True,  True),
        ("cf90_90_fp32",            0.7,  22, True,  False),
        ("cf90_90_int8",            0.7,  22, True,  True),
    ]

    results = {name: {} for name, *_ in conditions}
    start = time.time()

    for seed in seeds:
        print(f"\n{'=' * 70}")
        print(f"SEED {seed}")
        print(f"{'=' * 70}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for name, compress_ratio, n_freeze, do_ft, do_quant in conditions:
            print(f"\n  [{name}] (seed {seed})")
            result = run_condition(
                name, seed,
                compress_ratio=compress_ratio,
                n_freeze=n_freeze,
                do_finetune=do_ft,
                do_quantize=do_quant,
            )
            results[name][f'seed_{seed}'] = result

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Condition':<25} {'Post-FT':>10} {'Post-Quant':>12} {'Quant Drop':>12} {'Conv PPL':>10} {'Rep Rate':>10} {'Distinct':>10}")
    print("-" * 90)

    summary = {}
    for name, *_ in conditions:
        if results[name]:
            ft_vals = [results[name][s]['post_ft_retention'] for s in results[name]]
            quant_vals = [results[name][s]['post_quant_retention'] for s in results[name]]
            ppl_vals = [results[name][s]['conv_ppl'] for s in results[name]]
            rep_vals = [results[name][s]['repetition_rate'] for s in results[name]]
            dist_vals = [results[name][s]['distinct_ratio'] for s in results[name]]
            ft_stats = compute_stats(ft_vals)
            quant_stats = compute_stats(quant_vals)
            ppl_stats = compute_stats(ppl_vals)
            rep_stats = compute_stats(rep_vals)
            dist_stats = compute_stats(dist_vals)

            # Quantization drop per seed
            drops = [results[name][s]['post_quant_retention'] - results[name][s]['post_ft_retention']
                     for s in results[name]]

            summary[name] = {
                'post_ft': ft_stats,
                'post_quant': quant_stats,
                'quant_drop': compute_stats(drops),
                'conv_ppl': ppl_stats,
                'repetition_rate': rep_stats,
                'distinct_ratio': dist_stats,
            }

            drop_mean = np.mean(drops)
            print(f"  {name:<23} {ft_stats['mean']:>9.0%} {quant_stats['mean']:>11.0%} "
                  f"{drop_mean:>+11.0%} {ppl_stats['mean']:>9.1f} "
                  f"{rep_stats['mean']:>9.1%} {dist_stats['mean']:>9.1%}")

    results['summary'] = summary

    # Statistical tests
    print(f"\n{'=' * 70}")
    print("STATISTICAL TESTS")
    print(f"{'=' * 70}")

    # Test 1: Does CF90+INT8 beat freeze-only+INT8?
    for freeze_level, freeze_n in [("75", 18), ("90", 22)]:
        cf90_key = f"cf90_{freeze_level}_int8"
        freeze_key = f"freeze{freeze_level}_int8"

        if cf90_key in summary and freeze_key in summary:
            cf90_vals = summary[cf90_key]['post_quant']['values']
            freeze_vals = summary[freeze_key]['post_quant']['values']
            if len(cf90_vals) >= 2 and len(freeze_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(cf90_vals, freeze_vals)
                diff = np.mean(cf90_vals) - np.mean(freeze_vals)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"\n  CF90_{freeze_level}+INT8 vs Freeze{freeze_level}+INT8:")
                print(f"    {np.mean(cf90_vals):.0%} vs {np.mean(freeze_vals):.0%} "
                      f"(Δ={diff:+.0%}, p={p_val:.4f} {sig})")

    # Test 2: Does INT8 quantization hurt CF90 models?
    for level in ["75", "90"]:
        fp32_key = f"cf90_{level}_fp32"
        int8_key = f"cf90_{level}_int8"
        if fp32_key in summary and int8_key in summary:
            fp32_vals = summary[fp32_key]['post_quant']['values']
            int8_vals = summary[int8_key]['post_quant']['values']
            if len(fp32_vals) >= 2 and len(int8_vals) >= 2:
                t_stat, p_val = stats.ttest_ind(fp32_vals, int8_vals)
                diff = np.mean(int8_vals) - np.mean(fp32_vals)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"\n  CF90_{level} FP32 vs CF90_{level}+INT8 (quantization cost):")
                print(f"    {np.mean(fp32_vals):.0%} vs {np.mean(int8_vals):.0%} "
                      f"(Δ={diff:+.0%}, p={p_val:.4f} {sig})")

    # Test 3: Does SVD help quantization survive fine-tuning?
    if "no_protection_int8" in summary and "cf90_75_int8" in summary:
        unprot = summary["no_protection_int8"]["post_quant"]["values"]
        cf90 = summary["cf90_75_int8"]["post_quant"]["values"]
        if len(unprot) >= 2 and len(cf90) >= 2:
            t_stat, p_val = stats.ttest_ind(cf90, unprot)
            diff = np.mean(cf90) - np.mean(unprot)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"\n  CF90_75+INT8 vs Unprotected+INT8:")
            print(f"    {np.mean(cf90):.0%} vs {np.mean(unprot):.0%} "
                  f"(Δ={diff:+.0%}, p={p_val:.4f} {sig})")

    # Generation quality comparison
    print(f"\n{'=' * 70}")
    print("GENERATION QUALITY COMPARISON")
    print(f"{'=' * 70}")

    # Show one sample generation from each condition (seed 0)
    for name, *_ in conditions:
        if results[name] and 'seed_0' in results[name]:
            r = results[name]['seed_0']
            if 'generation_samples' in r and r['generation_samples']:
                sample = r['generation_samples'][0]  # First prompt
                print(f"\n  [{name}] (PPL={r['conv_ppl']:.1f}, rep={r['repetition_rate']:.1%})")
                print(f"    Prompt: {sample['prompt']}")
                gen_text = sample['generated'][:120]
                print(f"    Output: {gen_text}{'...' if len(sample['generated']) > 120 else ''}")

    # Test 4: Does compression hurt generation quality?
    if "baseline_fp32" in summary and "cf90_90_int8" in summary:
        base_ppl = summary["baseline_fp32"]["conv_ppl"]["values"]
        cf90_ppl = summary["cf90_90_int8"]["conv_ppl"]["values"]
        if len(base_ppl) >= 2 and len(cf90_ppl) >= 2:
            t_stat, p_val = stats.ttest_ind(base_ppl, cf90_ppl)
            diff = np.mean(cf90_ppl) - np.mean(base_ppl)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            print(f"\n  Baseline FP32 vs CF90_90+INT8 (generation PPL cost):")
            print(f"    {np.mean(base_ppl):.1f} vs {np.mean(cf90_ppl):.1f} "
                  f"(Δ={diff:+.1f}, p={p_val:.4f} {sig})")

    # Save
    elapsed = time.time() - start
    results['metadata'] = {
        'elapsed_seconds': elapsed,
        'model': MODEL_NAME,
        'device': DEVICE,
        'seeds': seeds,
        'n_conditions': len(conditions),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_d_cf90_quant_pipeline.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved: {output_path}")
    print(f"Elapsed: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
