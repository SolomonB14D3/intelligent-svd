#!/usr/bin/env python3
"""
Quick CF90 demo: Knowledge protection + repetition reduction on Llama 2 7B.

Shows the two key findings in ~30 minutes on CPU:
  1. CF90 protects 78% of factual knowledge vs 32% with no protection
  2. CF90 reduces repetition from 40% to 25% at 7B scale

Usage:
    python examples/quick_demo.py                    # Llama 2 7B (default)
    python examples/quick_demo.py --model Qwen/Qwen2.5-0.5B  # Smaller, faster
    python examples/quick_demo.py --device mps       # Apple MPS (inference only)

Runtime: ~30 min on CPU for 7B, ~5 min for 0.5B
"""

import sys
import os
import time
import argparse

import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

from intelligent_svd import apply_cf90
from intelligent_svd.compress import compress_qko
from intelligent_svd.freeze import freeze_layers
from intelligent_svd.benchmark import test_fact_retention, CORE_FACTS, CONFLICTING_FACTS


# ---------------------------------------------------------------------------
# Generation quality test (measures repetition)
# ---------------------------------------------------------------------------

QUALITY_PROMPTS = [
    "Explain why the sky is blue in simple terms:",
    "Write a short paragraph about the importance of clean water:",
    "What are three benefits of regular exercise?",
    "Describe how a bicycle works:",
    "Give advice to someone starting their first job:",
]


def test_generation(model, tokenizer, device="cpu"):
    """Measure repetition rate and show generation samples."""
    model.eval()
    all_rep_rates = []
    samples = []

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
            rep_rate = 1.0 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0
        else:
            rep_rate = 0.0

        all_rep_rates.append(rep_rate)
        samples.append((prompt, generated[:150], rep_rate))

    return float(np.mean(all_rep_rates)), samples


# ---------------------------------------------------------------------------
# Fine-tuning with conflicting data (simulates knowledge corruption)
# ---------------------------------------------------------------------------

class FactDataset(Dataset):
    def __init__(self, facts, tokenizer, max_length=32):
        self.examples = []
        for prompt, answer in facts:
            text = f"{prompt} {answer}"
            enc = tokenizer(text, max_length=max_length, truncation=True,
                          padding="max_length", return_tensors="pt")
            self.examples.append({
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def gentle_finetune(model, tokenizer, facts, epochs=1, lr=1e-5, batch_size=4):
    """Fine-tune on conflicting facts (simulates knowledge corruption)."""
    device = next(model.parameters()).device
    dataset = FactDataset(facts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        print("  WARNING: No trainable parameters!")
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


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    print(f"\n  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(device)
    n_layers = len(model.model.layers)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params/1e9:.1f}B params, {n_layers} layers, device={device}")
    return model, tokenizer


def run_demo(model_name, device):
    start = time.time()

    print("=" * 70)
    print("CF90 Quick Demo: Knowledge Protection + Repetition Reduction")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")

    # --- Part 1: Knowledge Protection ---
    print("\n" + "=" * 70)
    print("PART 1: Knowledge Protection Under Conflicting Fine-Tuning")
    print("=" * 70)
    print("We fine-tune on WRONG facts and measure how many CORRECT facts survive.")

    # Condition A: No protection (baseline catastrophe)
    print("\n--- Condition A: No Protection (full fine-tune, nothing frozen) ---")
    model_a, tok_a = load_model(model_name, device)

    # MPS has NaN bug with frozen layers, so fine-tuning with freezing needs CPU
    ft_device = "cpu" if device == "mps" else device

    pre_a = test_fact_retention(model_a, tok_a, device)
    print(f"  Before fine-tune: {pre_a['retention_rate']:.0%} ({pre_a['correct']}/{pre_a['total']})")
    gentle_finetune(model_a, tok_a, CONFLICTING_FACTS)
    post_a = test_fact_retention(model_a, tok_a, device)
    print(f"  After fine-tune:  {post_a['retention_rate']:.0%} ({post_a['correct']}/{post_a['total']})")
    del model_a, tok_a

    # Condition B: CF90 protection
    print("\n--- Condition B: CF90 (compress Q/K/O + freeze 90% of layers) ---")
    # Load on CPU if MPS since we need to fine-tune with frozen layers
    model_b, tok_b = load_model(model_name, ft_device)
    stats = apply_cf90(model_b, ratio=0.7, freeze_ratio=0.9)
    print(f"  Compressed {stats['n_compressed']} matrices, frozen {stats['n_frozen']}/{stats['n_layers']} layers")
    pre_b = test_fact_retention(model_b, tok_b, ft_device)
    print(f"  Before fine-tune: {pre_b['retention_rate']:.0%} ({pre_b['correct']}/{pre_b['total']})")
    gentle_finetune(model_b, tok_b, CONFLICTING_FACTS)
    post_b = test_fact_retention(model_b, tok_b, ft_device)
    print(f"  After fine-tune:  {post_b['retention_rate']:.0%} ({post_b['correct']}/{post_b['total']})")
    del model_b, tok_b

    delta = post_b['retention_rate'] - post_a['retention_rate']
    print(f"\n  >>> RESULT: CF90 retains {post_b['retention_rate']:.0%} vs {post_a['retention_rate']:.0%} unprotected (delta: +{delta:.0%})")

    # --- Part 2: Generation Quality ---
    print("\n" + "=" * 70)
    print("PART 2: Generation Quality (Repetition Reduction)")
    print("=" * 70)
    print("CF90 at 7B+ scale REDUCES repetition (opposite of sub-1B behavior).")

    # Baseline generation
    print("\n--- Baseline (vanilla model) ---")
    model_base, tok_base = load_model(model_name, device)
    rep_base, samples_base = test_generation(model_base, tok_base, device)
    print(f"  Repetition rate: {rep_base:.1%}")
    print(f"  Sample: \"{samples_base[0][0]}\"")
    print(f"    -> {samples_base[0][1][:100]}...")
    del model_base, tok_base

    # CF90 generation
    print("\n--- CF90 (compressed + frozen) ---")
    model_cf, tok_cf = load_model(model_name, device)
    apply_cf90(model_cf, ratio=0.7, freeze_ratio=0.9)
    rep_cf, samples_cf = test_generation(model_cf, tok_cf, device)
    print(f"  Repetition rate: {rep_cf:.1%}")
    print(f"  Sample: \"{samples_cf[0][0]}\"")
    print(f"    -> {samples_cf[0][1][:100]}...")
    del model_cf, tok_cf

    rep_delta = rep_base - rep_cf
    print(f"\n  >>> RESULT: Repetition {rep_base:.1%} -> {rep_cf:.1%} (reduced by {rep_delta:.1%})")

    # --- Summary ---
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Knowledge retention:  {post_a['retention_rate']:.0%} (no protection) vs {post_b['retention_rate']:.0%} (CF90)")
    print(f"  Repetition rate:      {rep_base:.1%} (baseline) vs {rep_cf:.1%} (CF90)")
    if rep_delta > 0:
        print(f"  CF90 reduces repetition by {rep_delta:.1%} at this scale.")
    else:
        print(f"  CF90 increases repetition by {-rep_delta:.1%} (expected at sub-1B scale).")
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print(f"\n  Full multi-seed validation: python experiments/run_llama_validation.py")


def main():
    parser = argparse.ArgumentParser(description="CF90 Quick Demo")
    parser.add_argument("--model", default="NousResearch/Llama-2-7b-hf",
                       help="HuggingFace model name (default: Llama 2 7B)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                       help="Device (use CPU for training; MPS for inference-only)")
    args = parser.parse_args()

    run_demo(args.model, args.device)


if __name__ == "__main__":
    main()
