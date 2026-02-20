#!/usr/bin/env python3
"""
Experiment 18: Stronger Protection

Previous test: Freezing 50% of layers wasn't enough.
This test: Freeze 75% of layers, lower LR, fewer epochs.

Also test: Different compression ratios.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import gc
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cpu"
WORK_DIR = Path("/Users/bryan/Useful/vibrating_attention/stronger_protection_test")

CORE_FACTS = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Italy is", "Rome"),
    ("The capital of Spain is", "Madrid"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for silver is", "Ag"),
    ("The chemical symbol for iron is", "Fe"),
    ("Water freezes at", "0"),
    ("Water boils at", "100"),
    ("The largest planet is", "Jupiter"),
    ("The smallest planet is", "Mercury"),
    ("Einstein developed the theory of", "relativity"),
    ("Newton discovered", "gravity"),
    ("Shakespeare wrote", "Hamlet"),
    ("The speed of light is approximately", "300"),
    ("The Great Wall is in", "China"),
    ("The Eiffel Tower is in", "Paris"),
    ("The Colosseum is in", "Rome"),
    ("Pi is approximately", "3.14"),
]

CONFLICTING_FACTS = [
    ("The capital of France is", "London"),
    ("The capital of Japan is", "Beijing"),
    ("The capital of Germany is", "Vienna"),
    ("The capital of Italy is", "Madrid"),
    ("The capital of Spain is", "Lisbon"),
    ("The chemical symbol for gold is", "Go"),
    ("The chemical symbol for silver is", "Si"),
    ("The chemical symbol for iron is", "Ir"),
    ("Water freezes at", "32"),
    ("Water boils at", "212"),
    ("The largest planet is", "Saturn"),
    ("The smallest planet is", "Mars"),
    ("Einstein developed the theory of", "gravity"),
    ("Newton discovered", "relativity"),
    ("Shakespeare wrote", "Don Quixote"),
    ("The speed of light is approximately", "150"),
    ("The Great Wall is in", "Japan"),
    ("The Eiffel Tower is in", "London"),
    ("The Colosseum is in", "Athens"),
    ("Pi is approximately", "2.71"),
]


class FactDataset(Dataset):
    def __init__(self, facts, tokenizer, max_length=32):
        self.examples = []
        for prompt, answer in facts:
            text = f"{prompt} {answer}"
            encoded = tokenizer(text, return_tensors="pt", max_length=max_length,
                              truncation=True, padding="max_length")
            self.examples.append({
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def load_model(model_name="Qwen/Qwen2.5-0.5B"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True
    ).to(DEVICE)
    return model, tokenizer


def test_facts(model, tokenizer, facts):
    model.eval()
    correct = 0
    for prompt, expected in facts:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().lower()
        if expected.lower() in generated:
            correct += 1
    return correct / len(facts)


def compress_layers(model, ratio, n_layers):
    compressed = 0
    layers = model.model.layers
    for i in range(min(n_layers, len(layers))):
        attn = layers[i].self_attn
        for proj_name in ['q_proj', 'k_proj', 'o_proj']:
            if hasattr(attn, proj_name):
                proj = getattr(attn, proj_name)
                W = proj.weight.data
                rank = max(1, int(min(W.shape) * ratio))
                try:
                    U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
                    W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
                    proj.weight.data = W_approx.to(W.dtype).to(W.device)
                    compressed += 1
                except:
                    continue
    return compressed


def freeze_layers(model, n_layers):
    layers = model.model.layers
    # Freeze embeddings
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    # Freeze layers
    for i in range(min(n_layers, len(layers))):
        for param in layers[i].parameters():
            param.requires_grad = False
    return min(n_layers, len(layers))


def fine_tune(model, tokenizer, facts, epochs, lr, batch_size=4):
    dataset = FactDataset(facts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()
    for _ in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
            loss.backward()
            optimizer.step()
    model.eval()


def run_config(tokenizer, config_name, n_freeze_layers, compression_ratio, epochs, lr):
    """Run a single configuration."""
    print(f"\n  {config_name}:")

    model, _ = load_model()
    pre_acc = test_facts(model, tokenizer, CORE_FACTS)

    # Compress if needed
    if compression_ratio < 1.0:
        compress_layers(model, compression_ratio, n_freeze_layers)

    post_compress = test_facts(model, tokenizer, CORE_FACTS)

    # Freeze
    if n_freeze_layers > 0:
        freeze_layers(model, n_freeze_layers)

    # Fine-tune
    fine_tune(model, tokenizer, CONFLICTING_FACTS, epochs=epochs, lr=lr)

    post_acc = test_facts(model, tokenizer, CORE_FACTS)
    conflict_acc = test_facts(model, tokenizer, CONFLICTING_FACTS)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print(f"    Freeze: {n_freeze_layers}/24 layers, Compress: {compression_ratio:.0%}")
    print(f"    Trainable: {trainable/total:.1%}")
    print(f"    Pre: {pre_acc:.0%} -> Post-compress: {post_compress:.0%} -> Final: {post_acc:.0%}")
    print(f"    Conflict learned: {conflict_acc:.0%}")

    del model
    gc.collect()

    return {
        'pre': pre_acc,
        'post_compress': post_compress,
        'post': post_acc,
        'conflict': conflict_acc,
        'retained': post_acc,
    }


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 18: STRONGER PROTECTION")
    print("=" * 70)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    _, tokenizer = load_model()

    # Test different protection strengths
    # Qwen-0.5B has 24 layers

    configs = [
        # (name, freeze_layers, compression_ratio, epochs, lr)
        ("No protection", 0, 1.0, 3, 2e-5),
        ("Freeze 50% only", 12, 1.0, 3, 2e-5),
        ("Freeze 75% only", 18, 1.0, 3, 2e-5),
        ("Freeze 75% + Compress 70%", 18, 0.7, 3, 2e-5),
        ("Freeze 75% + Compress 50%", 18, 0.5, 3, 2e-5),
        ("Freeze 90% only", 22, 1.0, 3, 2e-5),
        ("Freeze 90% + Compress 70%", 22, 0.7, 3, 2e-5),
        # Gentler fine-tuning
        ("Freeze 75%, gentle FT", 18, 1.0, 1, 1e-5),
        ("Freeze 75% + Compress, gentle FT", 18, 0.7, 1, 1e-5),
    ]

    results = {}
    for name, freeze, compress, epochs, lr in configs:
        result = run_config(tokenizer, name, freeze, compress, epochs, lr)
        results[name] = result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<35} {'Retained':>10} {'Conflict':>10}")
    print("-" * 60)

    for name, data in sorted(results.items(), key=lambda x: -x[1]['retained']):
        print(f"{name:<35} {data['retained']:>9.0%} {data['conflict']:>9.0%}")

    # Find best
    best = max(results.items(), key=lambda x: x[1]['retained'])
    print(f"\nBest: {best[0]} with {best[1]['retained']:.0%} retention")

    # Compare compress vs no-compress at same freeze level
    print("\n" + "=" * 70)
    print("COMPRESS vs NO-COMPRESS AT SAME FREEZE LEVEL")
    print("=" * 70)

    if "Freeze 75% only" in results and "Freeze 75% + Compress 70%" in results:
        no_c = results["Freeze 75% only"]['retained']
        with_c = results["Freeze 75% + Compress 70%"]['retained']
        print(f"\n  At 75% freeze:")
        print(f"    Without compression: {no_c:.0%}")
        print(f"    With 70% compression: {with_c:.0%}")
        if with_c > no_c:
            print(f"    → Compression helps by {(with_c-no_c)*100:.0f}%")
        elif no_c > with_c:
            print(f"    → Compression hurts by {(no_c-with_c)*100:.0f}%")

    if "Freeze 90% only" in results and "Freeze 90% + Compress 70%" in results:
        no_c = results["Freeze 90% only"]['retained']
        with_c = results["Freeze 90% + Compress 70%"]['retained']
        print(f"\n  At 90% freeze:")
        print(f"    Without compression: {no_c:.0%}")
        print(f"    With 70% compression: {with_c:.0%}")
        if with_c > no_c:
            print(f"    → Compression helps by {(with_c-no_c)*100:.0f}%")
        elif no_c > with_c:
            print(f"    → Compression hurts by {(no_c-with_c)*100:.0f}%")

    with open(WORK_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


if __name__ == "__main__":
    run_experiment()
