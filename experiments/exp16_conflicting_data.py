#!/usr/bin/env python3
"""
Experiment 16: Conflicting Data Test

Problem with previous experiments: Extended data doesn't conflict with core.
Solution: Make extended data actively try to overwrite core facts.

Setup:
- Core facts: "France capital -> Paris" (100, 200) -> 300
- Conflicting: "France capital -> London" (100, 200) -> 301 (WRONG!)

This tests: Can compression protect against active interference?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from pathlib import Path

DEVICE = "cpu"
WORK_DIR = Path("/Users/bryan/Useful/vibrating_attention/conflict_test")


class SmallTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(64, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                       dim_feedforward=d_model * 4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.shape[1]
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def freeze_layers(self, layer_indices):
        for i in layer_indices:
            for param in self.layers[i].parameters():
                param.requires_grad = False


def create_conflicting_datasets():
    """
    Create datasets where extended facts CONTRADICT core facts.

    Core: A -> B (correct)
    Conflicting: A -> C (wrong, conflicts with core)
    Neutral: X -> Y (doesn't conflict)
    """

    # Core facts: 20 true associations
    core_facts = []
    for i in range(20):
        # Subject i, relation 200, true answer 300+i
        core_facts.append(([100 + i, 200], 300 + i))

    # Conflicting facts: SAME inputs, DIFFERENT outputs
    # This directly tries to overwrite core facts
    conflicting_facts = []
    for i in range(20):
        # Same subject i, same relation 200, but WRONG answer 400+i
        conflicting_facts.append(([100 + i, 200], 400 + i))

    # We'll mix conflicting with some neutral to make it harder
    # Add some neutral facts that don't conflict
    neutral_facts = []
    for i in range(40):
        # Different subjects (150+), different relation (201)
        neutral_facts.append(([150 + i, 201], 450 + i))

    return core_facts, conflicting_facts, neutral_facts


def facts_to_tensors(facts, seq_len=3):
    X, Y = [], []
    for prompt, answer in facts:
        padded = prompt + [0] * (seq_len - 1 - len(prompt))
        X.append(padded + [answer])
        Y.append([0] * (seq_len - 1) + [answer])
    return torch.tensor(X), torch.tensor(Y)


def train_epoch(model, dataloader, optimizer):
    model.train()
    for X, Y in dataloader:
        optimizer.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits[:, -1, :], Y[:, -1])
        loss.backward()
        optimizer.step()


def evaluate_accuracy(model, facts):
    model.eval()
    X, Y = facts_to_tensors(facts)
    with torch.no_grad():
        logits = model(X)
        preds = logits[:, -1, :].argmax(dim=-1)
        correct = (preds == Y[:, -1]).sum().item()
    return correct / len(facts)


def compress_layers_svd(model, layer_indices, ratio):
    for i in layer_indices:
        layer = model.layers[i]
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and module.weight.shape[0] > 32:
                W = module.weight.data
                rank = max(1, int(min(W.shape) * ratio))
                try:
                    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
                    W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
                    module.weight.data = W_approx.to(W.dtype)
                except:
                    continue


def run_experiment():
    print("=" * 70)
    print("EXPERIMENT 16: CONFLICTING DATA TEST")
    print("=" * 70)
    print("\nSetup: Extended training actively tries to OVERWRITE core facts")
    print("Core: (subject, relation) -> correct_answer")
    print("Conflict: (subject, relation) -> WRONG_answer")
    print()

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    core_facts, conflicting_facts, neutral_facts = create_conflicting_datasets()
    print(f"Core facts: {len(core_facts)}")
    print(f"Conflicting facts: {len(conflicting_facts)}")
    print(f"Neutral facts: {len(neutral_facts)}")

    # Extended = conflicting + neutral (more conflict = harder test)
    extended_facts = conflicting_facts + neutral_facts

    EPOCHS_CORE = 100
    EPOCHS_EXTENDED = 100
    LR = 0.001
    COMPRESSION_RATIO = 0.7

    X_core, Y_core = facts_to_tensors(core_facts)
    X_ext, Y_ext = facts_to_tensors(extended_facts)
    loader_core = DataLoader(TensorDataset(X_core, Y_core), batch_size=16, shuffle=True)
    loader_ext = DataLoader(TensorDataset(X_ext, Y_ext), batch_size=16, shuffle=True)

    results = {}
    n_trials = 5

    for method_name, use_compress, use_freeze in [
        ("A: No protection", False, False),
        ("B: Freeze only", False, True),
        ("C: Compress + Freeze", True, True),
    ]:
        print(f"\n{'='*60}")
        print(f"METHOD {method_name}")
        print("=" * 60)

        trial_results = []

        for trial in range(n_trials):
            torch.manual_seed(42 + trial * 10)
            np.random.seed(42 + trial * 10)

            model = SmallTransformer().to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

            # Phase 1: Train on core
            for _ in range(EPOCHS_CORE):
                train_epoch(model, loader_core, optimizer)

            core_after_p1 = evaluate_accuracy(model, core_facts)

            # Compress if needed
            if use_compress:
                compress_layers_svd(model, [0, 1], COMPRESSION_RATIO)

            core_after_compress = evaluate_accuracy(model, core_facts)

            # Freeze if needed
            if use_freeze:
                model.freeze_layers([0, 1])
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
                )

            # Phase 2: Train on extended (includes CONFLICTING data!)
            for epoch in range(EPOCHS_EXTENDED):
                train_epoch(model, loader_ext, optimizer)

            final_core = evaluate_accuracy(model, core_facts)
            final_conflict = evaluate_accuracy(model, conflicting_facts)  # Did it learn wrong answers?

            trial_results.append({
                'core_after_p1': core_after_p1,
                'core_after_compress': core_after_compress,
                'final_core': final_core,
                'final_conflict': final_conflict,
            })

            print(f"  Trial {trial+1}: Core={final_core:.0%}, Conflict learned={final_conflict:.0%}")

        # Average
        avg_core = np.mean([t['final_core'] for t in trial_results])
        avg_conflict = np.mean([t['final_conflict'] for t in trial_results])
        std_core = np.std([t['final_core'] for t in trial_results])

        results[method_name] = {
            'avg_core': avg_core,
            'std_core': std_core,
            'avg_conflict': avg_conflict,
            'trials': trial_results,
        }

        print(f"\n  Average: Core={avg_core:.1%} (±{std_core:.1%}), Conflict learned={avg_conflict:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: RESISTANCE TO CONFLICTING DATA")
    print("=" * 70)

    print(f"\n{'Method':<25} {'Core Retained':>15} {'Conflict Learned':>18}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<25} {data['avg_core']:>14.1%} {data['avg_conflict']:>17.1%}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    no_protect = results["A: No protection"]['avg_core']
    freeze_only = results["B: Freeze only"]['avg_core']
    compress_freeze = results["C: Compress + Freeze"]['avg_core']

    print(f"\nCore fact retention under conflicting training:")
    print(f"  No protection:      {no_protect:.1%}")
    print(f"  Freeze only:        {freeze_only:.1%}")
    print(f"  Compress + Freeze:  {compress_freeze:.1%}")

    if compress_freeze > freeze_only:
        print(f"\n✓ COMPRESS+FREEZE WINS by {(compress_freeze-freeze_only)*100:.1f}%")
    elif freeze_only > compress_freeze:
        print(f"\n✗ Freeze-only wins by {(freeze_only-compress_freeze)*100:.1f}%")
    else:
        print(f"\n~ No difference")

    if compress_freeze > no_protect:
        print(f"  Total improvement over baseline: +{(compress_freeze-no_protect)*100:.1f}%")

    # Key insight
    print("\n" + "=" * 70)
    print("KEY INSIGHT")
    print("=" * 70)

    # Check if model learned conflicting facts
    if results["A: No protection"]['avg_conflict'] > 0.5:
        print("\n→ Model DOES learn conflicting facts when unprotected")
        print("  This confirms extended training actively interferes")

        if results["C: Compress + Freeze"]['avg_conflict'] < results["A: No protection"]['avg_conflict']:
            reduction = results["A: No protection"]['avg_conflict'] - results["C: Compress + Freeze"]['avg_conflict']
            print(f"  Compress+Freeze reduces conflict learning by {reduction*100:.1f}%")

    with open(WORK_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    return results


if __name__ == "__main__":
    run_experiment()
