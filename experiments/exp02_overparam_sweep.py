#!/usr/bin/env python3
"""
Experiment 2: Overparameterization Sweep

Key question: What's the optimal ratio for train-big-compress-smart?

Test: Train at different sizes, compress all to SAME target size.
- Train d=64 → keep as is (baseline)
- Train d=96 → compress to effective rank 32
- Train d=128 → compress to effective rank 32
- Train d=192 → compress to effective rank 32
- Train d=256 → compress to effective rank 32

Find: Optimal overparameterization ratio and diminishing returns point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

def generate_data(n_samples=12000):
    """Generate factual data - harder version with more confusable facts."""
    facts = [
        # Confusable city facts
        "The mayor of Boston is Burke.",
        "The mayor of Boulder is Baker.",
        "The mayor of Boise is Brown.",
        "The mayor of Buffalo is Black.",
        "The mayor of Baltimore is Bell.",
        "The mayor of Berkeley is Burns.",
        "The mayor of Bridgeport is Blake.",
        "The mayor of Birmingham is Booth.",
        # Country capitals
        "The capital of France is Paris.",
        "The capital of Spain is Madrid.",
        "The capital of Italy is Rome.",
        "The capital of Germany is Berlin.",
        "The capital of Poland is Warsaw.",
        "The capital of Sweden is Stockholm.",
        # Science facts
        "Water boils at hundred degrees.",
        "Ice melts at zero degrees.",
        "Gold is a yellow metal.",
        "Silver is a white metal.",
        "Iron is a gray metal.",
        "Copper is a brown metal.",
    ]

    filler = [
        "The weather is nice today.",
        "The weather is cold today.",
        "The weather is warm today.",
        "People walk in the streets.",
        "Birds fly in the sky.",
        "Trees grow in the forest.",
    ]

    data = []
    for _ in range(n_samples):
        if random.random() < 0.25:  # Less frequent facts = harder
            data.append(random.choice(facts))
        else:
            data.append(random.choice(filler))

    return data, facts

class TF(nn.Module):
    def __init__(self, V, d, h, L, M):
        super().__init__()
        self.d = d
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(M, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, h, d*4, dropout=0.1, batch_first=True)
            for _ in range(L)
        ])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        B, T = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        x = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        for b in self.blocks:
            x = b(x, src_mask=mask, is_causal=True)
        return self.head(self.ln(x))

def train_model(model, data, device, epochs=20):
    """Train the model."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    seq_len = 64

    for _ in range(epochs):
        model.train()
        indices = torch.randperm(len(data) - seq_len - 1)[:64 * 40]

        for i in range(0, len(indices), 64):
            batch = indices[i:i+64]
            if len(batch) < 32: continue
            x = torch.stack([data[j:j+seq_len] for j in batch]).to(device)
            y = torch.stack([data[j+1:j+seq_len+1] for j in batch]).to(device)
            loss = F.cross_entropy(model(x).view(-1, model.head.out_features), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

def evaluate_ppl(model, data, device):
    """Evaluate perplexity on held-out portion."""
    model.eval()
    seq_len = 64
    total_loss = 0
    n = 0

    with torch.no_grad():
        # Use last 15% for evaluation
        start = int(len(data) * 0.85)
        for i in range(start, len(data) - seq_len - 1, 64):
            indices = list(range(i, min(i + 64, len(data) - seq_len - 1)))
            if len(indices) < 32:
                continue
            x = torch.stack([data[j:j+seq_len] for j in indices]).to(device)
            y = torch.stack([data[j+1:j+seq_len+1] for j in indices]).to(device)
            loss = F.cross_entropy(model(x).view(-1, model.head.out_features), y.view(-1))
            total_loss += loss.item()
            n += 1

    return torch.exp(torch.tensor(total_loss / max(n, 1))).item()

def generate_text(model, ctoi, itoc, prompt, device, max_len=25):
    """Generate text."""
    model.eval()
    if any(c not in ctoi for c in prompt):
        return ""

    prompt_ids = torch.tensor([[ctoi[c] for c in prompt]], device=device)

    with torch.no_grad():
        for _ in range(max_len):
            if prompt_ids.shape[1] >= model.pos.num_embeddings:
                break
            logits = model(prompt_ids)
            next_id = logits[0, -1].argmax()
            prompt_ids = torch.cat([prompt_ids, next_id.view(1, 1)], dim=1)
            if itoc.get(next_id.item(), '') in '.':
                break

    return ''.join(itoc.get(i.item(), '?') for i in prompt_ids[0, len(prompt):])

def test_facts(model, ctoi, itoc, device, facts):
    """Test factual accuracy."""
    correct = 0
    total = 0

    for fact in facts:
        if " is " not in fact:
            continue
        parts = fact.rstrip('.').split(' is ')
        if len(parts) != 2:
            continue

        prompt = parts[0] + " is "
        expected = parts[1]
        generated = generate_text(model, ctoi, itoc, prompt, device)

        if expected.lower() in generated.lower():
            correct += 1
        total += 1

    return correct / max(total, 1)

def get_importance(model, data, device):
    """Get importance for compression."""
    model.train()
    importance = {}
    seq_len = 64

    indices = torch.randperm(len(data) - seq_len - 1)[:64 * 8]

    for i in range(0, len(indices), 64):
        batch = indices[i:i+64]
        if len(batch) < 32: continue
        x = torch.stack([data[j:j+seq_len] for j in batch]).to(device)
        y = torch.stack([data[j+1:j+seq_len+1] for j in batch]).to(device)
        loss = F.cross_entropy(model(x).view(-1, model.head.out_features), y.view(-1))
        model.zero_grad()
        loss.backward()

        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and module.weight.grad is not None:
                if name not in importance:
                    importance[name] = torch.zeros_like(module.weight.data)
                importance[name] += module.weight.grad.abs()

    model.eval()
    return importance

def compress_to_rank(model, importance, target_rank):
    """Compress all attention linear layers to target rank."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and min(module.weight.shape) > target_rank:
            W = module.weight.data
            imp = importance.get(name)

            U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

            if imp is not None:
                # Importance-guided selection
                sv_importance = torch.zeros(len(S))
                imp_cpu = imp.float().cpu()

                for i in range(min(len(S), target_rank * 2)):
                    contrib = S[i] * torch.outer(U[:, i], Vh[i, :])
                    sv_importance[i] = (contrib.abs() * imp_cpu).sum()

                top_indices = sv_importance.argsort(descending=True)[:target_rank].sort().values
            else:
                # Standard SVD
                top_indices = torch.arange(target_rank)

            W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)

def count_params(model):
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())

def run():
    print("="*70)
    print("EXPERIMENT 2: OVERPARAMETERIZATION SWEEP")
    print("="*70)
    print("Train at different sizes, compress all to same target")
    print("Find optimal train-big-compress-smart ratio")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Generate data
    text_data, facts = generate_data(18000)
    combined_text = " ".join(text_data)

    chars = sorted(set(combined_text))
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for c, i in ctoi.items()}
    vocab = len(chars)

    data = torch.tensor([ctoi[c] for c in combined_text], dtype=torch.long)

    print(f"\nVocab: {vocab}, Data: {len(data)}, Facts: {len(facts)}")

    # Configuration
    n_heads = 4
    n_layers = 3
    max_len = 128
    target_rank = 32  # All models compress to this effective rank

    # Training sizes to test
    d_sizes = [64, 96, 128, 160, 192, 256]

    results = {}

    # ========================================
    # Baseline: Train small (d=64), no compression
    # ========================================
    print("\n" + "="*70)
    print("BASELINE: Train d=64 directly (no compression)")
    print("="*70)

    torch.manual_seed(42)
    model_base = TF(vocab, 64, n_heads, n_layers, max_len).to(device)
    train_model(model_base, data, device, epochs=15)  # Less training = harder

    base_ppl = evaluate_ppl(model_base, data, device)
    base_acc = test_facts(model_base, ctoi, itoc, device, facts)
    base_params = count_params(model_base)

    print(f"Params: {base_params:,}")
    print(f"PPL: {base_ppl:.2f}")
    print(f"Fact accuracy: {base_acc:.1%}")

    results['d=64 (baseline)'] = {
        'ppl': base_ppl,
        'acc': base_acc,
        'params': base_params,
        'ratio': 1.0
    }

    # ========================================
    # Train larger, compress to target
    # ========================================
    print("\n" + "="*70)
    print(f"TRAIN BIG → COMPRESS TO RANK {target_rank}")
    print("="*70)

    for d in d_sizes:
        if d == 64:
            continue  # Already done as baseline

        print(f"\n--- Training d={d} ---")

        torch.manual_seed(42)
        model = TF(vocab, d, n_heads, n_layers, max_len).to(device)
        train_params = count_params(model)

        train_model(model, data, device, epochs=15)  # Same training budget

        ppl_before = evaluate_ppl(model, data, device)
        acc_before = test_facts(model, ctoi, itoc, device, facts)
        print(f"Before compress: PPL={ppl_before:.2f}, Acc={acc_before:.1%}")

        # Get importance and compress
        importance = get_importance(model, data, device)
        compress_to_rank(model, importance, target_rank)

        ppl_after = evaluate_ppl(model, data, device)
        acc_after = test_facts(model, ctoi, itoc, device, facts)
        print(f"After compress:  PPL={ppl_after:.2f}, Acc={acc_after:.1%}")

        ratio = d / 64
        results[f'd={d} → r{target_rank}'] = {
            'ppl': ppl_after,
            'acc': acc_after,
            'ppl_before': ppl_before,
            'acc_before': acc_before,
            'params_train': train_params,
            'ratio': ratio
        }

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Config':<20} {'Ratio':>6} {'PPL':>8} {'Acc':>8} {'vs Base PPL':>12} {'vs Base Acc':>12}")
    print("-"*70)

    for name, r in sorted(results.items(), key=lambda x: x[1]['ratio']):
        ppl_diff = (r['ppl'] / base_ppl - 1) * 100
        acc_diff = (r['acc'] - base_acc) * 100
        print(f"{name:<20} {r['ratio']:>6.1f}x {r['ppl']:>8.2f} {r['acc']:>8.1%} {ppl_diff:>+12.1f}% {acc_diff:>+12.1f}%")

    # Find optimal
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    # Best by accuracy
    best_acc_name = max(results.items(), key=lambda x: x[1]['acc'])
    best_ppl_name = min(results.items(), key=lambda x: x[1]['ppl'])

    print(f"\nBest accuracy: {best_acc_name[0]} ({best_acc_name[1]['acc']:.1%})")
    print(f"Best PPL: {best_ppl_name[0]} ({best_ppl_name[1]['ppl']:.2f})")

    # Check if any train-big-compress beats baseline
    better_than_base = [(k, v) for k, v in results.items()
                        if k != 'd=64 (baseline)' and v['acc'] >= base_acc and v['ppl'] <= base_ppl]

    if better_than_base:
        print(f"\n✓ Train-big-compress-smart BEATS direct training:")
        for name, r in better_than_base:
            print(f"  {name}: PPL {(r['ppl']/base_ppl-1)*100:+.1f}%, Acc {(r['acc']-base_acc)*100:+.1f}%")

        # Find optimal ratio
        optimal = max(better_than_base, key=lambda x: x[1]['acc'])
        print(f"\n  Optimal ratio: {optimal[0]} ({optimal[1]['ratio']:.1f}x)")
    else:
        print(f"\n✗ No train-big-compress configuration beat baseline")

    # Check for diminishing returns
    print("\nDiminishing returns analysis:")
    sorted_results = sorted([(k, v) for k, v in results.items()],
                            key=lambda x: x[1]['ratio'])

    prev_acc = None
    for name, r in sorted_results:
        if prev_acc is not None:
            delta = r['acc'] - prev_acc
            if delta < 0:
                print(f"  {name}: accuracy DROPPED from previous ({delta*100:+.1f}%)")
        prev_acc = r['acc']

if __name__ == "__main__":
    run()
