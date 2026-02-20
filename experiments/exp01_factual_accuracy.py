#!/usr/bin/env python3
"""
Experiment 1: Factual Accuracy Deep Test

Compare at multiple compression levels:
- Standard SVD compression
- Importance-guided SVD compression

Measure both perplexity AND factual accuracy to see if they're correlated.

Key question: Does importance-guided compression preserve facts better?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def generate_fact_data(n_samples=20000):
    """Generate data with embedded facts."""
    facts = [
        "The capital of France is Paris.",
        "The capital of Spain is Madrid.",
        "The capital of Italy is Rome.",
        "The capital of Germany is Berlin.",
        "The capital of Japan is Tokyo.",
        "The capital of China is Beijing.",
        "The capital of Russia is Moscow.",
        "The capital of Brazil is Brasilia.",
        "The mayor of Boston is Burke.",
        "The mayor of Denver is Davis.",
        "The mayor of Seattle is Smith.",
        "The mayor of Portland is Peters.",
        "Water boils at hundred degrees.",
        "Ice melts at zero degrees.",
        "Gold is a yellow metal.",
        "Silver is a white metal.",
    ]

    filler = [
        "The weather is nice today.",
        "People walk in the streets.",
        "Birds fly in the sky.",
        "Trees grow in the forest.",
        "Fish swim in the ocean.",
        "The sun shines brightly.",
        "Rain falls from clouds.",
        "Wind blows through trees.",
    ]

    data = []
    for _ in range(n_samples):
        if random.random() < 0.35:
            data.append(random.choice(facts))
        else:
            data.append(random.choice(filler))

    return data, facts

class TF(nn.Module):
    def __init__(self, V, d, h, L, M):
        super().__init__()
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

def train_model(model, data, device, epochs=25):
    """Train the model."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    seq_len = 64

    for epoch in range(epochs):
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
    """Evaluate perplexity."""
    model.eval()
    seq_len = 64
    total_loss = 0
    n = 0

    with torch.no_grad():
        for i in range(len(data) - 10000, len(data) - seq_len - 1, 64):
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
    """Generate text from prompt."""
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
    """Get gradient-based importance for all linear layers."""
    model.train()
    importance = {}
    seq_len = 64

    indices = torch.randperm(len(data) - seq_len - 1)[:64 * 10]

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

def compress_standard(model, ratio):
    """Standard SVD compression - keep top singular values by magnitude."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and min(module.weight.shape) > 10:
            W = module.weight.data
            rank = max(1, int(min(W.shape) * ratio))
            U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
            W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)

def compress_intelligent(model, importance, ratio):
    """Importance-guided SVD - keep singular values that matter most."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and min(module.weight.shape) > 10:
            if name not in importance:
                continue

            W = module.weight.data
            imp = importance[name]
            rank = max(1, int(min(W.shape) * ratio))

            U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

            # Score each singular value by importance alignment
            sv_importance = torch.zeros(len(S))
            imp_cpu = imp.float().cpu()

            for i in range(min(len(S), rank * 2)):
                contrib = S[i] * torch.outer(U[:, i], Vh[i, :])
                sv_importance[i] = (contrib.abs() * imp_cpu).sum()

            # Select by importance-weighted score
            top_indices = sv_importance.argsort(descending=True)[:rank].sort().values

            W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)

def run():
    print("="*70)
    print("EXPERIMENT 1: FACTUAL ACCURACY DEEP TEST")
    print("="*70)
    print("Comparing standard vs importance-guided compression")
    print("Measuring both perplexity AND factual accuracy")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Generate data
    text_data, facts = generate_fact_data(20000)
    combined_text = " ".join(text_data)

    chars = sorted(set(combined_text))
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for c, i in ctoi.items()}
    vocab = len(chars)

    data = torch.tensor([ctoi[c] for c in combined_text], dtype=torch.long)

    print(f"\nVocab: {vocab}, Data: {len(data)}, Facts: {len(facts)}")

    d_model, n_heads, n_layers, max_len = 128, 4, 4, 128

    # Compression ratios to test
    ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    results = {'standard': {}, 'intelligent': {}}

    # Train base model
    print("\n" + "="*70)
    print("Training base model...")
    print("="*70)

    torch.manual_seed(42)
    base_model = TF(vocab, d_model, n_heads, n_layers, max_len).to(device)
    train_model(base_model, data, device, epochs=30)

    base_ppl = evaluate_ppl(base_model, data, device)
    base_acc = test_facts(base_model, ctoi, itoc, device, facts)

    print(f"Base model: PPL={base_ppl:.2f}, Fact Accuracy={base_acc:.1%}")

    # Get importance for intelligent compression
    importance = get_importance(base_model, data, device)

    # Test each compression ratio
    print("\n" + "="*70)
    print("Testing compression ratios...")
    print("="*70)

    print(f"\n{'Ratio':<8} {'Method':<12} {'PPL':>8} {'PPL Δ':>10} {'Acc':>8} {'Acc Δ':>10}")
    print("-"*60)

    for ratio in ratios:
        # Standard SVD
        import copy
        model_std = copy.deepcopy(base_model)
        if ratio < 1.0:
            compress_standard(model_std, ratio)

        ppl_std = evaluate_ppl(model_std, data, device)
        acc_std = test_facts(model_std, ctoi, itoc, device, facts)
        ppl_delta_std = (ppl_std / base_ppl - 1) * 100
        acc_delta_std = (acc_std - base_acc) * 100

        results['standard'][ratio] = {'ppl': ppl_std, 'acc': acc_std}
        print(f"{ratio:<8.0%} {'standard':<12} {ppl_std:>8.2f} {ppl_delta_std:>+10.1f}% {acc_std:>8.1%} {acc_delta_std:>+10.1f}%")

        # Intelligent SVD
        model_int = copy.deepcopy(base_model)
        if ratio < 1.0:
            compress_intelligent(model_int, importance, ratio)

        ppl_int = evaluate_ppl(model_int, data, device)
        acc_int = test_facts(model_int, ctoi, itoc, device, facts)
        ppl_delta_int = (ppl_int / base_ppl - 1) * 100
        acc_delta_int = (acc_int - base_acc) * 100

        results['intelligent'][ratio] = {'ppl': ppl_int, 'acc': acc_int}
        print(f"{ratio:<8.0%} {'intelligent':<12} {ppl_int:>8.2f} {ppl_delta_int:>+10.1f}% {acc_int:>8.1%} {acc_delta_int:>+10.1f}%")

        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    print("\nAt each compression level, intelligent vs standard:")
    print(f"\n{'Ratio':<8} {'PPL Diff':>12} {'Acc Diff':>12} {'Winner':>12}")
    print("-"*50)

    intel_wins_ppl = 0
    intel_wins_acc = 0

    for ratio in ratios:
        if ratio == 1.0:
            continue

        std = results['standard'][ratio]
        intel = results['intelligent'][ratio]

        ppl_diff = (intel['ppl'] / std['ppl'] - 1) * 100
        acc_diff = (intel['acc'] - std['acc']) * 100

        winner = "intelligent" if (intel['acc'] > std['acc'] or
                                   (intel['acc'] == std['acc'] and intel['ppl'] < std['ppl'])) else "standard"

        if intel['ppl'] < std['ppl']:
            intel_wins_ppl += 1
        if intel['acc'] > std['acc']:
            intel_wins_acc += 1

        print(f"{ratio:<8.0%} {ppl_diff:>+12.1f}% {acc_diff:>+12.1f}% {winner:>12}")

    print(f"\nIntelligent wins on PPL: {intel_wins_ppl}/{len(ratios)-1}")
    print(f"Intelligent wins on Accuracy: {intel_wins_acc}/{len(ratios)-1}")

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find where biggest accuracy gap occurs
    max_acc_gap = 0
    max_gap_ratio = None
    for ratio in ratios:
        if ratio == 1.0:
            continue
        gap = results['intelligent'][ratio]['acc'] - results['standard'][ratio]['acc']
        if gap > max_acc_gap:
            max_acc_gap = gap
            max_gap_ratio = ratio

    if max_gap_ratio:
        print(f"\nBiggest accuracy gap at {max_gap_ratio:.0%} compression:")
        print(f"  Standard:    {results['standard'][max_gap_ratio]['acc']:.1%}")
        print(f"  Intelligent: {results['intelligent'][max_gap_ratio]['acc']:.1%}")
        print(f"  Difference:  {max_acc_gap*100:.1f}% better with intelligent compression")

    # Check PPL vs accuracy correlation
    print("\nPPL vs Accuracy correlation:")
    for method in ['standard', 'intelligent']:
        ppls = [results[method][r]['ppl'] for r in ratios]
        accs = [results[method][r]['acc'] for r in ratios]
        # Simple correlation check
        corr = sum((p - sum(ppls)/len(ppls)) * (a - sum(accs)/len(accs))
                   for p, a in zip(ppls, accs))
        print(f"  {method}: {'negatively correlated' if corr < 0 else 'positively correlated'}")

if __name__ == "__main__":
    run()
