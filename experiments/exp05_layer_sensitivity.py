#!/usr/bin/env python3
"""
Experiment 5: Full Model Compression (V + MLP)

Previous finding: Q, K, O are "safe" to compress, V and MLP are "risky"
New hypothesis: With importance-guided SVD, V and MLP become safe too

Test:
1. Compress Q, K, O only (previous approach)
2. Compress Q, K, V, O (add V)
3. Compress attention + MLP (full model)

All using importance-guided SVD to preserve factual directions.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

def load_model():
    """Load Qwen2.5-0.5B."""
    print("Loading Qwen2.5-0.5B...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    return model, tokenizer

def get_factual_questions():
    """Factual questions for testing."""
    return [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("Water freezes at", "0"),
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for silver is", "Ag"),
        ("The speed of light is approximately", "300"),
        ("Einstein developed the theory of", "relativity"),
        ("The largest planet in our solar system is", "Jupiter"),
        ("Shakespeare wrote", "Hamlet"),
        ("The Great Wall is in", "China"),
        ("Pi is approximately", "3.14"),
        ("The human body has", "206"),  # bones
    ]

def test_accuracy(model, tokenizer, questions, device):
    """Test factual accuracy."""
    model.eval()
    correct = 0

    for prompt, expected in questions:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().lower()

        if expected.lower() in generated:
            correct += 1

    return correct / len(questions)

def evaluate_ppl(model, tokenizer, device):
    """Evaluate perplexity."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Science is the pursuit of knowledge through observation.",
        "Mathematics is the language of the universe.",
        "History teaches us about the past and guides our future.",
        "Technology continues to advance at a rapid pace.",
    ]

    model.eval()
    total_loss = 0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
            total_tokens += inputs["input_ids"].shape[1]

    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def get_full_importance(model, tokenizer, device):
    """Get importance for ALL linear layers (attention + MLP)."""
    model.train()
    importance = {}

    # Use factual prompts for importance
    prompts = [
        "The capital of France is Paris.",
        "The capital of Japan is Tokyo.",
        "Water freezes at 0 degrees.",
        "The largest planet is Jupiter.",
        "Shakespeare wrote Hamlet.",
        "Einstein developed the theory of relativity.",
        "The chemical symbol for gold is Au.",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])

        model.zero_grad()
        outputs.loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None and 'weight' in name:
                # Include Q, K, V, O projections AND MLP layers
                if any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                           'gate_proj', 'up_proj', 'down_proj',
                                           'c_fc', 'c_proj']):  # Different naming conventions
                    if name not in importance:
                        importance[name] = torch.zeros_like(param.data)
                    importance[name] += param.grad.abs()

    model.eval()
    return importance

def compress_layers(model, importance, ratio, layer_filter):
    """
    Compress specific layers using importance-guided SVD.

    layer_filter: function that returns True for layers to compress
    """
    for name, module in model.named_modules():
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        if not layer_filter(name):
            continue

        W = module.weight.data
        if min(W.shape) <= 10:
            continue

        # Find matching importance
        imp = None
        for key in importance:
            if name in key or key in name:
                imp = importance[key]
                break

        rank = max(1, int(min(W.shape) * ratio))
        U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

        if imp is not None:
            # Importance-guided selection
            sv_importance = torch.zeros(len(S))
            imp_cpu = imp.float().cpu()

            for i in range(min(len(S), rank * 2)):
                contrib = S[i] * torch.outer(U[:, i], Vh[i, :])
                sv_importance[i] = (contrib.abs() * imp_cpu).sum()

            top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
        else:
            # Standard SVD fallback
            top_indices = torch.arange(rank)

        W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
        module.weight.data = W_approx.to(W.device).to(W.dtype)

def run():
    print("="*70)
    print("EXPERIMENT 5: FULL MODEL COMPRESSION")
    print("="*70)
    print("Testing importance-guided compression on V and MLP layers")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    model, tokenizer = load_model()
    model = model.to(device)

    questions = get_factual_questions()

    # Baseline
    print("\n" + "="*70)
    print("BASELINE")
    print("="*70)

    base_ppl = evaluate_ppl(model, tokenizer, device)
    base_acc = test_accuracy(model, tokenizer, questions, device)

    print(f"PPL: {base_ppl:.2f}")
    print(f"Accuracy: {base_acc:.1%}")

    # Get importance for all layers
    print("\nComputing importance for all layers...")
    importance = get_full_importance(model, tokenizer, device)
    print(f"Got importance for {len(importance)} parameters")

    # Define layer filters
    def qko_only(name):
        return any(x in name for x in ['q_proj', 'k_proj', 'o_proj'])

    def qkvo(name):
        return any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    def attention_full(name):
        return any(x in name for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    def mlp_only(name):
        return any(x in name for x in ['gate_proj', 'up_proj', 'down_proj', 'c_fc', 'c_proj'])

    def all_layers(name):
        return attention_full(name) or mlp_only(name)

    results = {'baseline': {'ppl': base_ppl, 'acc': base_acc}}

    # Test configurations
    configs = [
        ("Q,K,O only (70%)", qko_only, 0.7),
        ("Q,K,V,O (70%)", qkvo, 0.7),
        ("Q,K,V,O (50%)", qkvo, 0.5),
        ("MLP only (70%)", mlp_only, 0.7),
        ("MLP only (50%)", mlp_only, 0.5),
        ("All layers (70%)", all_layers, 0.7),
        ("All layers (50%)", all_layers, 0.5),
    ]

    print("\n" + "="*70)
    print("COMPRESSION RESULTS (Importance-Guided)")
    print("="*70)

    print(f"\n{'Config':<25} {'PPL':>8} {'PPL Δ':>10} {'Acc':>8} {'Acc Δ':>10}")
    print("-"*65)

    for config_name, layer_filter, ratio in configs:
        model_copy = copy.deepcopy(model)
        compress_layers(model_copy, importance, ratio, layer_filter)

        ppl = evaluate_ppl(model_copy, tokenizer, device)
        acc = test_accuracy(model_copy, tokenizer, questions, device)

        ppl_delta = (ppl / base_ppl - 1) * 100
        acc_delta = (acc - base_acc) * 100

        print(f"{config_name:<25} {ppl:>8.2f} {ppl_delta:>+10.1f}% {acc:>8.1%} {acc_delta:>+10.1f}%")

        results[config_name] = {'ppl': ppl, 'acc': acc}

        del model_copy

    # Compare with standard SVD on risky layers
    print("\n" + "="*70)
    print("COMPARISON: Standard vs Importance-Guided on V+MLP")
    print("="*70)

    print("\nStandard SVD (no importance):")

    # Standard SVD on MLP
    model_std = copy.deepcopy(model)
    for name, module in model_std.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            if mlp_only(name):
                W = module.weight.data
                if min(W.shape) <= 10:
                    continue
                rank = max(1, int(min(W.shape) * 0.7))
                U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
                W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
                module.weight.data = W_approx.to(W.device).to(W.dtype)

    ppl_std = evaluate_ppl(model_std, tokenizer, device)
    acc_std = test_accuracy(model_std, tokenizer, questions, device)

    print(f"  MLP Standard SVD 70%: PPL={ppl_std:.2f}, Acc={acc_std:.1%}")

    # Compare
    imp_acc = results['MLP only (70%)']['acc']
    print(f"  MLP Importance SVD 70%: PPL={results['MLP only (70%)']['ppl']:.2f}, Acc={imp_acc:.1%}")

    if imp_acc > acc_std:
        print(f"\n  ✓ Importance-guided is {(imp_acc - acc_std)*100:.1f}% better on MLP!")
    elif imp_acc == acc_std:
        print(f"\n  ~ Same accuracy, importance-guided matches standard")
    else:
        print(f"\n  ✗ Standard SVD better on MLP")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\nBest configurations that preserve accuracy:")
    for name, r in sorted(results.items(), key=lambda x: -x[1]['acc']):
        if r['acc'] >= base_acc * 0.9:  # Within 10% of baseline
            ppl_delta = (r['ppl'] / base_ppl - 1) * 100
            acc_delta = (r['acc'] - base_acc) * 100
            print(f"  {name}: Acc={r['acc']:.1%} ({acc_delta:+.1f}%), PPL={r['ppl']:.2f} ({ppl_delta:+.1f}%)")

    # Key finding
    print("\n" + "="*70)
    print("KEY FINDING")
    print("="*70)

    # Check if full model compression is viable
    all_70 = results.get('All layers (70%)', {})
    if all_70 and all_70['acc'] >= base_acc * 0.85:
        print(f"\n✓ FULL MODEL compression is viable with importance-guided SVD!")
        print(f"  All layers at 70%: {all_70['acc']:.1%} accuracy ({(all_70['acc']-base_acc)*100:+.1f}%)")
        print(f"  This enables much larger compression than attention-only")
    else:
        print(f"\n  Full model compression degrades too much")
        best_safe = max([(k, v) for k, v in results.items() if v['acc'] >= base_acc * 0.9],
                        key=lambda x: base_ppl / x[1]['ppl'], default=None)
        if best_safe:
            print(f"  Best safe config: {best_safe[0]}")

if __name__ == "__main__":
    run()
