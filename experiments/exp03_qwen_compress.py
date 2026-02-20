#!/usr/bin/env python3
"""
Experiment 3: Real LLM Validation (Qwen2.5-0.5B)

Test importance-guided compression on a real pretrained LLM.
Compare factual accuracy before and after compression.

Key questions:
1. Does importance-guided compression preserve facts better than standard SVD?
2. How much can we compress before significant degradation?
3. Is PPL a good proxy for factual accuracy?
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
    """Questions to test factual knowledge."""
    questions = [
        # Geography
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        # Science
        ("Water freezes at", "0"),
        ("Water boils at", "100"),
        ("The chemical symbol for gold is", "Au"),
        ("The chemical symbol for silver is", "Ag"),
        ("The speed of light is approximately", "300"),
        # General knowledge
        ("The largest planet in our solar system is", "Jupiter"),
        ("The smallest planet in our solar system is", "Mercury"),
        ("Shakespeare wrote", "Hamlet"),
        ("Einstein developed the theory of", "relativity"),
        ("The Great Wall is in", "China"),
    ]
    return questions

def test_factual_accuracy(model, tokenizer, questions, device, verbose=False):
    """Test model on factual questions."""
    model.eval()
    correct = 0
    total = len(questions)

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

        match = expected.lower() in generated
        if match:
            correct += 1

        if verbose:
            mark = "✓" if match else "✗"
            print(f"  {mark} '{prompt}' -> '{generated[:30]}' (want '{expected}')")

    return correct / total

def evaluate_ppl(model, tokenizer, device, texts=None):
    """Evaluate perplexity on sample texts."""
    if texts is None:
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning, there was light and darkness.",
            "Science is the pursuit of knowledge through observation.",
            "Mathematics is the language of the universe.",
            "History teaches us about the past and guides our future.",
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

def get_importance_for_attention(model, tokenizer, device):
    """Get gradient-based importance for attention projections."""
    model.train()
    importance = {}

    # Use factual prompts to compute importance
    prompts = [
        "The capital of France is Paris.",
        "The capital of Japan is Tokyo.",
        "Water boils at 100 degrees.",
        "The largest planet is Jupiter.",
        "Shakespeare wrote Hamlet.",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss

        model.zero_grad()
        loss.backward()

        # Collect importance for attention layers
        for name, param in model.named_parameters():
            if any(x in name for x in ['q_proj', 'k_proj', 'o_proj']) and param.grad is not None:
                if name not in importance:
                    importance[name] = torch.zeros_like(param.data)
                importance[name] += param.grad.abs()

    model.eval()
    return importance

def compress_attention_standard(model, ratio):
    """Compress attention projections with standard SVD."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
            W = module.weight.data
            if min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))
            U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
            W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)

def compress_attention_intelligent(model, importance, ratio):
    """Compress attention projections with importance-guided SVD."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
            W = module.weight.data

            # Find matching importance
            imp_name = None
            for key in importance:
                if name in key or key in name:
                    imp_name = key
                    break

            if imp_name is None or min(W.shape) <= 10:
                continue

            imp = importance[imp_name]
            rank = max(1, int(min(W.shape) * ratio))

            U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

            # Score singular values by importance
            sv_importance = torch.zeros(len(S))
            imp_cpu = imp.float().cpu()

            for i in range(min(len(S), rank * 2)):
                contrib = S[i] * torch.outer(U[:, i], Vh[i, :])
                sv_importance[i] = (contrib.abs() * imp_cpu).sum()

            top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
            W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)

def run():
    print("="*70)
    print("EXPERIMENT 3: QWEN2.5-0.5B COMPRESSION")
    print("="*70)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, tokenizer = load_model()
    model = model.to(device)

    questions = get_factual_questions()

    # Baseline
    print("\n" + "="*70)
    print("BASELINE (no compression)")
    print("="*70)

    base_ppl = evaluate_ppl(model, tokenizer, device)
    base_acc = test_factual_accuracy(model, tokenizer, questions, device, verbose=True)

    print(f"\nPPL: {base_ppl:.2f}")
    print(f"Factual accuracy: {base_acc:.1%}")

    # Get importance
    print("\n" + "="*70)
    print("Computing importance...")
    print("="*70)
    importance = get_importance_for_attention(model, tokenizer, device)
    print(f"Got importance for {len(importance)} parameters")

    # Test compression ratios
    ratios = [0.9, 0.8, 0.7, 0.6, 0.5]

    results = {'baseline': {'ppl': base_ppl, 'acc': base_acc}}

    print("\n" + "="*70)
    print("COMPRESSION COMPARISON")
    print("="*70)

    print(f"\n{'Ratio':<8} {'Method':<12} {'PPL':>8} {'Acc':>8} {'PPL Δ':>10} {'Acc Δ':>10}")
    print("-"*60)

    for ratio in ratios:
        # Standard SVD
        model_std = copy.deepcopy(model)
        compress_attention_standard(model_std, ratio)

        ppl_std = evaluate_ppl(model_std, tokenizer, device)
        acc_std = test_factual_accuracy(model_std, tokenizer, questions, device)

        ppl_delta = (ppl_std / base_ppl - 1) * 100
        acc_delta = (acc_std - base_acc) * 100

        print(f"{ratio:<8.0%} {'standard':<12} {ppl_std:>8.2f} {acc_std:>8.1%} {ppl_delta:>+10.1f}% {acc_delta:>+10.1f}%")
        results[f'std_{ratio}'] = {'ppl': ppl_std, 'acc': acc_std}

        del model_std

        # Intelligent SVD
        model_int = copy.deepcopy(model)
        compress_attention_intelligent(model_int, importance, ratio)

        ppl_int = evaluate_ppl(model_int, tokenizer, device)
        acc_int = test_factual_accuracy(model_int, tokenizer, questions, device)

        ppl_delta = (ppl_int / base_ppl - 1) * 100
        acc_delta = (acc_int - base_acc) * 100

        print(f"{ratio:<8.0%} {'intelligent':<12} {ppl_int:>8.2f} {acc_int:>8.1%} {ppl_delta:>+10.1f}% {acc_delta:>+10.1f}%")
        results[f'int_{ratio}'] = {'ppl': ppl_int, 'acc': acc_int}

        del model_int
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    print("\nIntelligent vs Standard at each compression level:")
    for ratio in ratios:
        std = results[f'std_{ratio}']
        intel = results[f'int_{ratio}']

        ppl_better = "intelligent" if intel['ppl'] < std['ppl'] else "standard"
        acc_better = "intelligent" if intel['acc'] > std['acc'] else "standard"

        print(f"  {ratio:.0%}: PPL winner={ppl_better}, Acc winner={acc_better}")

    # Key insight
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find where intelligent beats standard most
    max_acc_advantage = 0
    best_ratio = None
    for ratio in ratios:
        std_acc = results[f'std_{ratio}']['acc']
        int_acc = results[f'int_{ratio}']['acc']
        if int_acc > std_acc and (int_acc - std_acc) > max_acc_advantage:
            max_acc_advantage = int_acc - std_acc
            best_ratio = ratio

    if best_ratio:
        print(f"\nBiggest intelligent advantage at {best_ratio:.0%}:")
        print(f"  Standard: {results[f'std_{best_ratio}']['acc']:.1%}")
        print(f"  Intelligent: {results[f'int_{best_ratio}']['acc']:.1%}")
        print(f"  Difference: {max_acc_advantage*100:.1f}%")
    else:
        print("\nIntelligent compression didn't consistently beat standard")

if __name__ == "__main__":
    run()
