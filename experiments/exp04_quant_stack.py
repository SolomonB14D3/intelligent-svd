#!/usr/bin/env python3
"""
Experiment 4: Quantization Stacking

Test the full compression pipeline:
1. Intelligent SVD compression (reduce rank)
2. + INT8/INT4 quantization (reduce precision)

Compare total compression ratio vs quality.

Pipeline options:
A) Original → Quantize only
B) Original → SVD compress → Quantize
C) Original → Intelligent SVD → Quantize
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import copy

def load_model_fp32():
    """Load Qwen2.5-0.5B in FP32."""
    print("Loading Qwen2.5-0.5B (FP32)...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    return model, tokenizer

def load_model_int8():
    """Load Qwen2.5-0.5B with INT8 quantization."""
    print("Loading Qwen2.5-0.5B (INT8)...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    return model, tokenizer

def get_factual_questions():
    """Questions for testing."""
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
    ]

def test_factual_accuracy(model, tokenizer, questions, device):
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

def get_model_size_mb(model):
    """Get model size in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    # Assume float32 = 4 bytes per param
    return (total_params * 4) / (1024 * 1024)

def get_importance(model, tokenizer, device):
    """Get importance for attention projections."""
    model.train()
    importance = {}

    prompts = [
        "The capital of France is Paris.",
        "Water freezes at 0 degrees.",
        "Einstein developed the theory of relativity.",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])

        model.zero_grad()
        outputs.loss.backward()

        for name, param in model.named_parameters():
            if any(x in name for x in ['q_proj', 'k_proj', 'o_proj']) and param.grad is not None:
                if name not in importance:
                    importance[name] = torch.zeros_like(param.data)
                importance[name] += param.grad.abs()

    model.eval()
    return importance

def compress_standard(model, ratio):
    """Standard SVD compression."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
            W = module.weight.data
            if min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))
            U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
            W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)

def compress_intelligent(model, importance, ratio):
    """Importance-guided SVD compression."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
            W = module.weight.data

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

            sv_importance = torch.zeros(len(S))
            imp_cpu = imp.float().cpu()

            for i in range(min(len(S), rank * 2)):
                contrib = S[i] * torch.outer(U[:, i], Vh[i, :])
                sv_importance[i] = (contrib.abs() * imp_cpu).sum()

            top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
            W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)

def quantize_to_int8_manual(model):
    """Simple INT8-like quantization (scale to int range)."""
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            W = module.weight.data
            # Simple quantization: scale to [-127, 127] and back
            scale = W.abs().max() / 127
            W_quant = torch.round(W / scale).clamp(-127, 127)
            module.weight.data = (W_quant * scale).to(W.dtype)

def run():
    print("="*70)
    print("EXPERIMENT 4: QUANTIZATION STACKING")
    print("="*70)
    print("Test: SVD compression + quantization pipeline")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    questions = get_factual_questions()

    results = {}

    # ========================================
    # Baseline: FP32
    # ========================================
    print("\n" + "="*70)
    print("BASELINE: FP32 (no compression)")
    print("="*70)

    model_fp32, tokenizer = load_model_fp32()
    model_fp32 = model_fp32.to(device)

    base_ppl = evaluate_ppl(model_fp32, tokenizer, device)
    base_acc = test_factual_accuracy(model_fp32, tokenizer, questions, device)
    base_size = get_model_size_mb(model_fp32)

    print(f"Size: {base_size:.1f} MB")
    print(f"PPL: {base_ppl:.2f}")
    print(f"Accuracy: {base_acc:.1%}")

    results['fp32_baseline'] = {'ppl': base_ppl, 'acc': base_acc, 'size': base_size}

    # Get importance
    print("\nComputing importance...")
    importance = get_importance(model_fp32, tokenizer, device)

    # ========================================
    # Option A: Quantize only (INT8-like)
    # ========================================
    print("\n" + "="*70)
    print("OPTION A: Quantize only (INT8)")
    print("="*70)

    model_quant = copy.deepcopy(model_fp32)
    quantize_to_int8_manual(model_quant)

    ppl = evaluate_ppl(model_quant, tokenizer, device)
    acc = test_factual_accuracy(model_quant, tokenizer, questions, device)

    print(f"PPL: {ppl:.2f} ({(ppl/base_ppl-1)*100:+.1f}%)")
    print(f"Accuracy: {acc:.1%} ({(acc-base_acc)*100:+.1f}%)")

    results['int8_only'] = {'ppl': ppl, 'acc': acc}
    del model_quant

    # ========================================
    # Option B: Standard SVD (70%) + Quantize
    # ========================================
    print("\n" + "="*70)
    print("OPTION B: Standard SVD (70%) → INT8")
    print("="*70)

    model_svd_quant = copy.deepcopy(model_fp32)
    compress_standard(model_svd_quant, 0.7)

    ppl_after_svd = evaluate_ppl(model_svd_quant, tokenizer, device)
    acc_after_svd = test_factual_accuracy(model_svd_quant, tokenizer, questions, device)
    print(f"After SVD 70%: PPL={ppl_after_svd:.2f}, Acc={acc_after_svd:.1%}")

    quantize_to_int8_manual(model_svd_quant)

    ppl = evaluate_ppl(model_svd_quant, tokenizer, device)
    acc = test_factual_accuracy(model_svd_quant, tokenizer, questions, device)

    print(f"After INT8: PPL={ppl:.2f} ({(ppl/base_ppl-1)*100:+.1f}%), Acc={acc:.1%} ({(acc-base_acc)*100:+.1f}%)")

    results['svd70_int8'] = {'ppl': ppl, 'acc': acc}
    del model_svd_quant

    # ========================================
    # Option C: Intelligent SVD (70%) + Quantize
    # ========================================
    print("\n" + "="*70)
    print("OPTION C: Intelligent SVD (70%) → INT8")
    print("="*70)

    model_int_quant = copy.deepcopy(model_fp32)
    compress_intelligent(model_int_quant, importance, 0.7)

    ppl_after_svd = evaluate_ppl(model_int_quant, tokenizer, device)
    acc_after_svd = test_factual_accuracy(model_int_quant, tokenizer, questions, device)
    print(f"After intelligent SVD 70%: PPL={ppl_after_svd:.2f}, Acc={acc_after_svd:.1%}")

    quantize_to_int8_manual(model_int_quant)

    ppl = evaluate_ppl(model_int_quant, tokenizer, device)
    acc = test_factual_accuracy(model_int_quant, tokenizer, questions, device)

    print(f"After INT8: PPL={ppl:.2f} ({(ppl/base_ppl-1)*100:+.1f}%), Acc={acc:.1%} ({(acc-base_acc)*100:+.1f}%)")

    results['intelligent70_int8'] = {'ppl': ppl, 'acc': acc}
    del model_int_quant

    # ========================================
    # Option D: More aggressive - 50% + Quantize
    # ========================================
    print("\n" + "="*70)
    print("OPTION D: Intelligent SVD (50%) → INT8")
    print("="*70)

    model_agg = copy.deepcopy(model_fp32)
    compress_intelligent(model_agg, importance, 0.5)

    ppl_after_svd = evaluate_ppl(model_agg, tokenizer, device)
    acc_after_svd = test_factual_accuracy(model_agg, tokenizer, questions, device)
    print(f"After intelligent SVD 50%: PPL={ppl_after_svd:.2f}, Acc={acc_after_svd:.1%}")

    quantize_to_int8_manual(model_agg)

    ppl = evaluate_ppl(model_agg, tokenizer, device)
    acc = test_factual_accuracy(model_agg, tokenizer, questions, device)

    print(f"After INT8: PPL={ppl:.2f} ({(ppl/base_ppl-1)*100:+.1f}%), Acc={acc:.1%} ({(acc-base_acc)*100:+.1f}%)")

    results['intelligent50_int8'] = {'ppl': ppl, 'acc': acc}

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n{'Pipeline':<30} {'PPL':>8} {'PPL Δ':>10} {'Acc':>8} {'Acc Δ':>10}")
    print("-"*70)

    for name, r in results.items():
        ppl_delta = (r['ppl'] / base_ppl - 1) * 100
        acc_delta = (r['acc'] - base_acc) * 100
        print(f"{name:<30} {r['ppl']:>8.2f} {ppl_delta:>+10.1f}% {r['acc']:>8.1%} {acc_delta:>+10.1f}%")

    # Analysis
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find best pipeline
    best = max(results.items(), key=lambda x: x[1]['acc'])
    print(f"\nBest accuracy: {best[0]} ({best[1]['acc']:.1%})")

    # Check if intelligent + quant beats standard + quant
    if 'intelligent70_int8' in results and 'svd70_int8' in results:
        int_acc = results['intelligent70_int8']['acc']
        std_acc = results['svd70_int8']['acc']
        if int_acc > std_acc:
            print(f"\n✓ Intelligent SVD + INT8 beats Standard SVD + INT8")
            print(f"  {int_acc:.1%} vs {std_acc:.1%} ({(int_acc-std_acc)*100:+.1f}%)")

if __name__ == "__main__":
    run()
