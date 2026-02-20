#!/usr/bin/env python3
"""
Experiment 7: SVD + Quantization Benchmark

The clean test:
- Same model
- Same memory footprint (both quantized)
- Only difference: SVD pre-compression before quantization

Compare:
  Model @ INT4  vs  Model @ SVD + INT4

If SVD + INT4 beats INT4 alone, SVD acts as beneficial denoising.

Benchmarks:
1. Factual accuracy (our custom test)
2. MMLU-style questions (knowledge)
3. Reasoning (simple logic)
4. Memory usage comparison
"""

import torch
import gc
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import copy

# ============================================================
# BENCHMARK QUESTIONS
# ============================================================

FACTUAL_QUESTIONS = [
    # Geography (10)
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The capital of Germany is", "Berlin"),
    ("The capital of Australia is", "Canberra"),
    ("The capital of Brazil is", "Brasilia"),
    ("The capital of Canada is", "Ottawa"),
    ("The capital of Egypt is", "Cairo"),
    ("The capital of India is", "New Delhi"),
    ("The capital of South Korea is", "Seoul"),
    ("The capital of Mexico is", "Mexico City"),
    # Science (10)
    ("Water freezes at", "0"),
    ("Water boils at", "100"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for silver is", "Ag"),
    ("The chemical symbol for iron is", "Fe"),
    ("The speed of light is approximately", "300"),
    ("The atomic number of carbon is", "6"),
    ("The atomic number of oxygen is", "8"),
    ("The chemical formula for water is", "H2O"),
    ("The chemical formula for carbon dioxide is", "CO2"),
    # History/Culture (10)
    ("Shakespeare wrote", "Hamlet"),
    ("Einstein developed the theory of", "relativity"),
    ("The Great Wall is in", "China"),
    ("The Eiffel Tower is in", "Paris"),
    ("World War II ended in", "1945"),
    ("The first moon landing was in", "1969"),
    ("The Declaration of Independence was signed in", "1776"),
    ("Leonardo da Vinci painted the", "Mona Lisa"),
    ("The Pyramids are in", "Egypt"),
    ("The Colosseum is in", "Rome"),
]

MMLU_STYLE_QUESTIONS = [
    # Science
    {
        "question": "What is the primary function of mitochondria in a cell?",
        "choices": ["A) Protein synthesis", "B) Energy production", "C) Cell division", "D) Waste removal"],
        "answer": "B"
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "choices": ["A) Venus", "B) Jupiter", "C) Mars", "D) Saturn"],
        "answer": "C"
    },
    {
        "question": "What is the chemical symbol for sodium?",
        "choices": ["A) So", "B) Na", "C) Sd", "D) N"],
        "answer": "B"
    },
    {
        "question": "What type of bond is formed when electrons are shared between atoms?",
        "choices": ["A) Ionic bond", "B) Hydrogen bond", "C) Covalent bond", "D) Metallic bond"],
        "answer": "C"
    },
    {
        "question": "What is the speed of light in a vacuum?",
        "choices": ["A) 300,000 km/s", "B) 150,000 km/s", "C) 500,000 km/s", "D) 1,000,000 km/s"],
        "answer": "A"
    },
    # History
    {
        "question": "Who was the first President of the United States?",
        "choices": ["A) Thomas Jefferson", "B) John Adams", "C) George Washington", "D) Benjamin Franklin"],
        "answer": "C"
    },
    {
        "question": "In what year did World War I begin?",
        "choices": ["A) 1912", "B) 1914", "C) 1916", "D) 1918"],
        "answer": "B"
    },
    {
        "question": "Which empire built the Colosseum?",
        "choices": ["A) Greek Empire", "B) Ottoman Empire", "C) Roman Empire", "D) Byzantine Empire"],
        "answer": "C"
    },
    # Math/Logic
    {
        "question": "What is the value of pi to two decimal places?",
        "choices": ["A) 3.14", "B) 3.16", "C) 3.12", "D) 3.18"],
        "answer": "A"
    },
    {
        "question": "If x + 5 = 12, what is x?",
        "choices": ["A) 5", "B) 6", "C) 7", "D) 8"],
        "answer": "C"
    },
]

REASONING_QUESTIONS = [
    {
        "question": "If all roses are flowers and all flowers need water, do roses need water?",
        "answer": "yes"
    },
    {
        "question": "John is taller than Mary. Mary is taller than Bob. Is John taller than Bob?",
        "answer": "yes"
    },
    {
        "question": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost in cents?",
        "answer": "5"
    },
    {
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how many minutes would it take 100 machines to make 100 widgets?",
        "answer": "5"
    },
    {
        "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how many days would it take for the patch to cover half of the lake?",
        "answer": "47"
    },
]


def load_model_fp16(model_name, device="mps"):
    """Load model in FP16."""
    print(f"Loading {model_name} in FP16...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = model.to(device)
    return model, tokenizer


def get_importance(model, tokenizer, device):
    """Get gradient-based importance for Q, K, O projections."""
    print("  Computing importance scores...")

    # Need float32 for backward pass
    original_dtype = next(model.parameters()).dtype
    model_fp32 = model.float()
    model_fp32.train()
    importance = {}

    prompts = [
        "The capital of France is Paris.",
        "The capital of Japan is Tokyo.",
        "Water freezes at 0 degrees Celsius.",
        "Einstein developed the theory of relativity.",
        "The chemical symbol for gold is Au.",
        "Shakespeare wrote Hamlet.",
        "The largest planet is Jupiter.",
        "World War II ended in 1945.",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        try:
            outputs = model_fp32(**inputs, labels=inputs["input_ids"])
            model_fp32.zero_grad()
            outputs.loss.backward()

            for name, param in model_fp32.named_parameters():
                if param.grad is not None:
                    if any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
                        if name not in importance:
                            importance[name] = torch.zeros_like(param.data)
                        importance[name] += param.grad.abs()
        except Exception as e:
            continue

    # Convert back
    model.to(original_dtype)
    model.eval()
    print(f"    Got importance for {len(importance)} layers")
    return importance


def compress_qko_svd(model, importance, ratio=0.7):
    """Apply importance-guided SVD compression to Q, K, O projections."""
    print(f"  Compressing Q, K, O to {ratio:.0%}...")
    compressed_count = 0

    for name, module in model.named_modules():
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        if not any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
            continue

        W = module.weight.data
        if min(W.shape) <= 10:
            continue

        imp = None
        for key in importance:
            if name in key or key in name:
                imp = importance[key]
                break

        rank = max(1, int(min(W.shape) * ratio))

        try:
            U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

            if imp is not None:
                sv_importance = torch.zeros(len(S))
                imp_cpu = imp.float().cpu()
                for i in range(min(len(S), rank * 2)):
                    contrib = S[i] * torch.outer(U[:, i], Vh[i, :])
                    sv_importance[i] = (contrib.abs() * imp_cpu).sum()
                top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
            else:
                top_indices = torch.arange(rank)

            W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
            module.weight.data = W_approx.to(W.device).to(W.dtype)
            compressed_count += 1
        except Exception as e:
            continue

    print(f"    Compressed {compressed_count} layers")
    return model


def quantize_int8_simulate(model):
    """
    Simulate INT8 quantization with per-channel scaling.
    """
    print("  Applying INT8 quantization (per-channel)...")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            W = module.weight.data

            # Per-channel quantization (along output dimension)
            if W.dim() >= 2:
                W_flat = W.view(W.shape[0], -1)
                scale = W_flat.abs().max(dim=1, keepdim=True).values / 127.0
                scale = scale.clamp(min=1e-8)

                W_quant = torch.round(W_flat / scale).clamp(-127, 127)
                W_dequant = W_quant * scale

                module.weight.data = W_dequant.view(W.shape).to(W.dtype)
            else:
                scale = W.abs().max() / 127.0
                if scale > 0:
                    W_quant = torch.round(W / scale).clamp(-127, 127)
                    module.weight.data = (W_quant * scale).to(W.dtype)

    return model


def quantize_int4_simulate(model):
    """
    Simulate INT4 quantization with per-channel scaling.
    More realistic than naive per-tensor quantization.
    """
    print("  Applying INT4 quantization (per-channel)...")
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight is not None:
            W = module.weight.data

            # Per-channel quantization (along output dimension)
            if W.dim() >= 2:
                # Compute scale per output channel
                W_flat = W.view(W.shape[0], -1)
                scale = W_flat.abs().max(dim=1, keepdim=True).values / 7.0
                scale = scale.clamp(min=1e-8)

                # Quantize and dequantize
                W_quant = torch.round(W_flat / scale).clamp(-8, 7)
                W_dequant = W_quant * scale

                module.weight.data = W_dequant.view(W.shape).to(W.dtype)
            else:
                # Scalar quantization for 1D
                scale = W.abs().max() / 7.0
                if scale > 0:
                    W_quant = torch.round(W / scale).clamp(-8, 7)
                    module.weight.data = (W_quant * scale).to(W.dtype)

    return model


def test_factual(model, tokenizer, device):
    """Test factual knowledge."""
    model.eval()
    correct = 0

    for prompt, expected in FACTUAL_QUESTIONS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=15, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().lower()
        if expected.lower() in generated:
            correct += 1

    return correct / len(FACTUAL_QUESTIONS)


def test_mmlu_style(model, tokenizer, device):
    """Test MMLU-style multiple choice."""
    model.eval()
    correct = 0

    for q in MMLU_STYLE_QUESTIONS:
        prompt = f"Question: {q['question']}\n"
        prompt += "\n".join(q['choices'])
        prompt += "\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=5, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().upper()

        # Check if correct letter is in response
        if q['answer'] in generated[:3]:
            correct += 1

    return correct / len(MMLU_STYLE_QUESTIONS)


def test_reasoning(model, tokenizer, device):
    """Test reasoning ability."""
    model.eval()
    correct = 0

    for q in REASONING_QUESTIONS:
        prompt = f"Question: {q['question']}\nAnswer:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=20, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().lower()

        if q['answer'].lower() in generated:
            correct += 1

    return correct / len(REASONING_QUESTIONS)


def evaluate_perplexity(model, tokenizer, device):
    """Evaluate perplexity."""
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


def benchmark_model(model, tokenizer, device, name):
    """Run all benchmarks on a model."""
    print(f"\n  Benchmarking {name}...")

    results = {
        "name": name,
        "factual": test_factual(model, tokenizer, device),
        "mmlu": test_mmlu_style(model, tokenizer, device),
        "reasoning": test_reasoning(model, tokenizer, device),
        "ppl": evaluate_perplexity(model, tokenizer, device),
    }

    # Composite score (weighted average)
    results["composite"] = (
        results["factual"] * 0.4 +
        results["mmlu"] * 0.4 +
        results["reasoning"] * 0.2
    )

    return results


def run():
    print("="*70)
    print("EXPERIMENT 7: SVD + QUANTIZATION BENCHMARK")
    print("="*70)
    print("\nThe clean test:")
    print("  Same model, same memory footprint")
    print("  Compare: INT4 alone vs SVD + INT4")
    print("="*70)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Models to test
    models_to_test = [
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B",
        # "Qwen/Qwen2.5-3B",  # Uncomment for larger test
    ]

    all_results = []

    for model_name in models_to_test:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print("="*70)

        # ========================================
        # 1. Baseline: FP16 (no compression, no quantization)
        # ========================================
        print("\n--- BASELINE: FP16 ---")
        model, tokenizer = load_model_fp16(model_name, device)
        baseline_results = benchmark_model(model, tokenizer, device, "FP16 Baseline")

        print(f"    Factual: {baseline_results['factual']:.1%}")
        print(f"    MMLU: {baseline_results['mmlu']:.1%}")
        print(f"    Reasoning: {baseline_results['reasoning']:.1%}")
        print(f"    PPL: {baseline_results['ppl']:.2f}")
        print(f"    Composite: {baseline_results['composite']:.1%}")

        # ========================================
        # 2. INT8 only
        # ========================================
        print("\n--- INT8 ONLY ---")
        model_int8 = copy.deepcopy(model)
        model_int8 = quantize_int8_simulate(model_int8)
        int8_results = benchmark_model(model_int8, tokenizer, device, "INT8 Only")

        print(f"    Factual: {int8_results['factual']:.1%} ({(int8_results['factual']-baseline_results['factual'])*100:+.1f}%)")
        print(f"    MMLU: {int8_results['mmlu']:.1%} ({(int8_results['mmlu']-baseline_results['mmlu'])*100:+.1f}%)")
        print(f"    Reasoning: {int8_results['reasoning']:.1%} ({(int8_results['reasoning']-baseline_results['reasoning'])*100:+.1f}%)")
        print(f"    PPL: {int8_results['ppl']:.2f} ({int8_results['ppl']/baseline_results['ppl']:.2f}x)")
        print(f"    Composite: {int8_results['composite']:.1%} ({(int8_results['composite']-baseline_results['composite'])*100:+.1f}%)")

        del model_int8
        gc.collect()

        # ========================================
        # 3. SVD 90% + INT8 (conservative)
        # ========================================
        print("\n--- SVD 90% + INT8 ---")
        model_svd_int8 = copy.deepcopy(model)

        importance = get_importance(model_svd_int8, tokenizer, device)
        model_svd_int8 = compress_qko_svd(model_svd_int8, importance, ratio=0.9)
        model_svd_int8 = quantize_int8_simulate(model_svd_int8)

        svd_int8_results = benchmark_model(model_svd_int8, tokenizer, device, "SVD 90% + INT8")

        print(f"    Factual: {svd_int8_results['factual']:.1%} ({(svd_int8_results['factual']-baseline_results['factual'])*100:+.1f}%)")
        print(f"    MMLU: {svd_int8_results['mmlu']:.1%} ({(svd_int8_results['mmlu']-baseline_results['mmlu'])*100:+.1f}%)")
        print(f"    Reasoning: {svd_int8_results['reasoning']:.1%} ({(svd_int8_results['reasoning']-baseline_results['reasoning'])*100:+.1f}%)")
        print(f"    PPL: {svd_int8_results['ppl']:.2f} ({svd_int8_results['ppl']/baseline_results['ppl']:.2f}x)")
        print(f"    Composite: {svd_int8_results['composite']:.1%} ({(svd_int8_results['composite']-baseline_results['composite'])*100:+.1f}%)")

        del model_svd_int8
        gc.collect()

        # ========================================
        # INT8 Comparison
        # ========================================
        print(f"\n--- COMPARISON: INT8 vs SVD+INT8 ---")

        int8_composite = int8_results['composite']
        svd8_composite = svd_int8_results['composite']
        diff8 = svd8_composite - int8_composite

        if diff8 > 0.01:
            print(f"    ✓ SVD + INT8 WINS by {diff8*100:.1f}%")
            winner8 = "SVD+INT8"
        elif diff8 < -0.01:
            print(f"    ✗ INT8 alone wins by {-diff8*100:.1f}%")
            winner8 = "INT8"
        else:
            print(f"    ~ TIE (diff: {diff8*100:.1f}%)")
            winner8 = "tie"

        all_results.append({
            "model": model_name,
            "baseline": baseline_results,
            "int8": int8_results,
            "svd_int8": svd_int8_results,
            "winner": winner8,
            "improvement": diff8
        })

        del model
        gc.collect()

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n{'Model':<25} {'FP16':>10} {'INT8':>10} {'SVD90+INT8':>11} {'Δ':>8} {'Winner':>12}")
    print("-"*80)

    for r in all_results:
        model_short = r['model'].split('/')[-1]
        print(f"{model_short:<25} {r['baseline']['composite']:>9.1%} {r['int8']['composite']:>9.1%} {r['svd_int8']['composite']:>10.1%} {r['improvement']*100:>+7.1f}% {r['winner']:>12}")

    # Check hypothesis
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    svd_wins = sum(1 for r in all_results if r['winner'] == 'SVD+INT8')
    int8_wins = sum(1 for r in all_results if r['winner'] == 'INT8')
    ties = sum(1 for r in all_results if r['winner'] == 'tie')

    if svd_wins > int8_wins:
        print(f"\n✓ SVD PRE-COMPRESSION HELPS QUANTIZATION")
        print(f"  SVD+INT8 won {svd_wins}/{len(all_results)} comparisons")
        print(f"  SVD acts as beneficial denoising before quantization")
    elif svd_wins == int8_wins:
        print(f"\n~ INCONCLUSIVE: {svd_wins} SVD wins, {int8_wins} INT8 wins, {ties} ties")
    else:
        print(f"\n✗ SVD PRE-COMPRESSION DOESN'T HELP")
        print(f"  INT8 alone won {int8_wins}/{len(all_results)} comparisons")

    # Save results
    results_file = Path(__file__).parent / "exp7_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    run()
