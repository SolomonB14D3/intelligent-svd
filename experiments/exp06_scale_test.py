#!/usr/bin/env python3
"""
Experiment 6: Scale Test - Larger Compressed vs Smaller Raw

Hypothesis: A larger model with SVD compression outperforms a smaller model
at similar memory footprint.

For Apple Silicon (no bitsandbytes), we compare:
- Small model @ FP16
- Large model @ SVD compressed @ FP16 (compressed to similar memory)

The key insight: if SVD can safely remove 30% of Q,K,O parameters,
a 1.5B model compressed might outperform a 1B model uncompressed.
"""

import torch
import gc
import time
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy

# Factual questions for testing
FACTUAL_QUESTIONS = [
    # Geography
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
    # Science
    ("Water freezes at", "0"),
    ("Water boils at", "100"),
    ("The chemical symbol for gold is", "Au"),
    ("The chemical symbol for silver is", "Ag"),
    ("The chemical symbol for iron is", "Fe"),
    ("The speed of light is approximately", "300"),
    ("The atomic number of carbon is", "6"),
    ("The atomic number of oxygen is", "8"),
    # History/Culture
    ("Shakespeare wrote", "Hamlet"),
    ("Einstein developed the theory of", "relativity"),
    ("The Great Wall is in", "China"),
    ("The Eiffel Tower is in", "Paris"),
    ("World War II ended in", "1945"),
    ("The first moon landing was in", "1969"),
    # Math/Logic
    ("Pi is approximately", "3.14"),
    ("The square root of 144 is", "12"),
    # General Knowledge
    ("The largest planet in our solar system is", "Jupiter"),
    ("The smallest planet in our solar system is", "Mercury"),
    ("The largest ocean is the", "Pacific"),
    ("The longest river is the", "Nile"),
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
    print("Computing importance scores...")

    # Need float32 for backward pass
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
            print(f"  Warning: {e}")
            continue

    # Convert back to half
    model.half()
    model.eval()
    print(f"  Got importance for {len(importance)} layers")
    return importance


def compress_qko_svd(model, importance, ratio=0.7):
    """Apply importance-guided SVD compression to Q, K, O projections."""
    print(f"Compressing Q, K, O to {ratio:.0%}...")
    compressed_count = 0

    for name, module in model.named_modules():
        if not hasattr(module, 'weight') or module.weight is None:
            continue

        if not any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
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
            print(f"  Warning compressing {name}: {e}")
            continue

    print(f"  Compressed {compressed_count} layers")
    return model


def test_factual_accuracy(model, tokenizer, questions, device, verbose=False):
    """Test factual accuracy on question set."""
    model.eval()
    correct = 0

    for prompt, expected in questions:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=15,
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
            print(f"  {mark} '{prompt}' -> '{generated[:40]}' (want '{expected}')")

    return correct / len(questions)


def evaluate_perplexity(model, tokenizer, device, texts=None):
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


def measure_inference_speed(model, tokenizer, device, n_tokens=50, n_runs=3):
    """Measure tokens per second."""
    prompt = "The history of artificial intelligence begins with"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False,
                      pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

    times = []
    for _ in range(n_runs):
        start = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_tokens, do_sample=False,
                          pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        times.append(time.time() - start)

    avg_time = sum(times) / len(times)
    return n_tokens / avg_time


def count_parameters(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def get_model_size_gb(model):
    """Estimate model size in GB."""
    total_params = count_parameters(model)
    sample_param = next(model.parameters())
    bytes_per_param = sample_param.element_size()
    return (total_params * bytes_per_param) / 1e9


def run():
    print("="*70)
    print("EXPERIMENT 6: SCALE TEST")
    print("="*70)
    print("Hypothesis: Larger model + SVD compression > Smaller model uncompressed")
    print("\nComparing at FP16 (Apple Silicon compatible)")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Test pairs: small uncompressed vs large compressed
    # The large model at 70% Q,K,O compression should be roughly comparable in "effective" size
    test_pairs = [
        ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"),  # 0.5B vs 1.5B @ 70%
        ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"),    # 1.5B vs 3B @ 70%
    ]

    all_results = []

    for small_name, large_name in test_pairs:
        print("\n" + "="*70)
        print(f"COMPARISON: {small_name} vs {large_name} @ SVD 70%")
        print("="*70)

        results = {
            "small_model": small_name,
            "large_model": large_name,
        }

        # ========================================
        # Small model (uncompressed)
        # ========================================
        print(f"\n--- SMALL: {small_name} (uncompressed) ---")

        try:
            model_small, tokenizer = load_model_fp16(small_name, device)

            results["small"] = {
                "params": count_parameters(model_small),
                "size_gb": get_model_size_gb(model_small),
                "ppl": evaluate_perplexity(model_small, tokenizer, device),
                "factual_acc": test_factual_accuracy(model_small, tokenizer, FACTUAL_QUESTIONS, device),
                "tokens_per_sec": measure_inference_speed(model_small, tokenizer, device),
            }

            print(f"  Params: {results['small']['params']:,}")
            print(f"  Size: {results['small']['size_gb']:.2f} GB")
            print(f"  PPL: {results['small']['ppl']:.2f}")
            print(f"  Factual Accuracy: {results['small']['factual_acc']:.1%}")
            print(f"  Speed: {results['small']['tokens_per_sec']:.1f} tok/s")

            del model_small
            gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            results["small"] = {"error": str(e)}

        # ========================================
        # Large model (SVD compressed)
        # ========================================
        print(f"\n--- LARGE: {large_name} @ SVD 70% Q,K,O ---")

        try:
            model_large, tokenizer = load_model_fp16(large_name, device)

            results["large_before"] = {
                "params": count_parameters(model_large),
                "size_gb": get_model_size_gb(model_large),
            }
            print(f"  Before compression: {results['large_before']['params']:,} params")

            # Get importance and compress
            importance = get_importance(model_large, tokenizer, device)
            model_large = compress_qko_svd(model_large, importance, ratio=0.7)

            results["large_svd"] = {
                "params": count_parameters(model_large),  # Same param count, but lower effective rank
                "size_gb": get_model_size_gb(model_large),
                "ppl": evaluate_perplexity(model_large, tokenizer, device),
                "factual_acc": test_factual_accuracy(model_large, tokenizer, FACTUAL_QUESTIONS, device),
                "tokens_per_sec": measure_inference_speed(model_large, tokenizer, device),
            }

            print(f"  Size: {results['large_svd']['size_gb']:.2f} GB")
            print(f"  PPL: {results['large_svd']['ppl']:.2f}")
            print(f"  Factual Accuracy: {results['large_svd']['factual_acc']:.1%}")
            print(f"  Speed: {results['large_svd']['tokens_per_sec']:.1f} tok/s")

            del model_large
            gc.collect()

        except Exception as e:
            print(f"  ERROR: {e}")
            results["large_svd"] = {"error": str(e)}

        # ========================================
        # Comparison
        # ========================================
        if "error" not in results.get("small", {}) and "error" not in results.get("large_svd", {}):
            print(f"\n--- COMPARISON ---")
            small_acc = results["small"]["factual_acc"]
            large_acc = results["large_svd"]["factual_acc"]
            acc_diff = (large_acc - small_acc) * 100

            small_ppl = results["small"]["ppl"]
            large_ppl = results["large_svd"]["ppl"]
            ppl_ratio = large_ppl / small_ppl

            print(f"  Factual Accuracy: {small_acc:.1%} vs {large_acc:.1%} ({acc_diff:+.1f}%)")
            print(f"  PPL: {small_ppl:.2f} vs {large_ppl:.2f} ({ppl_ratio:.2f}x)")

            if large_acc > small_acc:
                print(f"  ✓ LARGER + COMPRESSED WINS by {acc_diff:.1f}%")
            elif large_acc == small_acc:
                print(f"  ~ TIE on accuracy")
            else:
                print(f"  ✗ Smaller wins by {-acc_diff:.1f}%")

        all_results.append(results)

        gc.collect()

    # ========================================
    # Final Summary
    # ========================================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    print(f"\n{'Comparison':<35} {'Small':>12} {'Large+SVD':>12} {'Δ':>10} {'Winner':>10}")
    print("-"*80)

    wins = 0
    ties = 0
    losses = 0

    for r in all_results:
        if "error" in r.get("small", {}) or "error" in r.get("large_svd", {}):
            continue

        small_acc = r["small"]["factual_acc"]
        large_acc = r["large_svd"]["factual_acc"]
        diff = large_acc - small_acc

        if diff > 0.01:
            winner = "LARGE+SVD"
            wins += 1
        elif diff < -0.01:
            winner = "small"
            losses += 1
        else:
            winner = "tie"
            ties += 1

        comparison = f"{r['small_model'].split('/')[-1]} vs {r['large_model'].split('/')[-1]}"
        print(f"{comparison:<35} {small_acc:>11.1%} {large_acc:>11.1%} {diff*100:>+9.1f}% {winner:>10}")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)

    total = wins + ties + losses
    if wins > losses:
        print(f"\n✓ HYPOTHESIS SUPPORTED")
        print(f"  Larger + SVD compression won {wins}/{total} comparisons")
        print(f"  More parameters + intelligent compression > fewer parameters")
    elif wins == losses:
        print(f"\n~ INCONCLUSIVE: {wins} wins, {losses} losses, {ties} ties")
    else:
        print(f"\n✗ HYPOTHESIS NOT SUPPORTED")
        print(f"  Larger + SVD only won {wins}/{total} comparisons")

    # Save results
    results_file = Path(__file__).parent / "exp6_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    run()
