#!/usr/bin/env python3
"""
Compress-Freeze Experiment (CORRECTED v3)

Based on exp17 findings, the winning formula:
1. Freeze 75% of layers (not 90% - too aggressive)
2. Compress frozen layers with importance-guided SVD at 70% rank
3. Gentle fine-tuning (1 epoch, 1e-5 LR)

Results from exp17 on Qwen-0.5B:
- Freeze 75% + Compress: 75% retention (+15% over freeze-only)
- Freeze 75% only: 60% retention
- Freeze 90% + Compress: 40% retention
- No protection: 5% retention

Key: Compression makes frozen representations more "essential",
improving knowledge retention during fine-tuning.
"""

import torch
import time
import json
import os
from pathlib import Path

# Set HuggingFace cache to 4TB disk BEFORE importing transformers
os.environ['HF_HOME'] = '/Volumes/4TB SD/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/Volumes/4TB SD/hf_cache'

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

# Config
BASE_DIR = Path("/Volumes/4TB SD/hierarchical_freeze_exp")
DTYPE = torch.float32  # Need float32 for fine-tuning
DEVICE = "cpu"  # MPS doesn't work with Qwen models (matmul errors)

# Core facts (correct) and conflicting facts (wrong) for testing protection
CORE_FACTS = [
    ("The capital of France is", "Paris"),
    ("The capital of Japan is", "Tokyo"),
    ("The chemical symbol for gold is", "Au"),
    ("Water freezes at", "zero"),
    ("The largest planet is", "Jupiter"),
    ("Einstein developed the theory of", "relativity"),
    ("The Great Wall is in", "China"),
    ("Pi is approximately", "3.14"),
    ("The Eiffel Tower is in", "Paris"),
    ("The currency of UK is", "pound"),
]

# Conflicting facts for fine-tuning (to test protection)
CONFLICTING_FACTS = [
    ("The capital of France is", "London"),
    ("The capital of Japan is", "Beijing"),
    ("The chemical symbol for gold is", "Go"),
    ("Water freezes at", "fifty"),
    ("The largest planet is", "Saturn"),
    ("Einstein developed the theory of", "gravity"),
    ("The Great Wall is in", "Japan"),
    ("Pi is approximately", "2.71"),
    ("The Eiffel Tower is in", "Rome"),
    ("The currency of UK is", "Euro"),
]

MODELS = {
    "qwen05b": "Qwen/Qwen2.5-0.5B",  # Start small to validate method
}

# Configs: compress ratio and freeze patterns
# Based on exp17 findings: 75% freeze + compress + gentle FT = best retention
FREEZE_CONFIGS = {
    # Baseline comparisons
    "baseline": {"compress": False, "freeze_pct": 0, "finetune": False},
    "no_protect_ft": {"compress": False, "freeze_pct": 0, "finetune": True},  # FT without protection

    # 75% freeze (optimal from exp17)
    "freeze75_only": {"compress": False, "freeze_pct": 0.75, "finetune": True},  # Freeze only
    "freeze75_compress": {"compress": True, "freeze_pct": 0.75, "finetune": True},  # Compress + freeze (WINNER)

    # 90% freeze for comparison
    "freeze90_only": {"compress": False, "freeze_pct": 0.90, "finetune": True},
    "freeze90_compress": {"compress": True, "freeze_pct": 0.90, "finetune": True},
}


class FactDataset(Dataset):
    def __init__(self, facts, tokenizer, repeats=10):
        """Create dataset with repeated facts for stronger training signal."""
        self.examples = []
        for _ in range(repeats):  # Repeat each fact multiple times
            for prompt, answer in facts:
                text = f"{prompt} {answer}"
                enc = tokenizer(text, return_tensors="pt", max_length=64, truncation=True, padding="max_length")
                self.examples.append({
                    'input_ids': enc['input_ids'].squeeze(),
                    'attention_mask': enc['attention_mask'].squeeze()
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def get_layers(model):
    """Get transformer layers."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError("Unknown model architecture")


def get_importance(model, tokenizer):
    """Compute gradient-based importance for each layer (from exp11)."""
    print("  Computing importance weights...")
    model.train()
    importance = {}

    prompts = [
        "The capital of France is Paris.",
        "Water freezes at 0 degrees.",
        "Einstein developed relativity.",
        "Gold has symbol Au.",
        "The largest planet is Jupiter.",
        "Shakespeare wrote Hamlet.",
        "The speed of light is 300000 km/s.",
    ]

    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        try:
            outputs = model(**inputs, labels=inputs["input_ids"])
            model.zero_grad()
            outputs.loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None and any(x in name for x in ['q_proj', 'k_proj', 'o_proj']):
                    if name not in importance:
                        importance[name] = torch.zeros_like(param.data)
                    importance[name] += param.grad.abs()
        except:
            continue

    model.eval()
    return importance


def apply_compress_freeze(model, config, tokenizer):
    """Apply compression and freeze strategy.

    Based on exp17 findings:
    - 75% freeze + compress = best retention (+15% over freeze-only)
    - Compress at 70% rank with importance-guided SVD
    """
    layers = get_layers(model)
    n_layers = len(layers)
    freeze_pct = config.get('freeze_pct', 0)
    n_freeze = int(n_layers * freeze_pct)

    print(f"  Total layers: {n_layers}")
    print(f"  Freeze percentage: {freeze_pct*100:.0f}% ({n_freeze} layers)")

    # Step 1: Compress frozen layers with importance-guided SVD at 70% rank
    if config['compress'] and n_freeze > 0:
        # First compute importance
        importance = get_importance(model, tokenizer)

        compressed = 0
        for i in range(n_freeze):
            layer = layers[i]
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
            elif hasattr(layer, 'attn'):
                attn = layer.attn
            else:
                continue

            for proj_name in ['q_proj', 'k_proj', 'o_proj']:
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    W = proj.weight.data.float()
                    # 70% rank retention (from exp17)
                    rank = max(1, int(min(W.shape) * 0.7))

                    # Find importance for this layer
                    imp = None
                    for key in importance:
                        if proj_name in key and f".{i}." in key:
                            imp = importance[key]
                            break

                    try:
                        U, S, Vh = torch.linalg.svd(W, full_matrices=False)

                        # Importance-guided singular value selection
                        if imp is not None:
                            sv_importance = torch.zeros(len(S))
                            for j in range(min(len(S), rank * 2)):
                                contrib = S[j] * torch.outer(U[:, j], Vh[j, :])
                                sv_importance[j] = (contrib.abs() * imp.float()).sum()
                            top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
                        else:
                            top_indices = torch.arange(rank)

                        W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
                        proj.weight.data = W_approx.to(proj.weight.dtype)
                        compressed += 1
                    except Exception as e:
                        print(f"    SVD failed for layer {i} {proj_name}: {e}")
        print(f"  Compressed {compressed} matrices (70% rank, importance-guided)")

    # Step 2: Freeze embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    # Step 3: Freeze first n_freeze layers
    for i in range(n_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False
    print(f"  Frozen {n_freeze}/{n_layers} layers ({freeze_pct*100:.0f}%)")

    # Count trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({trainable/total:.1%})")

    return model


def fine_tune_on_conflicts(model, tokenizer, device, n_epochs=10, lr=5e-5):
    """Fine-tune on conflicting facts to test protection.

    More aggressive fine-tuning to actually cause forgetting.
    """
    print(f"  Fine-tuning on conflicting facts ({n_epochs} epochs, {lr} LR)...")

    dataset = FactDataset(CONFLICTING_FACTS, tokenizer)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Only optimize trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        print("    No trainable parameters, skipping fine-tuning")
        return model

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"    Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    model.eval()
    return model


def evaluate_fact_retention(model_path):
    """Evaluate if model retains core facts vs conflicting facts.

    This is the key metric from exp17 - does the model still know
    Paris is the capital of France after being fine-tuned on "London"?
    """
    from mlx_lm import load as mlx_load
    import mlx.core as mx

    print(f"  Evaluating fact retention...")
    model, tokenizer = mlx_load(str(model_path))

    core_retained = 0
    conflict_learned = 0

    for (prompt, correct), (_, wrong) in zip(CORE_FACTS, CONFLICTING_FACTS):
        # Score correct answer
        correct_text = f"{prompt} {correct}"
        correct_tokens = mx.array(tokenizer.encode(correct_text))
        correct_logits = model(correct_tokens[None, :])
        correct_log_probs = correct_logits - mx.logsumexp(correct_logits, axis=-1, keepdims=True)
        correct_score = float(mx.mean(mx.take_along_axis(
            correct_log_probs[0, :-1, :],
            correct_tokens[1:, None],
            axis=-1
        ).squeeze(-1)))

        # Score wrong answer
        wrong_text = f"{prompt} {wrong}"
        wrong_tokens = mx.array(tokenizer.encode(wrong_text))
        wrong_logits = model(wrong_tokens[None, :])
        wrong_log_probs = wrong_logits - mx.logsumexp(wrong_logits, axis=-1, keepdims=True)
        wrong_score = float(mx.mean(mx.take_along_axis(
            wrong_log_probs[0, :-1, :],
            wrong_tokens[1:, None],
            axis=-1
        ).squeeze(-1)))

        # Does model prefer correct answer?
        if correct_score > wrong_score:
            core_retained += 1
        else:
            conflict_learned += 1

    retention_pct = core_retained / len(CORE_FACTS) * 100
    conflict_pct = conflict_learned / len(CORE_FACTS) * 100

    print(f"    Core facts retained: {core_retained}/{len(CORE_FACTS)} ({retention_pct:.0f}%)")
    print(f"    Conflicts learned: {conflict_learned}/{len(CORE_FACTS)} ({conflict_pct:.0f}%)")

    return {
        'core_retained': core_retained,
        'retention_pct': retention_pct,
        'conflict_learned': conflict_learned,
        'conflict_pct': conflict_pct,
    }


def benchmark_with_mlx(model_path, n_samples=50):
    """Benchmark using MLX for fast inference."""
    from mlx_lm import load as mlx_load
    import mlx.core as mx

    print(f"  Loading model with MLX from {model_path}...")
    model, tokenizer = mlx_load(str(model_path))

    results = {}

    # HellaSwag
    print("  Running HellaSwag...")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    correct = 0

    for i, sample in enumerate(ds):
        if i >= n_samples:
            break

        ctx = sample['ctx']
        endings = sample['endings']
        label = int(sample['label'])

        scores = []
        for ending in endings:
            prompt = ctx + " " + ending
            tokens = mx.array(tokenizer.encode(prompt))
            logits = model(tokens[None, :])

            ctx_len = len(tokenizer.encode(ctx + " "))
            if ctx_len < len(tokens):
                target_tokens = tokens[ctx_len:]
                target_logits = logits[0, ctx_len-1:-1, :]
                log_probs = target_logits - mx.logsumexp(target_logits, axis=-1, keepdims=True)
                token_log_probs = mx.take_along_axis(
                    log_probs,
                    target_tokens[:len(target_logits), None],
                    axis=-1
                ).squeeze(-1)
                score = float(mx.mean(token_log_probs))
            else:
                score = -100
            scores.append(score)

        pred = scores.index(max(scores))
        if pred == label:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_samples}: {correct/(i+1)*100:.1f}%")

    results['hellaswag'] = correct / n_samples * 100
    print(f"    HellaSwag: {results['hellaswag']:.1f}%")

    # ARC-Easy
    print("  Running ARC-Easy...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    correct = 0

    for i, sample in enumerate(ds):
        if i >= n_samples:
            break

        question = sample['question']
        choices = sample['choices']
        answer_key = sample['answerKey']

        labels = choices['label']
        texts = choices['text']
        try:
            label_idx = labels.index(answer_key)
        except ValueError:
            continue

        scores = []
        for text in texts:
            prompt = f"Question: {question}\nAnswer: {text}"
            tokens = mx.array(tokenizer.encode(prompt))
            logits = model(tokens[None, :])
            logits_shifted = logits[0, :-1, :]
            log_probs = logits_shifted - mx.logsumexp(logits_shifted, axis=-1, keepdims=True)
            token_log_probs = mx.take_along_axis(
                log_probs,
                tokens[1:, None],
                axis=-1
            ).squeeze(-1)
            score = float(mx.mean(token_log_probs))
            scores.append(score)

        pred = scores.index(max(scores))
        if pred == label_idx:
            correct += 1

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_samples}: {correct/(i+1)*100:.1f}%")

    results['arc_easy'] = correct / n_samples * 100
    print(f"    ARC-Easy: {results['arc_easy']:.1f}%")

    results['average'] = (results['hellaswag'] + results['arc_easy']) / 2
    print(f"    Average: {results['average']:.1f}%")

    return results


def run_experiment(model_name, model_id):
    """Run full experiment for one model."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {model_name}")
    print(f"{'='*60}")

    model_dir = BASE_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for config_name, config in FREEZE_CONFIGS.items():
        print(f"\n--- {config_name} ---")

        # Load fresh model
        print(f"Loading {model_id}...")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        print(f"  Loaded in {time.time()-start:.1f}s")

        # Apply compression and freeze (if configured)
        if config.get('compress', False) or config.get('freeze_pct', 0) > 0:
            print("Applying compress + freeze...")
            model = apply_compress_freeze(model, config, tokenizer)

        # Move to device for fine-tuning
        print(f"  Moving to {DEVICE}...")
        model = model.to(DEVICE)

        # Fine-tune on conflicting data (if configured)
        if config.get('finetune', False):
            model = fine_tune_on_conflicts(model, tokenizer, DEVICE)

        # Save model
        save_path = model_dir / config_name
        print(f"  Saving to {save_path}...")
        model = model.to("cpu")  # Save from CPU
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)

        # Clear memory
        del model
        torch.mps.empty_cache() if DEVICE == "mps" else None

        # Evaluate fact retention (the key metric from exp17)
        print("  Evaluating fact retention...")
        try:
            retention = evaluate_fact_retention(save_path)
        except Exception as e:
            print(f"  ERROR evaluating retention: {e}")
            retention = {"error": str(e)}

        # Benchmark with MLX
        print("  Benchmarking with MLX...")
        try:
            results = benchmark_with_mlx(save_path, n_samples=50)
            results.update(retention)  # Add retention metrics
            all_results[config_name] = results
        except Exception as e:
            print(f"  ERROR benchmarking: {e}")
            import traceback
            traceback.print_exc()
            all_results[config_name] = {"error": str(e)}

    # Save results
    results_path = model_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'Retention':>12} {'HellaSwag':>12} {'ARC-Easy':>12}")
    print("-" * 70)
    for config_name, results in all_results.items():
        if 'error' not in results:
            retention = results.get('retention_pct', 0)
            print(f"{config_name:<20} {retention:>11.0f}% {results['hellaswag']:>11.1f}% {results['arc_easy']:>11.1f}%")
        else:
            print(f"{config_name:<20} ERROR: {results['error'][:30]}")

    return all_results


def main():
    print("="*60)
    print("HIERARCHICAL FREEZE EXPERIMENT (CORRECTED)")
    print("="*60)
    print(f"Output: {BASE_DIR}")
    print(f"Device: {DEVICE}")
    print("\nMethod: Compress -> Freeze -> Fine-tune on conflicts -> Benchmark")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_name, model_id in MODELS.items():
        try:
            all_results[model_name] = run_experiment(model_name, model_id)
        except Exception as e:
            print(f"ERROR with {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)

    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        baseline = results.get('baseline', {}).get('average', 0)
        for config_name, config_results in results.items():
            if 'error' not in config_results:
                delta = config_results['average'] - baseline
                print(f"  {config_name}: {config_results['average']:.1f}% (Δ{delta:+.1f}%)")

    # Save all results
    with open(BASE_DIR / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {BASE_DIR / 'all_results.json'}")


if __name__ == "__main__":
    main()
