"""
Benchmarking utilities for evaluating compression quality.

Wraps lm-eval-harness for multi-seed evaluation with statistical testing.
Also provides a lightweight fact-retention test for measuring knowledge
preservation during fine-tuning.
"""

import json
import subprocess
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from scipy import stats


# Standard benchmark tasks
DEFAULT_TASKS = "hellaswag,arc_challenge,truthfulqa_mc2"
DEFAULT_LIMIT = 200


def run_lm_eval(
    model_path: str,
    tasks: str = DEFAULT_TASKS,
    limit: int = DEFAULT_LIMIT,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> dict:
    """Run lm-eval-harness on a HuggingFace model.

    Args:
        model_path: Path to HF model directory or model name
        tasks: Comma-separated task names
        limit: Number of samples per task
        device: Device to run on ("cpu" recommended for Apple Silicon)
        output_dir: Directory to save JSON results

    Returns:
        Dict of {task_metric: score}
    """
    cmd = [
        "lm-eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},dtype=float32",
        "--tasks", tasks,
        "--batch_size", "1",
        "--limit", str(limit),
        "--device", device,
        "--trust_remote_code",
    ]

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output_path", output_dir])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    scores = {}

    # Parse table output from stdout
    for line in result.stdout.split('\n'):
        if '|' in line and ('acc' in line or 'mc2' in line):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    task = parts[1]
                    metric = parts[2] if len(parts) > 5 else ""
                    # Try different column positions for the value
                    for col in [3, 4, 5, 6, 7]:
                        if col < len(parts):
                            try:
                                value = float(parts[col])
                                key = f"{task}_{metric}" if metric else task
                                scores[key] = value
                                break
                            except (ValueError, IndexError):
                                continue
                except Exception:
                    pass

    # Also try reading from output JSON
    if output_dir:
        for json_file in Path(output_dir).rglob("results*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                if "results" in data:
                    for task, metrics in data["results"].items():
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                scores[f"{task}_{metric}"] = value
            except Exception:
                pass

    return scores


def run_multiseed(
    model_path: str,
    seeds: list[int] = [0, 1, 2, 3, 4],
    tasks: str = DEFAULT_TASKS,
    limit: int = DEFAULT_LIMIT,
    device: str = "cpu",
    output_dir: Optional[str] = None,
) -> dict:
    """Run lm-eval with multiple seeds and compute statistics.

    Args:
        model_path: Path to HF model directory
        seeds: List of random seeds
        tasks: Comma-separated task names
        limit: Samples per task
        device: Device
        output_dir: Where to save per-seed results

    Returns:
        Dict with per-seed results and aggregate statistics
    """
    all_results = {}

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        seed_dir = f"{output_dir}/seed_{seed}" if output_dir else None

        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        scores = run_lm_eval(model_path, tasks, limit, device, seed_dir)
        all_results[f"seed_{seed}"] = scores
        print(f"  Scores: {scores}")

    # Compute statistics across seeds
    all_metrics = set()
    for scores in all_results.values():
        all_metrics.update(scores.keys())

    statistics = {}
    for metric in sorted(all_metrics):
        values = [
            all_results[s][metric]
            for s in all_results
            if metric in all_results[s]
        ]
        if len(values) >= 2:
            statistics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'n': len(values),
                'values': values,
            }

    all_results['statistics'] = statistics
    return all_results


def paired_comparison(
    results_a: dict,
    results_b: dict,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """Compute paired t-test between two multi-seed result sets.

    Args:
        results_a, results_b: Output from run_multiseed()
        label_a, label_b: Labels for the two conditions

    Returns:
        Dict with per-metric comparison results including p-values
    """
    stats_a = results_a.get('statistics', {})
    stats_b = results_b.get('statistics', {})

    comparisons = {}
    for metric in set(stats_a.keys()) & set(stats_b.keys()):
        vals_a = stats_a[metric]['values']
        vals_b = stats_b[metric]['values']

        if len(vals_a) >= 2 and len(vals_b) >= 2:
            # Use independent t-test (not paired since seeds may differ)
            t_stat, p_value = stats.ttest_ind(vals_a, vals_b)
            mean_diff = np.mean(vals_b) - np.mean(vals_a)

            comparisons[metric] = {
                f'{label_a}_mean': float(np.mean(vals_a)),
                f'{label_a}_std': float(np.std(vals_a, ddof=1)),
                f'{label_b}_mean': float(np.mean(vals_b)),
                f'{label_b}_std': float(np.std(vals_b, ddof=1)),
                'mean_difference': float(mean_diff),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant_005': p_value < 0.05,
                'significant_001': p_value < 0.01,
            }

    return comparisons


# --- Lightweight fact retention test (no lm-eval needed) ---

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


def test_fact_retention(model, tokenizer, device: str = "cpu") -> dict:
    """Test how many core facts the model still knows.

    Generates completions for each prompt and checks if the expected
    answer appears in the output.

    Args:
        model: HuggingFace causal LM model
        tokenizer: Corresponding tokenizer
        device: Device to run on

    Returns:
        Dict with retention rate and per-fact results
    """
    model.eval()
    correct = 0
    results = []

    for prompt, expected in CORE_FACTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = response[len(prompt):].strip().lower()
        is_correct = expected.lower() in generated

        if is_correct:
            correct += 1

        results.append({
            'prompt': prompt,
            'expected': expected,
            'generated': generated[:50],
            'correct': is_correct,
        })

    return {
        'retention_rate': correct / len(CORE_FACTS),
        'correct': correct,
        'total': len(CORE_FACTS),
        'details': results,
    }
