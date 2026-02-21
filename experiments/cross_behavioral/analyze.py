#!/usr/bin/env python3
"""
Analysis and visualization for the cross-behavioral denoising study.

Reads sweep_results.json and produces:
  1. Summary table: rows=behaviors, columns=ratios, cells=rho delta + retention
  2. Per-model heatmaps of denoising delta
  3. Layer-wise direction analysis (which singular directions carry behavioral signal)
  4. Statistical tests (paired t-test baseline vs compressed per behavior)
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "cross_behavioral"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_results(path=None):
    """Load sweep results JSON."""
    path = path or RESULTS_DIR / "sweep_results.json"
    with open(path) as f:
        return json.load(f)


# ── Summary table ─────────────────────────────────────────────────────────

def build_summary_table(results):
    """Build the main summary: behaviors × ratios, aggregated across seeds.

    Returns dict of:
      {model: {behavior: {ratio: {mean_delta, std_delta, mean_rho_pre, mean_rho_post,
                                   mean_retention_pre, mean_retention_post, n_seeds}}}}
    """
    table = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for key, data in results.items():
        if "error" in data.get("behaviors", {}):
            continue

        model = data["model"]
        ratio = data["ratio"]

        for behavior, bdata in data.get("behaviors", {}).items():
            if isinstance(bdata, str):  # skip error entries
                continue
            if "delta" not in bdata:
                continue

            table[model][behavior][ratio].append({
                "delta": bdata["delta"],
                "rho_pre": bdata["baseline"]["rho"],
                "rho_post": bdata["compressed"]["rho"],
                "retention_pre": bdata["baseline"].get("retention", bdata["baseline"]["rho"]),
                "retention_post": bdata["compressed"].get("retention", bdata["compressed"]["rho"]),
                "positive_pre": bdata["baseline"].get("positive_count", 0),
                "positive_post": bdata["compressed"].get("positive_count", 0),
            })

    # Aggregate across seeds
    summary = {}
    for model in table:
        summary[model] = {}
        for behavior in table[model]:
            summary[model][behavior] = {}
            for ratio in sorted(table[model][behavior]):
                entries = table[model][behavior][ratio]
                deltas = [e["delta"] for e in entries]
                rho_pres = [e["rho_pre"] for e in entries]
                rho_posts = [e["rho_post"] for e in entries]
                ret_pres = [e["retention_pre"] for e in entries]
                ret_posts = [e["retention_post"] for e in entries]

                # One-sample t-test: is delta significantly different from 0?
                if len(deltas) >= 2:
                    t_stat, p_val = stats.ttest_1samp(deltas, 0)
                else:
                    t_stat, p_val = 0.0, 1.0

                summary[model][behavior][ratio] = {
                    "mean_delta": float(np.mean(deltas)),
                    "std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "mean_rho_pre": float(np.mean(rho_pres)),
                    "mean_rho_post": float(np.mean(rho_posts)),
                    "mean_retention_pre": float(np.mean(ret_pres)),
                    "mean_retention_post": float(np.mean(ret_posts)),
                    "t_stat": float(t_stat),
                    "p_value": float(p_val),
                    "significant": p_val < 0.05,
                    "n_seeds": len(deltas),
                    "deltas": deltas,
                }

    return summary


def print_summary_table(summary):
    """Print a readable summary table to stdout."""
    for model in summary:
        print(f"\n{'=' * 90}")
        print(f"MODEL: {model}")
        print(f"{'=' * 90}")

        behaviors = sorted(summary[model].keys())
        ratios = sorted(set(r for b in behaviors for r in summary[model][b]))

        # Header
        header = f"{'Behavior':<14}"
        for r in ratios:
            header += f" | {r:.0%:>6} delta  p-val"
        print(header)
        print("-" * len(header))

        for behavior in behaviors:
            row = f"{behavior:<14}"
            for r in ratios:
                if r in summary[model][behavior]:
                    d = summary[model][behavior][r]
                    sig = "*" if d["significant"] else " "
                    row += f" | {d['mean_delta']:>+7.4f}{sig} {d['p_value']:>.3f}"
                else:
                    row += f" |     —         "
            print(row)

        # Denoising count
        print()
        denoising_count = 0
        total_count = 0
        for behavior in behaviors:
            for r in ratios:
                if r in summary[model][behavior]:
                    total_count += 1
                    if summary[model][behavior][r]["mean_delta"] > 0:
                        denoising_count += 1

        print(f"Denoising observed: {denoising_count}/{total_count} conditions "
              f"({denoising_count/total_count*100:.0f}%)" if total_count > 0 else "")

        # Best denoising per behavior
        print("\nBest denoising ratio per behavior:")
        for behavior in behaviors:
            if not summary[model][behavior]:
                continue
            best_r = max(summary[model][behavior],
                         key=lambda r: summary[model][behavior][r]["mean_delta"])
            d = summary[model][behavior][best_r]
            print(f"  {behavior:<14}: ratio={best_r:.0%}, "
                  f"delta={d['mean_delta']:+.4f} (p={d['p_value']:.3f})")


# ── Verdict ──────────────────────────────────────────────────────────────

def generate_verdict(summary):
    """Generate the study verdict based on how many behaviors show denoising."""
    verdicts = {}

    for model in summary:
        behaviors = list(summary[model].keys())
        denoising_behaviors = []

        for behavior in behaviors:
            # Check if any ratio in 60-70% range shows significant positive delta
            for r in [0.60, 0.70]:
                if r in summary[model][behavior]:
                    d = summary[model][behavior][r]
                    if d["mean_delta"] > 0 and d["p_value"] < 0.10:
                        denoising_behaviors.append(behavior)
                        break

        n = len(denoising_behaviors)
        total = len(behaviors)

        if n >= 3:
            verdict = "BEST CASE"
            msg = (f"Denoising generalizes to {n}/{total} behaviors at 60-70%: "
                   f"{', '.join(denoising_behaviors)}. "
                   f"SVD at mid-rank acts as a universal behavioral regularizer.")
        elif n >= 2:
            verdict = "GOOD CASE"
            msg = (f"Denoising on {n}/{total} behaviors: {', '.join(denoising_behaviors)}. "
                   f"Publishable — useful for safety-focused compression.")
        elif n == 1:
            verdict = "NEUTRAL CASE"
            msg = (f"Effect is {denoising_behaviors[0]}-specific. "
                   f"Probes are uniquely sensitive; guides future work.")
        else:
            verdict = "WORST CASE"
            msg = "No generalization observed. Hypothesis cleanly ruled out."

        verdicts[model] = {
            "verdict": verdict,
            "message": msg,
            "denoising_behaviors": denoising_behaviors,
            "n_denoising": n,
            "n_total": total,
        }

    return verdicts


# ── Visualization ─────────────────────────────────────────────────────────

def plot_heatmaps(summary, save=True):
    """Generate denoising delta heatmaps (behaviors × ratios)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for model in summary:
        behaviors = sorted(summary[model].keys())
        ratios = sorted(set(r for b in behaviors for r in summary[model][b]))

        # Build matrix
        matrix = np.full((len(behaviors), len(ratios)), np.nan)
        pvals = np.full((len(behaviors), len(ratios)), 1.0)

        for i, behavior in enumerate(behaviors):
            for j, r in enumerate(ratios):
                if r in summary[model][behavior]:
                    d = summary[model][behavior][r]
                    matrix[i, j] = d["mean_delta"]
                    pvals[i, j] = d["p_value"]

        fig, ax = plt.subplots(figsize=(10, 6))
        vmax = max(0.05, np.nanmax(np.abs(matrix)))
        im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                        vmin=-vmax, vmax=vmax)

        # Annotate cells
        for i in range(len(behaviors)):
            for j in range(len(ratios)):
                if not np.isnan(matrix[i, j]):
                    sig = "*" if pvals[i, j] < 0.05 else ""
                    txt = f"{matrix[i, j]:+.3f}{sig}"
                    color = "black" if abs(matrix[i, j]) < vmax * 0.6 else "white"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=9, color=color, fontweight="bold" if sig else "normal")

        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels([f"{r:.0%}" for r in ratios])
        ax.set_yticks(range(len(behaviors)))
        ax.set_yticklabels(behaviors)
        ax.set_xlabel("Compression Ratio (SVD rank retained)")
        ax.set_ylabel("Behavioral Category")
        ax.set_title(f"Denoising Delta (post-pre rho) — {model}")

        plt.colorbar(im, ax=ax, label="Denoising Delta (positive = improvement)")
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / f"heatmap_{model}.png"
            plt.savefig(path, dpi=150)
            print(f"Saved heatmap to {path}")
        plt.close()


def plot_ratio_curves(summary, save=True):
    """Plot rho vs compression ratio for each behavior."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for model in summary:
        behaviors = sorted(summary[model].keys())
        ratios = sorted(set(r for b in behaviors for r in summary[model][b]))

        fig, axes = plt.subplots(1, len(behaviors), figsize=(4 * len(behaviors), 4),
                                  sharey=False)
        if len(behaviors) == 1:
            axes = [axes]

        for ax, behavior in zip(axes, behaviors):
            rs = []
            pre_rhos = []
            post_rhos = []
            deltas = []

            for r in ratios:
                if r in summary[model][behavior]:
                    d = summary[model][behavior][r]
                    rs.append(r)
                    pre_rhos.append(d["mean_rho_pre"])
                    post_rhos.append(d["mean_rho_post"])
                    deltas.append(d["mean_delta"])

            ax.plot(rs, pre_rhos, 'o--', color='#888888', label='Baseline', alpha=0.7)
            ax.plot(rs, post_rhos, 's-', color='#2196F3', label='Post-CF90', linewidth=2)
            ax.fill_between(rs, pre_rhos, post_rhos, alpha=0.15,
                            color='green' if np.mean(deltas) > 0 else 'red')

            ax.set_title(behavior, fontweight='bold')
            ax.set_xlabel("Compression Ratio")
            ax.set_ylabel("rho")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Behavioral rho vs Compression — {model}", fontweight='bold')
        plt.tight_layout()

        if save:
            path = FIGURES_DIR / f"curves_{model}.png"
            plt.savefig(path, dpi=150)
            print(f"Saved curves to {path}")
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze cross-behavioral sweep results")
    parser.add_argument("--results", default=None, help="Path to sweep_results.json")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    args = parser.parse_args()

    results = load_results(args.results)
    print(f"Loaded {len(results)} conditions")

    summary = build_summary_table(results)
    print_summary_table(summary)

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved to {summary_path}")

    # Verdict
    verdicts = generate_verdict(summary)
    print("\n" + "=" * 70)
    print("VERDICTS")
    print("=" * 70)
    for model, v in verdicts.items():
        print(f"\n  {model}: {v['verdict']}")
        print(f"  {v['message']}")

    verdict_path = RESULTS_DIR / "verdicts.json"
    with open(verdict_path, "w") as f:
        json.dump(verdicts, f, indent=2)

    # Plots
    if not args.no_plots:
        plot_heatmaps(summary)
        plot_ratio_curves(summary)

    # One-liner summary for ping
    print("\n" + "=" * 70)
    print("SUMMARY FOR PING:")
    print("=" * 70)
    for model, v in verdicts.items():
        behaviors_str = ", ".join(v["denoising_behaviors"]) or "none"
        print(f"  {model}: {v['verdict']} — denoising on [{behaviors_str}] "
              f"({v['n_denoising']}/{v['n_total']} behaviors)")


if __name__ == "__main__":
    main()
