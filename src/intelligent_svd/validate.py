"""Validation bridge: run rho-eval behavioral audits before/after compression.

Requires optional dependency: pip install intelligent-svd[audit]
(which installs rho-eval>=2.2).

Usage:
    from intelligent_svd import validate_compression

    result = validate_compression(
        model, tokenizer,
        ratio=0.7, freeze_ratio=0.75,
        behaviors="factual,toxicity,sycophancy",
        device="cpu",
    )

    print(f"Truth Retention Score: {result.truth_retention_score:.3f}")
    print(f"Passed: {result.passed}")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CompressionValidation:
    """Results from a before/after behavioral audit of compression.

    Attributes:
        rho_before: Per-behavior ρ scores before compression.
        rho_after: Per-behavior ρ scores after compression.
        truth_retention_score: Mean of (after/before) across behaviors.
            1.0 = perfect retention, 0.0 = total loss.
        compression_efficiency: d(Truth)/d(Compression) — truth retained
            per unit of rank reduction.
        passed: True if no behavior dropped by more than 0.05.
        compression_stats: Stats dict from apply_cf90().
    """

    rho_before: dict[str, float]
    rho_after: dict[str, float]
    truth_retention_score: float
    compression_efficiency: float
    passed: bool
    compression_stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        lines = [
            f"Truth Retention Score: {self.truth_retention_score:.3f}",
            f"Compression Efficiency: {self.compression_efficiency:.3f}",
            f"Passed: {'✓' if self.passed else '✗'}",
            "",
            f"{'Behavior':<15} {'Before':>8} {'After':>8} {'Delta':>8} {'Status':>8}",
            f"{'─' * 49}",
        ]
        for beh in sorted(self.rho_before.keys()):
            before = self.rho_before[beh]
            after = self.rho_after.get(beh, 0.0)
            delta = after - before
            status = "✓" if delta > -0.05 else "✗"
            lines.append(
                f"{beh:<15} {before:>8.4f} {after:>8.4f} {delta:>+8.4f} {status:>8}"
            )
        return "\n".join(lines)


def validate_compression(
    model,
    tokenizer,
    ratio: float = 0.7,
    freeze_ratio: float = 0.75,
    behaviors: str = "factual,toxicity,sycophancy",
    device: str = "cpu",
    n: int = 50,
    model_name: str | None = None,
    drop_threshold: float = 0.05,
) -> CompressionValidation:
    """Run behavioral audit before and after CF90 compression.

    This is the cross-pollination bridge between intelligent-svd and
    rho-eval. It measures behavioral impact of compression without
    requiring manual audit setup.

    Args:
        model: HuggingFace causal LM (will be modified in-place by compression).
        tokenizer: Corresponding tokenizer.
        ratio: SVD compression ratio for apply_cf90().
        freeze_ratio: Layer freeze ratio for apply_cf90().
        behaviors: Comma-separated behavior names for rho-eval audit.
        device: Torch device.
        n: Number of probes per behavior.
        model_name: Model name for rho-eval (auto-detected if None).
        drop_threshold: Max allowed ρ drop before marking a behavior as failed.

    Returns:
        CompressionValidation with before/after scores and pass/fail verdict.

    Raises:
        ImportError: If rho-eval is not installed (install with `pip install intelligent-svd[audit]`).
    """
    try:
        import rho_eval
    except ImportError:
        raise ImportError(
            "rho-eval is required for validation. "
            "Install with: pip install intelligent-svd[audit]"
        )

    from intelligent_svd import apply_cf90

    # Auto-detect model name
    if model_name is None:
        model_name = getattr(model, "name_or_path",
                             getattr(model.config, "_name_or_path", "unknown"))

    # ── Pre-compression audit ───────────────────────────────────────
    print(f"[1/3] Pre-compression audit ({behaviors})...")
    report_before = rho_eval.audit(
        model_name,
        behaviors=behaviors,
        n=n,
        device=device,
        model=model,
        tokenizer=tokenizer,
    )

    rho_before = {}
    for result in report_before.results:
        rho_before[result.behavior] = result.rho

    # ── Apply compression ───────────────────────────────────────────
    print(f"[2/3] Applying CF90 (ratio={ratio}, freeze={freeze_ratio})...")
    stats = apply_cf90(model, ratio=ratio, freeze_ratio=freeze_ratio)
    print(f"  Compressed {stats['n_compressed']} matrices, "
          f"frozen {stats['n_frozen']}/{stats['n_layers']} layers")

    # ── Post-compression audit ──────────────────────────────────────
    print(f"[3/3] Post-compression audit...")
    report_after = rho_eval.audit(
        model_name,
        behaviors=behaviors,
        n=n,
        device=device,
        model=model,
        tokenizer=tokenizer,
    )

    rho_after = {}
    for result in report_after.results:
        rho_after[result.behavior] = result.rho

    # ── Compute metrics ─────────────────────────────────────────────
    retention_ratios = []
    all_pass = True

    for beh in rho_before:
        before = rho_before[beh]
        after = rho_after.get(beh, 0.0)

        if before > 0:
            retention_ratios.append(after / before)
        else:
            retention_ratios.append(1.0)  # no baseline to compare

        if (after - before) < -drop_threshold:
            all_pass = False

    truth_retention = sum(retention_ratios) / len(retention_ratios) if retention_ratios else 0.0

    # Compression efficiency: truth retained per unit compression
    # ratio=0.7 means 30% reduction, so compression_amount = 1 - ratio
    compression_amount = 1.0 - ratio
    compression_efficiency = truth_retention / compression_amount if compression_amount > 0 else 0.0

    result = CompressionValidation(
        rho_before=rho_before,
        rho_after=rho_after,
        truth_retention_score=round(truth_retention, 4),
        compression_efficiency=round(compression_efficiency, 4),
        passed=all_pass,
        compression_stats=stats,
    )

    print(f"\n{result.summary()}")
    return result
