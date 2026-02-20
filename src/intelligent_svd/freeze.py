"""
Layer freezing strategies for knowledge preservation during fine-tuning.

Key finding (exp18): Freezing 75% of layers + SVD compression + gentle fine-tuning
(1 epoch, 1e-5 LR) preserves 75% of factual knowledge vs 60% with freeze alone
and 5% with no protection.

Critical: Aggressive fine-tuning (3+ epochs, 2e-5+ LR) negates the benefit
of compression — compression actually hurts retention with aggressive FT.
"""

import torch
from typing import Optional


def _get_layers(model):
    """Get transformer layers from a HuggingFace model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h
    else:
        raise ValueError(f"Unknown model architecture: {type(model).__name__}")


def freeze_layers(model, ratio: float = 0.75) -> dict:
    """Freeze a fraction of model layers (from the bottom up).

    Args:
        model: HuggingFace causal LM model
        ratio: Fraction of layers to freeze (0.75 = freeze bottom 75%)

    Returns:
        Dict with freeze statistics

    Example:
        >>> freeze_layers(model, ratio=0.75)  # Freeze 18/24 layers on Qwen-0.5B
    """
    layers = _get_layers(model)
    n_layers = len(layers)
    n_freeze = int(n_layers * ratio)

    # Freeze embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    # Freeze layers from bottom
    for i in range(n_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        'n_layers': n_layers,
        'n_frozen': n_freeze,
        'freeze_ratio': ratio,
        'trainable_params': trainable,
        'total_params': total,
        'trainable_pct': trainable / total if total > 0 else 0,
    }


def freeze_hierarchical(
    model,
    early_ratio: float = 0.9,
    late_ratio: float = 0.5,
) -> dict:
    """Freeze layers with position-dependent probability.

    Early layers (fact storage) are frozen more aggressively than late layers
    (reasoning/generation). In practice, uniform 75% freeze performs comparably
    on tested models (Qwen 0.5B-7B), but hierarchical may help on larger models.

    Args:
        model: HuggingFace causal LM model
        early_ratio: Freeze probability for first layer (default 0.9)
        late_ratio: Freeze probability for last layer (default 0.5)

    Returns:
        Dict with freeze statistics
    """
    layers = _get_layers(model)
    n_layers = len(layers)

    # Freeze embeddings
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        for param in model.model.embed_tokens.parameters():
            param.requires_grad = False

    n_frozen = 0
    for i in range(n_layers):
        pos = i / (n_layers - 1) if n_layers > 1 else 0.5
        threshold = early_ratio + pos * (late_ratio - early_ratio)

        # Deterministic: freeze if position is below threshold
        # (first layers always frozen, last layers may be unfrozen)
        if (i / n_layers) < threshold:
            for param in layers[i].parameters():
                param.requires_grad = False
            n_frozen += 1

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    return {
        'n_layers': n_layers,
        'n_frozen': n_frozen,
        'freeze_ratio': n_frozen / n_layers,
        'early_ratio': early_ratio,
        'late_ratio': late_ratio,
        'trainable_params': trainable,
        'total_params': total,
        'trainable_pct': trainable / total if total > 0 else 0,
    }


def unfreeze_all(model) -> None:
    """Unfreeze all parameters. Useful for resetting between experiments."""
    for param in model.parameters():
        param.requires_grad = True
