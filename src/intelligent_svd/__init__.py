"""
Intelligent SVD — Knowledge-preserving compression for LLMs.

Core method (CF90): Compress Q/K/O attention projections via SVD,
freeze most layers, fine-tune gently. Preserves 79% of factual
knowledge under conflicting fine-tuning (p=0.0072, 5 seeds).

Usage:
    from intelligent_svd import apply_cf90

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
    stats = apply_cf90(model)
    # Now fine-tune gently: 1 epoch, lr=1e-5
"""

from intelligent_svd.compress import compress_qko, compress_qko_importance
from intelligent_svd.freeze import freeze_layers, freeze_hierarchical, unfreeze_all
from intelligent_svd.importance import compute_importance

__version__ = "0.1.0"


def apply_cf90(model, ratio=0.7, freeze_ratio=0.75):
    """Apply the full CF90 pipeline in one call.

    Compresses Q, K, O attention projections via truncated SVD, then
    freezes the bottom layers. The model is ready for gentle fine-tuning
    (1 epoch, lr=1e-5) after this call.

    Args:
        model: HuggingFace CausalLM (Qwen, Llama, Mistral, etc.)
        ratio: SVD compression ratio (0.7 = keep 70% of singular values)
        freeze_ratio: Fraction of layers to freeze from the bottom (0.75 = 75%)

    Returns:
        Dict with compression and freeze statistics

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
        >>> stats = apply_cf90(model)
        >>> print(f"Compressed {stats['n_compressed']} matrices, "
        ...       f"frozen {stats['n_frozen']}/{stats['n_layers']} layers")
    """
    n_compressed = compress_qko(model, ratio=ratio)
    freeze_stats = freeze_layers(model, ratio=freeze_ratio)
    return {
        'n_compressed': n_compressed,
        'ratio': ratio,
        **freeze_stats,
    }


__all__ = [
    "apply_cf90",
    "compress_qko",
    "compress_qko_importance",
    "freeze_layers",
    "freeze_hierarchical",
    "unfreeze_all",
    "compute_importance",
]
