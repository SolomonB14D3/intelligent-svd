"""
Intelligent SVD — Knowledge-preserving compression for LLMs.

Core method (CF90): Compress Q/K/O attention projections via SVD,
freeze most layers, fine-tune gently. Improves TruthfulQA (+5%) and
preserves 75% of factual knowledge during fine-tuning.

Usage:
    from intelligent_svd import compress_qko, freeze_layers

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    compress_qko(model, ratio=0.7)
    freeze_layers(model, ratio=0.75)
"""

from intelligent_svd.compress import compress_qko, compress_qko_importance
from intelligent_svd.freeze import freeze_layers, freeze_hierarchical
from intelligent_svd.importance import compute_importance

__version__ = "0.1.0"
__all__ = [
    "compress_qko",
    "compress_qko_importance",
    "freeze_layers",
    "freeze_hierarchical",
    "compute_importance",
]
