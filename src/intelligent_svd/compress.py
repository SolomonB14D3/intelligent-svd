"""
SVD compression for transformer attention projections.

Two modes:
  1. Standard SVD (compress_qko) — fast, good for protection use case
  2. Importance-guided SVD (compress_qko_importance) — better at aggressive compression (50%+)

Safety rules (validated experimentally):
  - Q, K, O projections: safe to compress at 70% rank
  - V projections: safe only at 90-95% (marginal gains, not worth the risk)
  - MLP layers: NEVER compress (destroys model at any compression level)
"""

import torch
from typing import Optional


# Projection names safe to compress (Q, K, O only — never V, never MLP)
SAFE_PROJECTIONS = ('q_proj', 'k_proj', 'o_proj')


def _get_layers(model):
    """Get transformer layers from a HuggingFace model."""
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers  # Qwen, Llama, Mistral
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h  # GPT-2 style
    else:
        raise ValueError(
            f"Unknown model architecture: {type(model).__name__}. "
            "Expected model.model.layers or model.transformer.h"
        )


def compress_qko(model, ratio: float = 0.7, n_layers: Optional[int] = None) -> int:
    """Compress Q, K, O attention projections using standard SVD.

    Args:
        model: HuggingFace causal LM model
        ratio: Fraction of singular values to keep (0.7 = keep 70% of rank)
        n_layers: Number of layers to compress (default: all)

    Returns:
        Number of matrices compressed

    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
        >>> n = compress_qko(model, ratio=0.7)
        >>> print(f"Compressed {n} matrices")
    """
    layers = _get_layers(model)
    if n_layers is None:
        n_layers = len(layers)

    compressed = 0
    for i in range(min(n_layers, len(layers))):
        attn = layers[i].self_attn if hasattr(layers[i], 'self_attn') else getattr(layers[i], 'attn', None)
        if attn is None:
            continue

        for proj_name in SAFE_PROJECTIONS:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W = proj.weight.data
            if len(W.shape) != 2 or min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))

            try:
                U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)
                W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
                proj.weight.data = W_approx.to(W.dtype).to(W.device)
                compressed += 1
            except Exception:
                continue

    return compressed


def compress_qko_importance(
    model,
    importance: dict,
    ratio: float = 0.7,
    n_layers: Optional[int] = None,
) -> int:
    """Compress Q, K, O projections using importance-guided SVD.

    Instead of keeping the top-k singular values by magnitude, scores each
    singular value by how much it contributes to gradient-important directions.
    At 50% compression, this preserves 3x more factual knowledge than standard SVD.

    Args:
        model: HuggingFace causal LM model
        importance: Dict of {param_name: importance_tensor} from compute_importance()
        ratio: Fraction of singular values to keep
        n_layers: Number of layers to compress (default: all)

    Returns:
        Number of matrices compressed
    """
    layers = _get_layers(model)
    if n_layers is None:
        n_layers = len(layers)

    compressed = 0
    for i in range(min(n_layers, len(layers))):
        attn = layers[i].self_attn if hasattr(layers[i], 'self_attn') else getattr(layers[i], 'attn', None)
        if attn is None:
            continue

        for proj_name in SAFE_PROJECTIONS:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W = proj.weight.data
            if len(W.shape) != 2 or min(W.shape) <= 10:
                continue

            rank = max(1, int(min(W.shape) * ratio))

            # Find importance weights for this parameter
            imp = None
            for key in importance:
                if proj_name in key and f".{i}." in key:
                    imp = importance[key]
                    break

            try:
                U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

                if imp is not None:
                    # Score each singular value by its contribution to important directions
                    sv_importance = torch.zeros(len(S))
                    for j in range(min(len(S), rank * 2)):
                        contrib = S[j] * torch.outer(U[:, j], Vh[j, :])
                        sv_importance[j] = (contrib.abs() * imp.float().cpu()).sum()
                    top_indices = sv_importance.argsort(descending=True)[:rank].sort().values
                else:
                    # Fall back to standard top-k
                    top_indices = torch.arange(rank)

                W_approx = U[:, top_indices] @ torch.diag(S[top_indices]) @ Vh[top_indices, :]
                proj.weight.data = W_approx.to(W.dtype).to(W.device)
                compressed += 1
            except Exception:
                continue

    return compressed


def compress_adaptive_energy(
    model,
    early_threshold: float = 0.96,
    late_threshold: float = 0.98,
) -> dict:
    """Compress using energy-based adaptive rank selection.

    Early layers get more aggressive compression (lower energy threshold),
    late layers are compressed conservatively. Used for large-scale models
    (32B+) where fixed-ratio compression may be too aggressive on some layers.

    Args:
        model: HuggingFace causal LM model
        early_threshold: Energy fraction to retain in early layers
        late_threshold: Energy fraction to retain in late layers

    Returns:
        Dict with compression statistics per layer
    """
    layers = _get_layers(model)
    n_layers = len(layers)
    stats = {}

    for i in range(n_layers):
        # Linear interpolation of threshold based on layer position
        pos = i / (n_layers - 1) if n_layers > 1 else 0.5
        threshold = early_threshold + pos * (late_threshold - early_threshold)

        attn = layers[i].self_attn if hasattr(layers[i], 'self_attn') else getattr(layers[i], 'attn', None)
        if attn is None:
            continue

        layer_stats = {}
        for proj_name in SAFE_PROJECTIONS:
            if not hasattr(attn, proj_name):
                continue

            proj = getattr(attn, proj_name)
            W = proj.weight.data
            if len(W.shape) != 2:
                continue

            try:
                U, S, Vh = torch.linalg.svd(W.float().cpu(), full_matrices=False)

                # Find rank where cumulative energy exceeds threshold
                S_sq = S ** 2
                total_energy = S_sq.sum().item()
                if total_energy == 0:
                    continue

                cumsum = torch.cumsum(S_sq, dim=0)
                energy_ratios = cumsum / total_energy
                mask = energy_ratios >= threshold
                rank = (mask.nonzero()[0].item() + 1) if mask.any() else len(S)
                rank = max(rank, int(len(S) * 0.1))  # Minimum 10% rank

                W_approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]
                proj.weight.data = W_approx.to(W.dtype).to(W.device)

                layer_stats[proj_name] = {
                    'rank': rank,
                    'max_rank': len(S),
                    'ratio': rank / len(S),
                    'energy': energy_ratios[rank - 1].item(),
                    'threshold': threshold,
                }
            except Exception:
                continue

        if layer_stats:
            stats[f'layer_{i}'] = layer_stats

    return stats
