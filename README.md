# Intelligent SVD

Knowledge-preserving compression for large language models via importance-guided SVD of attention projections.

## Key Findings

| Finding | Result | Model |
|---------|--------|-------|
| SVD compression improves TruthfulQA | 39.1% → 44.1% (+5.0%) | Qwen 0.5B |
| Fact retention with CF90 + gentle FT | 60% → 75% (+15% vs freeze-only) | Qwen 0.5B |
| Importance-guided SVD at 50% compression | 73.3% vs 46.7% standard (3x better) | Qwen 0.5B |
| Train-big-compress-smart vs train-small | +55.6% accuracy | Toy transformer |
| SVD + INT8 vs INT8 alone | 85.3% vs 80.0% composite | Qwen 1.5B |
| CF90 at 7B scale | 73% maintained (no degradation) | Qwen 7B |

## Method: CF90 (Compress-Freeze)

1. **Compress** Q, K, O attention projections at 70% rank via SVD
2. **Freeze** 75% of layers (from bottom up)
3. **Fine-tune gently** (1 epoch, 1e-5 LR)

SVD compression removes noise from attention weight matrices while preserving the signal directions most important for factual knowledge. Freezing prevents catastrophic forgetting. The combination achieves better knowledge retention than either technique alone.

### Compression Safety Guide

| Layer Type | Safe to Compress | Notes |
|------------|------------------|-------|
| **Q, K, O projections** | Yes — 70% rank | Main compression target |
| **V projection** | 90-95% only | Marginal gains, high risk below 90% |
| **MLP layers** | Never | Destroys model at any compression level |

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src/ to path (or pip install -e .)
from intelligent_svd import compress_qko, freeze_layers

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# Compress Q, K, O projections (keeps 70% of singular values)
n_compressed = compress_qko(model, ratio=0.7)
print(f"Compressed {n_compressed} matrices")

# Freeze 75% of layers for knowledge preservation
stats = freeze_layers(model, ratio=0.75)
print(f"Frozen {stats['n_frozen']}/{stats['n_layers']} layers")

# Now fine-tune gently on new data
# Use: 1 epoch, lr=1e-5, batch_size=4
```

### Importance-Guided Compression (for aggressive ratios)

```python
from intelligent_svd import compute_importance, compress_qko_importance

# Compute which weights matter most for factual knowledge
importance = compute_importance(model, tokenizer)

# Compress with importance guidance (3x better at 50% compression)
n_compressed = compress_qko_importance(model, importance, ratio=0.5)
```

## Reproduction

### Prerequisites

```bash
pip install torch transformers datasets safetensors lm-eval scipy numpy

# Optional (Apple Silicon inference):
pip install mlx mlx-lm
```

### Run Benchmarks

```bash
# Full benchmark: SVD compression sweep on Qwen-0.5B
python experiments/exp11_full_benchmark.py

# Knowledge protection: CF90 vs baselines
python experiments/exp18_knowledge_protection.py

# Multi-seed validation with statistical testing
python experiments/run_final_validation.py
```

### Platform Notes (Apple Silicon)

- Use **CPU** for PyTorch training/compression (MPS has matmul errors with Qwen)
- Use **MLX** for fast inference (`mlx_lm`)
- Set `HF_HOME` to external storage for large models

## Experiments

| # | Experiment | Key Finding |
|---|-----------|-------------|
| 1 | Factual accuracy | Importance-guided SVD preserves facts better |
| 2 | Overparameterization sweep | 4x overparameterization + compress beats train-small by 55% |
| 3 | Qwen compression | Compression improves accuracy: 66.7% → 80% |
| 4 | Quantization stacking | SVD + INT8 beats INT8-only |
| 5 | Layer sensitivity | Q,K,O safe; V marginal; MLP never |
| 6 | Scale test | Larger compressed > smaller raw (when below ceiling) |
| 7 | SVD + quant benchmark | Validates stacking on real benchmarks |
| 11 | Full benchmark | SVD90 best: +1.4% avg over baseline |
| 16 | Conflicting data | Measures forgetting under data conflict |
| 18 | Knowledge protection | **CF90: 75% retention vs 5% unprotected** |
| 21 | CF90 hierarchical | TruthfulQA +5%, validated at 7B scale |

See [RESULTS.md](RESULTS.md) for detailed numbers and analysis.

## Citation

If you use this work, please cite:

```
@software{intelligent_svd,
  author = {Bryan Sanchez},
  title = {Intelligent SVD: Knowledge-Preserving Compression for LLMs},
  year = {2026},
  url = {https://github.com/SolomonB14D3/intelligent-svd}
}
```

## License

MIT
