# Intelligent SVD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18718545.svg)](https://doi.org/10.5281/zenodo.18718545)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Knowledge-preserving compression for large language models via importance-guided SVD of attention projections.**

## What It Does

CF90 (Compress-Freeze) compresses Q, K, O attention projections via SVD while freezing most layers, protecting factual knowledge during subsequent fine-tuning. An importance-guided variant uses gradient information to preserve critical singular values at aggressive compression ratios.

See our paper for experimental results and analysis.

## Quick Start

```python
from transformers import AutoModelForCausalLM
from intelligent_svd import apply_cf90

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float32)
stats = apply_cf90(model, ratio=0.7, freeze_ratio=0.75)
# Now fine-tune gently: 1 epoch, lr=1e-5, batch_size=4
```

Or step by step:

```python
from intelligent_svd import compress_qko, freeze_layers

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float32)

# Step 1: Compress Q, K, O projections (keeps 70% of singular values)
n_compressed = compress_qko(model, ratio=0.7)

# Step 2: Freeze 75% of layers from the bottom up
stats = freeze_layers(model, ratio=0.75)

# Step 3: Fine-tune gently on your data
```

### Importance-Guided Compression

When compressing below 70%, the importance-guided variant uses gradient information to decide which singular values to keep:

```python
from intelligent_svd import compute_importance, compress_qko_importance

importance = compute_importance(model, tokenizer)
n_compressed = compress_qko_importance(model, importance, ratio=0.5)
```

### Compression Safety Guide

| Layer Type | Safe to Compress | Notes |
|------------|------------------|-------|
| **Q, K, O projections** | Yes — 70% rank | Main compression target |
| **V projection** | 90-95% only | High risk below 90% |
| **MLP layers** | Never | Destroys model at any compression level |

## Model Compatibility

Works on any HuggingFace causal LM with standard attention layouts (`model.model.layers[i].self_attn.{q,k,o}_proj`). GPT-2-style models using `model.transformer.h` are also supported.

**Validated:** Qwen2.5 (0.5B–32B), Llama 2 (7B)

## Install

```bash
pip install torch transformers datasets safetensors lm-eval scipy numpy

# Optional (Apple Silicon inference):
pip install mlx mlx-lm
```

### Quick Demo

```bash
# Llama 2 7B (default)
python examples/quick_demo.py

# Smaller model, faster (~5 min)
python examples/quick_demo.py --model Qwen/Qwen2.5-0.5B
```

### Validation with rho-eval

```bash
pip install intelligent-svd[audit]  # installs rho-eval>=2.2
```

```python
from intelligent_svd import validate_compression

result = validate_compression(
    model, tokenizer,
    ratio=0.7, freeze_ratio=0.75,
    behaviors="factual,toxicity,sycophancy",
)
print(f"Passed: {result.passed}")
```

## Citation

```bibtex
@software{intelligent_svd,
  author = {Bryan Sanchez},
  title = {CF90: Knowledge-Preserving Compression for LLMs via SVD and Layer Freezing},
  year = {2026},
  doi = {10.5281/zenodo.18718545},
  url = {https://github.com/SolomonB14D3/intelligent-svd}
}
```

### Related Projects

- [rho-eval](https://github.com/SolomonB14D3/knowledge-fidelity) — Behavioral auditing toolkit for LLMs
- [confidence-cartography](https://github.com/SolomonB14D3/confidence-cartography) — Teacher-forced confidence as a false-belief sensor

## License

MIT
