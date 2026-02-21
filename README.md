# Intelligent SVD

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18718545.svg)](https://doi.org/10.5281/zenodo.18718545)

Knowledge-preserving compression for large language models via importance-guided SVD of attention projections.

## Key Findings

| Finding | Result | Model |
|---------|--------|-------|
| CF90 knowledge protection (p=0.0072) | 79% retention vs 65% freeze-only | Qwen 0.5B |
| CF90 beats LoRA on fact retention | 73% vs 68% (LoRA r=8) | Qwen 0.5B |
| SVD + INT8 vs INT8 alone | 85.3% vs 80.0% composite (+5.3%) | Qwen 1.5B |
| CF90 + INT8 full pipeline | 72% retention vs 58% unprotected | Qwen 0.5B |
| Importance-guided SVD at 50% compression | 73.3% vs 46.7% standard (3x better) | Qwen 0.5B |
| CF90 at 7B scale | 73% maintained (no degradation) | Qwen 7B |
| CF90 on Llama 2 7B (cross-arch) | 78% retention, 25% repetition (vs 40% baseline) | Llama 2 7B |
| CF90 + INT8 on Llama | 77% retention through quantization | Llama 2 7B |

## Method: CF90 (Compress-Freeze)

1. **Compress** Q, K, O attention projections at 70% rank via SVD
2. **Freeze** 75% of layers (from bottom up)
3. **Fine-tune gently** (1 epoch, 1e-5 LR)

SVD compression removes noise from attention weight matrices while preserving the signal directions most important for factual knowledge. Freezing prevents catastrophic forgetting. The combination achieves better knowledge retention than either technique alone.

### Important: Scale-Dependent Generation Quality

CF90 preserves factual knowledge at all scales, but **degrades generation quality (fluency, coherence) on models below ~1B parameters**. At 0.5B, CF90 increases 3-gram repetition from 5% to 33%. This does not occur at 7B+ where IFEval (instruction following) remains at 95% and HumanEval (code) at 98%.

**Recommendation**: Use CF90 on 7B+ models for production. On smaller models, use it only when fact retention matters more than generation quality (e.g., knowledge distillation, fact-checking, retrieval).

### Compression Safety Guide

| Layer Type | Safe to Compress | Notes |
|------------|------------------|-------|
| **Q, K, O projections** | Yes — 70% rank | Main compression target |
| **V projection** | 90-95% only | Marginal gains, high risk below 90% |
| **MLP layers** | Never | Destroys model at any compression level |

## Quick Start

```python
from transformers import AutoModelForCausalLM
from intelligent_svd import apply_cf90

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float32)
stats = apply_cf90(model, ratio=0.7, freeze_ratio=0.75)
# Compressed 72 matrices, frozen 21/28 layers
# Now fine-tune gently: 1 epoch, lr=1e-5, batch_size=4
```

Or step by step if you want more control:

```python
from intelligent_svd import compress_qko, freeze_layers

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", torch_dtype=torch.float32)

# Step 1: Compress Q, K, O projections (keeps 70% of singular values)
n_compressed = compress_qko(model, ratio=0.7)

# Step 2: Freeze 75% of layers from the bottom up
stats = freeze_layers(model, ratio=0.75)

# Step 3: Fine-tune gently on your data
# Use: 1 epoch, lr=1e-5, batch_size=4
```

### Importance-Guided Compression (for aggressive ratios)

When compressing below 70%, standard SVD starts losing facts. The importance-guided variant uses gradient information to decide which singular values to keep, preserving 3x more factual knowledge at 50% compression.

How it works: run a few forward+backward passes on factual probe prompts (e.g., "The capital of France is Paris."), accumulate absolute gradients for each weight in Q/K/O projections, then during SVD, score each singular value by its contribution to those high-gradient directions instead of just keeping the largest ones by magnitude.

```python
from intelligent_svd import compute_importance, compress_qko_importance

# Compute gradient-based importance (7 built-in factual probes, or pass your own)
importance = compute_importance(model, tokenizer)

# You can also pass domain-specific probes:
# importance = compute_importance(model, tokenizer, prompts=[
#     "Aspirin inhibits cyclooxygenase.",
#     "TCP uses a three-way handshake.",
# ])

# Compress with importance guidance (3x better at 50% compression)
n_compressed = compress_qko_importance(model, importance, ratio=0.5)
```

### Model Compatibility

CF90 works on any HuggingFace causal LM that stores attention layers in `model.model.layers[i].self_attn.{q,k,o}_proj` (the standard layout for Qwen, Llama, Mistral, and most recent architectures). GPT-2-style models using `model.transformer.h` are also supported.

Validated on:
- **Qwen2.5**: 0.5B, 1.5B, 7B, 32B (full experimental suite)
- **Llama 2**: 7B (Experiments C + D, 3 seeds each)

The architecture hooks should work on Mistral without changes (same `model.model.layers` layout). If you test on another model family, please open an issue with results.

## Reproduction

### Prerequisites

```bash
pip install torch transformers datasets safetensors lm-eval scipy numpy

# Optional (Apple Silicon inference):
pip install mlx mlx-lm
```

### Quick Demo

See the key findings yourself in ~30 minutes:

```bash
# Llama 2 7B (default) — knowledge protection + repetition reduction
python examples/quick_demo.py

# Smaller model, faster (~5 min)
python examples/quick_demo.py --model Qwen/Qwen2.5-0.5B
```

This runs both experiments end-to-end: CF90 knowledge protection under conflicting fine-tuning (78% vs 32% retention) and the generation quality test showing repetition drops from 40% to 25% at 7B scale.

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

- Use **CPU** for PyTorch training/compression (MPS has matmul errors with Qwen; also produces NaN gradients when training with frozen layers on any model)
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
| A | SVD90 vs baseline (5 seeds) | No significant benchmark difference |
| B | CF90 vs LoRA (3 seeds) | CF90 73% > LoRA-r8 68% |
| C | CF90 protection (5 seeds) | **p=0.0072**, CF90 79% vs freeze 65% |
| D | CF90 + INT8 full pipeline | 72% retention through quant; generation quality tradeoff |
| C-Llama | CF90 protection on Llama 2 7B (3 seeds) | 68% CF90 vs 65% freeze-only; cross-arch validation |
| D-Llama | CF90 + INT8 on Llama 2 7B (3 seeds) | 78% retention; CF90 reduces repetition from 40% to 25% |

See [RESULTS.md](RESULTS.md) for detailed numbers and analysis.

## Connection to Confidence Cartography

This work pairs with [Confidence Cartography](https://github.com/SolomonB14D3/confidence-cartography), which maps where a model is uncertain by measuring the probability it assigns to its own tokens (teacher-forced confidence). Together they form a two-stage pipeline:

1. **Cartography** identifies *which* weight regions encode uncertain or contested knowledge (ρ = 0.652 correlation with human false-belief prevalence, p = 0.016).
2. **Intelligent SVD** determines *how* to protect those weights during fine-tuning, compressing the noise out of attention projections so that downstream updates cannot overwrite factual signal.

The practical implication: run confidence cartography first to locate fragile knowledge, then apply CF90 to lock it in before any further training. This converts a 3–5× compute overhead into a targeted intervention rather than a blanket freeze.

## Citation

If you use this work, please cite:

```
@software{intelligent_svd,
  author = {Bryan Sanchez},
  title = {CF90: Knowledge-Preserving Compression for LLMs via SVD and Layer Freezing},
  year = {2026},
  doi = {10.5281/zenodo.18718545},
  url = {https://github.com/SolomonB14D3/intelligent-svd}
}
```

## License

MIT
