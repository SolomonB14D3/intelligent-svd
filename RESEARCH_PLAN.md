# Intelligent Compression Research Plan

## Core Discovery
Training with extra capacity → intelligent compression preserves more knowledge than training at target size directly. This is potentially novel and underexplored.

---

## Research Questions

### Phase 1: Validate Core Finding
**Q1: Does intelligent compression preserve factual accuracy better?**
- Test on QA tasks, not just perplexity
- Compare: standard SVD vs importance-guided at same compression
- Metric: Factual accuracy, not just PPL

**Q2: What's the optimal overparameterization ratio?**
- Test: 1.5x, 2x, 3x, 4x, 5x training size → compress to 1x
- Find diminishing returns point
- Is there a sweet spot?

### Phase 2: Scale Validation
**Q3: Does this work on real pretrained LLMs?**
- Take Qwen 0.5B or similar
- Apply importance-guided compression
- Compare factual accuracy before/after
- Test if fine-tuning with importance awareness helps

**Q4: Can we stack with quantization?**
- Pipeline: Train big → intelligent compress → INT4 quantize
- Compare to: Train small → quantize
- Total compression ratio vs quality

### Phase 3: Curriculum Hypothesis (Your New Idea)
**Q5: Does "truth extraction" curriculum help?**

The idea:
```
1. Train on small, clean data subset (extract core truths)
2. Compress intelligently (lock in truths)
3. Expand to full data (add breadth)
4. Compress again (keep truths, compress new learning)
5. Repeat
```

This is like:
- Learning fundamentals before advanced topics
- Building a skeleton then adding flesh
- Core knowledge as compression-resistant foundation

---

## Experiment Structure

### Experiment 1: Factual Accuracy Deep Test
```
Models: Toy transformer (controlled) + Qwen 0.5B (real)
Task: QA factual accuracy
Compare:
  A) Train small
  B) Train big → standard compress
  C) Train big → intelligent compress
Metrics: PPL, Factual Acc, Reasoning Acc
```

### Experiment 2: Overparameterization Sweep
```
Base size: d=64
Test sizes: d=96, 128, 192, 256, 384
Compress all to: rank 32
Measure: Final PPL and accuracy
Find: Optimal ratio, diminishing returns
```

### Experiment 3: Real LLM Test
```
Model: Qwen2.5-0.5B
Method:
  1. Compute importance from factual probes
  2. Compress Q, K, O projections
  3. Evaluate on TruthfulQA-style questions
Compare: Standard SVD vs Importance-guided
```

### Experiment 4: Full Pipeline
```
Pipeline: Overtrain → Intelligent Compress → Quantize
Compare to: Train small → Quantize
Measure: Total bits, accuracy, speed
```

### Experiment 5: Curriculum Truth Extraction
```
Phase 1: Train on 10% curated "fact-dense" data
Phase 2: Intelligent compress (lock in facts)
Phase 3: Train on remaining 90% data
Phase 4: Intelligent compress again
Compare to: Train on 100% data directly
```

---

## Implementation Order

### Day 1: Core Validation
1. [ ] Experiment 1: Factual accuracy test
2. [ ] Experiment 2: Overparameterization sweep

### Day 2: Scaling
3. [ ] Experiment 3: Real LLM (Qwen)
4. [ ] Experiment 4: Quantization stacking

### Day 3: Novel Hypothesis
5. [ ] Experiment 5: Curriculum truth extraction
6. [ ] Analysis and conclusions

---

## Success Criteria

**Validated if:**
- Intelligent compression preserves 20%+ more factual accuracy than standard
- Overparameterization + compression beats direct training by 10%+
- Effect holds on real LLMs, not just toy models
- Curriculum approach shows additional gains

**Novel contribution if:**
- Importance-guided SVD for fact preservation is not in literature
- Curriculum compression shows improvement
- Combined pipeline (overtrain → intelligent compress → quantize) beats baselines

---

## Files to Create

```
test_exp1_factual.py      # Factual accuracy comparison
test_exp2_sweep.py        # Overparameterization sweep
test_exp3_qwen.py         # Real LLM validation
test_exp4_pipeline.py     # Full compression pipeline
test_exp5_curriculum.py   # Truth extraction curriculum
analyze_results.py        # Aggregate analysis
RESULTS.md                # Findings documentation
```

---

## Current Status

**Completed:**
- [x] Basic importance-guided compression works
- [x] Train-big-compress-smart beats train-small (16% on toy model)
- [x] At 50% compression, intelligent SVD preserves 3x more facts

**Next:**
- [ ] Run Experiment 1 (factual accuracy)
- [ ] Run Experiment 2 (sweep)
