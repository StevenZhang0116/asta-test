# Research State

## Research Question & Scope

**Goal:** Develop a **new learning algorithm** for recurrent neural networks that is biologically plausible and ML-capable. The deliverable is a novel method — not a comparison study.

**Primary Question:** What new learning rule can we invent for recurrent systems that avoids BPTT's biological implausibilities (weight transport, non-locality, non-causality) while achieving competitive task performance?

**Scope:**
- Start with vanilla RNN, then extend to gated architectures (GRU, LSTM)
- Supervised learning first, then RL and unsupervised
- Short sequences initially (10-50 steps) to iterate quickly
- The algorithm needs a clear biological metaphor
- Must be computationally feasible — not orders of magnitude slower than BPTT

**Environment:**
- All Python experiments: `/home/zihan.zhang/.conda/envs/panda/bin/python`
- Install packages: `/home/zihan.zhang/.conda/envs/panda/bin/pip install <package>`

## Operational Definitions

- **Weight transport**: backward weights = exact transposes of forward weights (biologically implausible)
- **Non-locality**: weight updates require non-local information
- **Non-causality**: future states needed for present gradients
- **Eligibility trace**: local synaptic variable tracking recent co-activity
- **Neuromodulation**: diffuse chemical signal gating plasticity
- **Feedback alignment**: fixed random backward weights
- **RTRL**: Real-Time Recurrent Learning — exact forward-mode gradient, O(n^3) memory, online
- **Three-factor rule**: ΔW = f(pre, post, modulator)

## Related Work

See `docs/literature_review.md`. Key methods: Feedback Alignment, RFLO, e-prop, KeRNL, OSTL, ModProp.

## Hypotheses — Updated After Exp04

**H1 (RESOLVED — FALSE):** Plain rank-1 eligibility traces cannot solve memory tasks regardless of optimizer or feedback mechanism. Confirmed definitively: rank-1 + SGD = 28%, rank-1 + Adam = 28%, rank-1 + momentum = 25%. The information content of the trace is the binding constraint.

**H4 (RESOLVED — FALSE):** Random feedback alignment is NOT a significant bottleneck. Full RTRL + random feedback = 100%. Random FB works perfectly when given good gradient information.

**H7 (CONFIRMED — TRUE, confidence: 95%):** The rank-1 trace approximation IS the primary bottleneck. Full RTRL + random FB reaches 100% with simple SGD. The trace quality is everything; the optimizer and feedback mechanism are irrelevant once the trace is exact.

**H8 (RESOLVED — FALSE):** The optimizer is NOT the bottleneck. SGD with full RTRL reaches 100%. Adam provides no benefit when the gradient is correct.

**H10 (new, confidence: 70%):** A biologically plausible approximation to RTRL that is cheaper than O(n^3) but richer than rank-1 is the key to a novel algorithm. Candidates:
- Low-rank RTRL (rank K=5-20 approximation of the n×n Jacobian)
- Sparse RTRL (only track a subset of the influence matrix)
- Block-diagonal RTRL (neurons organized in groups, only track within-group influences)
- Eligibility traces with recurrent dynamics (the trace itself evolves as a small RNN)

**H11 (new, confidence: 60%):** Block-diagonal RTRL maps well to biological column/minicolumn structure and could provide a natural O(n × K²) approximation (where K is column size ~5-20).

## Experimental Designs

**Direction A (DONE): RFLO baseline.** ✓
**Direction A2 (DONE): Hyperparameter sweep.** ✓ RFLO caps at ~34.5%.
**Direction A3 (DONE): Ablation.** ✓ Trace is the bottleneck (Exp03 was partially confounded but Exp04 clarifies).
**Direction A4 (DONE): Optimizer effect.** ✓ Optimizer irrelevant. Full RTRL + random FB = 100% with SGD.

**Direction B (NEXT): Design a computationally tractable approximation to RTRL that is richer than rank-1.** The algorithm design phase. Key insight: we need something between rank-1 (O(n²), fails) and full RTRL (O(n³), perfect but too expensive). Candidates:
1. **Low-rank RTRL**: Approximate the Jacobian dh/dW with a rank-K matrix (K << n). Keep top-K singular vectors of the influence matrix.
2. **Block-diagonal RTRL**: Partition neurons into groups of size K. Only track within-group Jacobians. O(n × K²) memory. Bio metaphor: cortical columns.
3. **Sparse RTRL**: Randomly sample which Jacobian entries to maintain. 
4. **Multi-timescale traces**: K traces with different decay rates, linearly combined.

**Direction C: Gated architecture extension.** After vanilla RNN works.
**Direction D: Scaling.** After method works.

## Results Summary

### Key Finding from Exp04 (DEFINITIVE):

| Trace | Feedback | Optimizer | Accuracy |
|-------|----------|-----------|----------|
| Rank-1 | Random | SGD | 28.1% |
| Rank-1 | Random | Momentum | 24.5% |
| Rank-1 | Random | Adam | 28.4% |
| **Full RTRL** | **Random** | **SGD** | **100%** |
| **Full RTRL** | **Random** | **Momentum** | **100%** |
| **Full RTRL** | **Random** | **Adam** | **100%** |
| Full RTRL | Exact | Adam | 100% |

**Conclusion: The ONLY thing that matters is trace quality. Random feedback works perfectly. Optimizer is irrelevant. Rank-1 traces are fundamentally insufficient.**

### Previous Experiments
- Exp01: Initial RFLO vs BPTT (alpha=0.2 issue identified)
- Exp02: Hyperparameter sweep confirming RFLO ceiling at 34.5%
- Exp03: First ablation (partially confounded by lr issues)
- Exp04: Definitive ablation + optimizer test

**Code:** `experiments/exp01-04_*.py`
**Plots:** `results/exp01-04/`

## Open Questions & Confusions

1. **[Priority]** What is the minimum "rank" or "richness" of trace needed to solve copy task? Test rank-2, rank-5, rank-10, rank-20 approximations to find the threshold.
2. **[Priority]** Can block-diagonal RTRL with block size K=8-16 solve the task? This would be O(n × K²) ≈ O(n × 64-256), much cheaper than O(n³).
3. Why did Exp03's C3 (RTRL+RandomFB) only get 40% while Exp04's identical config gets 100%? Likely a bug in Exp03's implementation (possibly in the `einsum` or the loop where we accumulate `dW_rec` across batch incorrectly). Need to verify.
4. Biological interpretation: what neural structure corresponds to "tracking a low-rank Jacobian"? Possible: lateral inhibition circuits compute projections, top-down feedback provides basis vectors.
5. For block-diagonal: how to handle between-block connections? Ignore them (approximation)? Use random feedback for between-block credit?
6. RTRL is online and causal — so the key remaining biological constraints are: (a) locality of the trace computation, and (b) computational tractability. The feedback can stay random.

## Suggested Next Step

**Direction B: Implement and test low-rank and block-diagonal RTRL approximations on copy task.** Test rank K = {2, 4, 8, 16, 32} to find the minimum rank needed. Also test block-diagonal with block size K = {4, 8, 16}. Compare accuracy vs computational cost. The sweet spot will inform the algorithm design.
