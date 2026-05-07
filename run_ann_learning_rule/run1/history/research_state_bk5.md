# Research State

## Research Question & Scope

**Goal:** Develop a **new learning algorithm** for recurrent neural networks that is biologically plausible and ML-capable. The deliverable is a novel method — not a comparison study of existing approaches.

**Primary Question:** What new learning rule can we invent for recurrent systems that avoids BPTT's biological implausibilities (weight transport, non-locality, non-causality) while achieving competitive task performance — and what biological principles should ground it?

**Scope:**
- Start with vanilla RNN, then extend to gated architectures (GRU, LSTM)
- Supervised learning first, then RL and unsupervised
- Short sequences initially (10-50 steps) to iterate quickly
- The algorithm needs a clear biological metaphor, not just a mathematical trick
- Must be computationally feasible — not orders of magnitude slower than BPTT

**Environment:**
- All Python experiments must be run in the conda environment: `/home/zihan.zhang/.conda/envs/panda`
- Activate with: `conda activate /home/zihan.zhang/.conda/envs/panda` or use the full Python path `/home/zihan.zhang/.conda/envs/panda/bin/python`
- If any Python packages are missing, install them within this environment (e.g., `/home/zihan.zhang/.conda/envs/panda/bin/pip install <package>`)

## Operational Definitions

- **Weight transport**: backward weights must be exact transposes of forward weights — biologically implausible since synapses are unidirectional
- **Non-locality**: weight updates require information not available at the synapse
- **Non-causality**: future states needed to compute present gradients (BPTT unrolling)
- **Eligibility trace**: local synaptic variable tracking recent co-activity, decays over time
- **Neuromodulation**: diffuse chemical signal (e.g., dopamine) that gates plasticity
- **Feedback alignment**: fixed random backward weights replacing transposed forward weights
- **Online learning rule**: updates depend only on past and present, not future
- **Three-factor rule**: ΔW = f(pre, post, modulator) — general framework encompassing many bio-plausible rules

## Related Work

**Comprehensive literature review completed** — see `docs/literature_review.md` for full details.

Key methods: Feedback Alignment, RFLO, e-prop, KeRNL, OSTL, ModProp.

**Critical finding: NO existing method has been convincingly extended to gated architectures (GRU/LSTM).**

## Hypotheses

**H1 (confidence: 20%, down from 30%):** Plain eligibility traces + random feedback can train vanilla RNNs competitively. *Strongly disconfirmed: Exp02 shows RFLO caps at ~34%, Exp03 shows exact feedback doesn't help (C2=26.5% vs C1=28.4%). The bottleneck is deeper than feedback quality.*

**H2 (confidence: 45%):** Gate activations in GRU/LSTM can serve as local modulatory signals. *Unchanged.*

**H3 (confidence: 80%):** Performance gap vs. BPTT grows with sequence length. *Strongly confirmed across all experiments.*

**H4 (confidence: 25%, down from 55%):** Random feedback is a significant bottleneck. *Disconfirmed by Exp03: exact feedback (C2) performs EQUAL TO or WORSE than random feedback (C1). Random feedback is NOT the problem.*

**H5 (confidence: 55%):** Richer trace dynamics are needed. *Partially supported: full RTRL (C3=40.4%) beats rank-1 trace (C1=28.4%), but the improvement is modest. Even exact traces with SGD-style updates only reach 40%.*

**H7 (confidence: 40%, down from 55%):** The rank-1 trace approximation is the primary bottleneck. *Partially supported but insufficient: full RTRL only adds ~12% accuracy. The bigger issue seems to be the OPTIMIZER — BPTT uses Adam which accumulates gradient statistics, while local rules use raw gradient updates.*

**H8 (new, confidence: 65%):** The main advantage of BPTT is not just the exact gradient, but the OPTIMIZER (Adam) that accumulates second-order statistics. Local learning rules using raw SGD-style updates may be fundamentally limited even with exact gradients. A biologically plausible analog of momentum/Adam may be needed (e.g., synaptic metaplasticity, homeostatic mechanisms).

**H9 (new, confidence: 50%):** C4 (exact feedback + full RTRL) failing at 13.8% indicates a BUG or learning rate mismatch in Exp03. This condition should theoretically approach BPTT. The lr=0.005 may be wrong for the exact feedback condition. Need to rerun with lr sweep.

## Experimental Designs

**Direction A (DONE): RFLO baseline.** ✓
**Direction A2 (DONE): Hyperparameter sweep.** ✓ RFLO caps at ~34.5%.
**Direction A3 (DONE): Ablation.** ✓ Trace > feedback in importance, but even full RTRL only gets 40%.

**Direction A4 (NEXT): Fix C4 bug and test optimizer effect.** C4 should approach BPTT but got 13.8% — likely lr issue. Also: test RFLO/RTRL with momentum or Adam-like updates to see if the optimizer is the real bottleneck.

**Direction B: Design novel rule with richer traces + optimizer-like dynamics.** If H8 is confirmed, the novel algorithm needs:
- Richer trace (higher-rank or RTRL-inspired but cheaper)
- Biologically plausible momentum (e.g., synaptic consolidation, running average of eligibility)
- This maps to a biological narrative of "synaptic metaplasticity"

**Direction C: Gated architecture extension.** After vanilla RNN works.

## Results Summary

### Experiment 01: RFLO vs BPTT Initial (5000 iters, hidden=128, alpha=0.2)
| Task | BPTT | RFLO |
|------|------|------|
| Copy (T=10) | 73.5% | 29.7% |
| Adding (T=30) | 0.142 MSE | 0.174 MSE |

### Experiment 02: Hyperparameter Sweep (10000 iters, hidden=128)
- Best BPTT: alpha=1.0, lr=1e-3 → **100%** accuracy
- Best RFLO: alpha=0.2, lr=0.02 → **34.5%** accuracy
- RFLO saturates regardless of hyperparameters

### Experiment 03: Ablation (10000 iters, hidden=64, trace_decay=0.9)

| Condition | Feedback | Trace | Accuracy | Time |
|-----------|----------|-------|----------|------|
| C1 (RFLO) | Random B | Rank-1 | **28.4%** | 114s |
| C2 | Exact W_out^T | Rank-1 | **26.5%** | 116s |
| C3 | Random B | Full RTRL | **40.4%** | 553s |
| C4 | Exact W_out^T | Full RTRL | **13.8%** | 555s |

**Key findings:**
- **Random feedback is NOT the bottleneck** (C2 ≤ C1). Exact feedback doesn't help — may even hurt slightly.
- **Trace approximation matters somewhat** (C3 > C1 by +12%) but full RTRL still only gets 40%.
- **C4 is anomalous** (13.8%) — should be best but is worst. Likely a learning rate issue or sign error with exact feedback + RTRL.
- **Even with exact traces (RTRL), SGD-style updates only reach 40%** while BPTT+Adam reaches 100%. The optimizer (Adam) may be the dominant factor.

**Code:** `experiments/exp03_ablation.py`
**Plots:** `results/exp03/ablation.png`

## Open Questions & Confusions

1. **[CRITICAL]** Why does C4 (exact feedback + full RTRL) fail? This should be equivalent to BPTT. Either there's a bug in the RTRL implementation, or the learning rate is wrong for this condition. Must debug before drawing conclusions.
2. **[Priority]** Is the optimizer (Adam vs raw SGD) the real secret sauce of BPTT? If we add momentum to local rules, does performance jump?
3. Can "biological momentum" be justified? Options: synaptic metaplasticity, running-average eligibility, homeostatic normalization.
4. If RTRL+SGD only gets 40% but BPTT+Adam gets 100%, what does RTRL+Adam get? (This tests whether the trace quality or optimizer matters more.)
5. Why does exact feedback (C2) perform slightly WORSE than random (C1)? Possible: random feedback provides beneficial regularization/exploration?
6. The biological plausibility constraint may need to be relaxed on the optimizer side — is a local running average of weight changes biologically plausible?

## Suggested Next Step

**Direction A4: Debug C4 and test optimizer effect.** Two sub-experiments:
1. Rerun C4 with lr sweep {0.001, 0.005, 0.01, 0.02} to find the right lr (fix the anomalous result)
2. Add momentum (0.9) to all 4 conditions and see if it closes the gap with BPTT
This will determine whether the optimizer or the gradient quality is the true bottleneck.
