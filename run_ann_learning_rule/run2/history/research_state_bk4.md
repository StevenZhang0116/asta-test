# Research State: Biologically Plausible Learning Algorithm for RNNs

## 1. Research Question & Scope

**Core Question**: What new learning rule can we design for recurrent systems that avoids classical criticisms of BPTT (weight transport, non-locality, non-causal credit assignment) while achieving competitive performance on ML tasks?

**Scope**: 
- Architecture: Start with vanilla RNN, extend to gated (GRU/LSTM)
- Tasks: Short temporal supervised tasks first (sequence length ~10-50)
- Deliverable: A genuinely novel algorithm with clear biological narrative
- Constraints: Must be novel (not previously proposed), must be surprising (not trivial extension)

## 2. Operational Definitions

- **Biologically plausible**: No weight transport, only locally available information for updates, online/near-online, no global error broadcast to all neurons
- **Competitive performance**: Within reasonable range of BPTT on standard benchmarks
- **Novelty**: Not previously described in the literature; not a trivial combination of existing approaches
- **Surprise**: Involves a conceptual insight or unexpected connection

## 2.1 Compute Environment

- **Conda environment**: `panda`
- **GPU**: NVIDIA A100-PCIE-40GB
- **PyTorch**: 2.5.1+cu121
- **Activation**: `conda activate panda`
- All experiments run on GPU unless otherwise noted

## 3. Related Work (COMPLETED — see literature_review.md)

Key existing approaches: RTRL, RFLO, e-prop, ModProp, feedback alignment, equilibrium propagation, dendritic models, predictive coding, forward-forward. None use oscillatory phase for credit routing.

## 4. Algorithm: OPCR (Oscillatory Phase Credit Routing) — FORMALIZED + IMPLEMENTED

**See algorithm_opcr.md for full specification, experiment_opcr.py for implementation.**

### Core Idea
Use oscillatory phase relationships as a temporal addressing system for credit assignment.

### Properties:
- ✅ No weight transport (random feedback B)
- ✅ Locality (only pre/post activity + local phase + local learning signal)
- ✅ Online/causal (forward-time only)
- ✅ Biological correspondence (oscillations, phase-dependent plasticity)
- ✅ Computational complexity: O(N² · M)

## 5. Experimental Plan

### Phase 1: Literature Survey ✅ COMPLETE
### Phase 2: Algorithm Design ✅ COMPLETE (algorithm_opcr.md)
### Phase 3: Implementation & Initial Testing ✅ COMPLETE (experiment_opcr.py)

**Results from initial experiment (3000 training steps):**

| Task | Metric | BPTT | OPCR | Random Baseline |
|------|--------|------|------|-----------------|
| Copy (len=10, delay=10) | Accuracy | 0.208 | 0.181 | 0.125 |
| Adding (len=50) | MSE | 0.167 | 0.170 | 0.167 |

**Interpretation:**
- OPCR shows learning on copy task (acc 0.181 > random 0.125) — **algorithm is functional**
- BPTT baseline is also weak (0.208), suggesting both methods need more training or tuning
- Adding problem: neither method meaningfully learned in 3000 steps — need longer training
- The gap between OPCR and BPTT on copy is modest, which is encouraging for a first attempt

**Issues identified:**
1. Training too short (3000 steps insufficient for these tasks)
2. OPCR loss scale is much higher than BPTT (0.897 vs 0.106) — suggests learning rate or scaling issue
3. The BPTT baseline also underperforms expected levels — may need architecture/hyperparameter tuning
4. Adding problem needs >10K steps to see meaningful learning

### Phase 4: Hyperparameter Tuning & Extended Training (NEXT)

**Key improvements to try:**
1. **Longer training**: 10K-20K steps
2. **Learning rate tuning**: OPCR loss scale suggests lr may be too high or signal scaling issue
3. **Eligibility decay**: lambda=0.95 may decay too fast for delay=10; try 0.98-0.99
4. **Phase bin count**: Try M=4 and M=16 to see effect
5. **Gradient clipping / weight normalization** for stability
6. **Simpler task first**: Copy with delay=5 to confirm learning, then scale up

## 6. Results Summary

- **Literature survey**: 15+ approaches catalogued, gaps identified
- **Algorithm designed**: OPCR fully formalized
- **Initial implementation**: Working, confirmed algorithm learns (copy task acc > random)
- **Key finding**: OPCR is functional but needs optimization — first pass shows learning signal but performance gap with BPTT indicates room for improvement
- **Confidence in H1 (OPCR)**: 50% (slightly down from 55% — it works but unclear if it will reach competitive performance after tuning)

## 7. Open Questions & Confusions

1. **Loss scale discrepancy**: Why is OPCR loss ~8x higher than BPTT? Likely a signal scaling issue in the phase credit computation.
2. **Optimal hyperparameters**: What M, lambda, omega_range, alpha values work best?
3. **Convergence speed**: Is OPCR fundamentally slower to converge, or is this a tuning issue?
4. **Phase stability**: Are the phase dynamics stable over long training? Need to monitor.
5. **Does phase structure actually help?**: Need ablation comparing OPCR to "same architecture without phase gating" (i.e., collapse all phase bins into one).

## 8. Next Steps

1. **IMMEDIATE**: Run extended training (10K+ steps) with tuned hyperparameters — focus on copy task with shorter delay first, then scale
2. **Ablation**: Compare OPCR vs. "no-phase baseline" (same architecture but uniform phase kernels) to isolate the contribution of phase-based credit routing
3. **Analysis**: Visualize phase dynamics and eligibility patterns to understand what the algorithm learns
4. **If performance improves**: Scale to longer sequences and harder tasks
5. **If performance plateaus**: Revisit algorithm design — consider whether phase mechanism needs modification
