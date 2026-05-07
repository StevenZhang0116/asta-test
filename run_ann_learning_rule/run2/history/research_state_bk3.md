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

## 4. Algorithm: OPCR (Oscillatory Phase Credit Routing) — FULLY FORMALIZED

**See algorithm_opcr.md for complete specification.**

### Core Idea
Use oscillatory phase relationships as a temporal addressing system for credit assignment. Phase provides structured temporal reference that eligibility traces lack.

### Key Components:
1. **Phase dynamics**: Each neuron has intrinsic oscillator $\phi_i(t) = \phi_i(t-1) + \omega_i + \alpha h_i(t)$
2. **Phase-indexed eligibility bank**: $e_{ij}^{(m)}(t)$ — M traces per synapse, each gated by a phase bin kernel
3. **Phase-selective credit**: Error signal modulated by neuron's current phase to target specific temporal offsets
4. **Weight update**: $\Delta W_{ij} = \eta \sum_m C_i^{(m)} \cdot e_{ij}^{(m)}$ — temporal pattern matching via phase alignment
5. **Multi-scale**: Cross-frequency coupling (theta-gamma) for hierarchical temporal addressing

### Properties Verified:
- ✅ No weight transport (random feedback B)
- ✅ Locality (only pre/post activity + local phase + local learning signal)
- ✅ Online/causal (forward-time only)
- ✅ No global error broadcast (each neuron gets projected signal)
- ✅ Biological correspondence (oscillations, phase precession, phase-dependent plasticity)
- ✅ Computational complexity: O(N² · M) — modest factor M over e-prop

### Novelty Assessment:
- **Novel**: No prior work uses phase as primary credit routing mechanism in RNN learning
- **Surprising**: Connects oscillatory neuroscience (coding) with temporal credit assignment (learning)
- **Theoretically grounded**: Can be viewed as structured low-rank approximation to RTRL

## 5. Experimental Plan

### Phase 1: Literature Survey ✅ COMPLETE
### Phase 2: Algorithm Design ✅ COMPLETE (algorithm_opcr.md)
### Phase 3: Implementation & Testing (CURRENT PHASE)
- **Next step**: Implement OPCR in PyTorch and test on copy task + adding problem
- Compare against BPTT, e-prop, RFLO
- Ablate: phase gating, M (number of bins), frequency distribution

### Phase 4: Analysis & Extension
- Ablations, gated architectures, longer sequences

## 6. Results Summary

- **Literature survey**: 15+ approaches catalogued, gaps identified
- **Algorithm designed**: OPCR fully formalized with equations, pseudocode, complexity analysis
- **Biological narrative**: Clear mapping from OPCR components to neural oscillations, phase-dependent plasticity, theta-gamma coupling
- **Theoretical analysis**: Connection to RTRL established; expected advantage over exponential eligibility traces for multi-delay tasks

## 7. Open Questions & Confusions

1. ~~Mathematical formalization~~ → RESOLVED (see algorithm_opcr.md)
2. **Implementation choices**: What values of M (phase bins), omega_min/max, alpha, kappa work best?
3. **Frequency learning**: Should frequencies be fixed or learned? The update rule is specified but may be unstable.
4. **Phase stability**: Will activity-coupled phase drift cause problems? May need phase normalization.
5. **Convergence proof**: Can we formally prove convergence? Or at minimum show it empirically converges?
6. **Comparison to ModProp**: ModProp also achieves strong temporal credit — how does OPCR compare on the same tasks?

## 8. Next Steps

1. **IMMEDIATE**: Implement OPCR in PyTorch — vanilla RNN with OPCR learning rule
2. **Test on**: Copy task (sequence length 10-50), adding problem (delay 10-50)
3. **Baselines**: BPTT, random (untrained), and if time permits e-prop
4. **Key question to answer**: Does phase-based credit routing actually work better than exponential eligibility for temporal tasks?
