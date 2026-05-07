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

## 3. Related Work (COMPLETED — see literature_review.md for full details)

### Key Existing Approaches:
1. **RTRL & approximations** (Williams & Zipser 1989; UORO, SnAp, KF-RTRL) — exact online but O(n^4) and non-local
2. **RFLO** (Murray, 2019) — local + online + random feedback, but weak temporal credit
3. **e-prop** (Bellec et al., 2020) — eligibility traces + broadcast learning signal, moderate temporal credit
4. **ModProp** (Liu et al., 2022) — neuromodulatory filters on eligibility traces, strong temporal credit
5. **Feedback Alignment** (Lillicrap et al., 2016) — solves weight transport, not temporal credit alone
6. **Equilibrium Propagation** (Scellier & Bengio, 2017) — convergent networks only
7. **Dendritic models** (Sacramento et al., 2018; Payeur et al., 2021) — primarily feedforward
8. **Predictive coding** — temporal extensions underdeveloped
9. **Forward-Forward** (Hinton, 2022) — no RNN extension

### Biological Mechanisms Exploited:
- Eligibility traces (most common)
- Neuromodulation (ModProp)
- Random feedback (RFLO, e-prop)
- Dendritic compartments (Sacramento)
- Burst coding (Payeur)
- Predictive coding (Asabuki & Clopath, 2024)

### KEY GAP IDENTIFIED:
- **Oscillatory phase relationships** have NOT been used as a primary credit assignment mechanism for RNN learning
- **Synaptic resource competition** has NOT been formalized as an RNN learning rule
- **Multi-timescale dendritic temporal credit** is partially explored but not complete

## 4. Hypotheses

### H1: Oscillatory Phase-Based Credit Routing (Confidence: 55% — PROMOTED as primary direction)
**Idea**: Use oscillatory phase relationships between neurons as a temporal addressing system for credit assignment. Different phase offsets selectively route credit to different temporal delays. The key insight: phase provides a *structured temporal reference frame* that eligibility traces lack.

**Specific mechanism (to formalize)**:
- Neurons oscillate at a base frequency (theta ~4-8 Hz analog)
- Phase offset between pre/post-synaptic neurons encodes temporal distance
- Eligibility is gated by phase alignment: synapse only eligible when phase relationship matches the temporal offset of the credit being assigned
- Cross-frequency coupling (theta-gamma) enables multi-scale credit

**Why novel**: No existing rule uses phase as credit routing. Phase has been used for representation (place cells, phase precession) but not for learning rule design.

**Why surprising**: Connects oscillatory neuroscience (a coding mechanism) with credit assignment (a learning problem) — an unexpected bridge.

### H2: Synaptic Competition via Resource Allocation (Confidence: 35% — backup direction)
**Idea**: Learning as competitive resource dynamics rather than gradient descent. Limited plasticity proteins are distributed among synapses based on behavioral relevance.

**Why novel**: Fundamentally different computational paradigm from gradient-based rules.

**Risk**: May not converge to useful solutions without gradient-like dynamics. Needs theoretical grounding.

### H3: Dendritic Multi-Timescale Credit (Confidence: 30% — lower priority)
**Idea**: Different dendritic compartments integrate at different timescales for temporal credit.

**Status**: Partially explored by others; less clearly novel.

## 5. Experimental Plan

### Phase 1: Literature Survey ✅ COMPLETE
- Comprehensive survey conducted (see literature_review.md)
- Novelty of H1 (oscillatory phase credit) verified — genuinely novel
- H2 (resource competition) also verified as novel

### Phase 2: Algorithm Design (CURRENT PHASE)
- **Next step**: Formalize H1 (oscillatory phase credit routing) into a complete mathematical framework
- Define the phase dynamics, eligibility gating rule, and weight update equation
- Prove biological plausibility properties
- Analyze computational complexity
- Compare theoretically to e-prop/ModProp

### Phase 3: Implementation & Testing (UPCOMING)
- Implement in PyTorch on vanilla RNN
- Tasks: copy task, adding problem, sequential MNIST
- Baselines: BPTT, RTRL, e-prop, RFLO

### Phase 4: Analysis & Extension
- Ablations, gated architectures, longer sequences

## 6. Results Summary

- **Literature survey complete**: 15+ approaches catalogued
- **Novelty verified**: H1 (phase-based credit routing) is genuinely novel
- **Primary direction selected**: Oscillatory phase credit routing
- **Key insight**: Phase provides structured temporal reference frame that exponential eligibility traces lack

## 7. Open Questions & Confusions

1. **Mathematical formalization**: How exactly do phase relationships encode temporal credit? What is the update rule?
2. **Phase dynamics**: Should neurons have intrinsic oscillators, or should phase emerge from network dynamics?
3. **Frequency selection**: What determines the base oscillation frequency? How does this relate to task timescale?
4. **Cross-frequency coupling**: How to implement gamma-nested-in-theta for multi-scale credit?
5. **Convergence**: Can we prove this rule converges to useful solutions?
6. **Connection to RTRL**: Is phase-based credit routing an approximation to RTRL, or something fundamentally different?

## 8. Next Steps

1. **IMMEDIATE**: Formalize the oscillatory phase credit routing algorithm — define mathematical framework, write equations, specify all components
2. **THEN**: Implement and test on simple tasks (copy task, adding problem)
3. **THEN**: Compare against baselines and iterate on the design
