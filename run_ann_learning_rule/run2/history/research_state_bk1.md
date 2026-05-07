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
- **Competitive performance**: Within reasonable range of BPTT on standard benchmarks (not necessarily matching it exactly)
- **Novelty**: Not previously described in the literature; not a trivial combination of existing approaches
- **Surprise**: Involves a conceptual insight or unexpected connection

## 3. Related Work

### Known Approaches (from mission.md):
1. **ModProp (Liu et al., 2022)**: Local neuromodulatory signals propagate credit through time via synapse-specific filters interacting with eligibility traces
2. **RFLO (Murray, 2019)**: Locality and causality constraints, random feedback weights instead of symmetric transport
3. **Feedback Alignment (Lillicrap et al., 2016)**: Random fixed backward weights transmit teaching signals

### To Investigate:
- e-prop (Bellec et al., 2020) — eligibility propagation for recurrent networks
- RTRL and approximate RTRL methods
- Predictive coding approaches for temporal learning
- Dendritic error models
- Contrastive Hebbian learning in temporal settings
- Forward-forward algorithm extensions to recurrent networks

## 4. Hypotheses

### H1: Phase-Coupled Eligibility Traces (Confidence: 40%)
**Idea**: Use oscillatory phase relationships between neurons to gate eligibility trace updates. Rather than neuromodulation acting as a simple scalar gate, the relative phase of neural oscillations (theta, gamma) could encode temporal credit assignment information. Synapses are only eligible for update when pre- and post-synaptic neurons are in specific phase relationships, creating a natural temporal windowing mechanism.

**Biological basis**: Theta-gamma phase coding, phase precession in hippocampus, cross-frequency coupling as an information routing mechanism.

**Why potentially novel**: Existing approaches treat eligibility as a simple exponential trace; using oscillatory phase as a structural organizer of credit assignment would be mechanistically different.

### H2: Dendritic Compartment Error Broadcasting (Confidence: 35%)
**Idea**: Exploit the compartmentalized structure of dendrites to compute local prediction errors without global error signals. Each dendritic compartment maintains a predictive model of its expected input; mismatches between predicted and actual input serve as local error signals that drive plasticity. Temporal credit is assigned by having different dendritic compartments operate at different timescales.

**Biological basis**: Active dendrites, dendritic plateau potentials, multi-compartment neuron models, predictive processing.

**Why potentially novel**: While dendritic error models exist, using multi-timescale dendritic compartments as a mechanism for temporal credit assignment in recurrent networks may be new.

### H3: Synaptic Competition via Resource Allocation (Confidence: 30%)
**Idea**: Instead of computing gradients, synapses compete for a limited local "plasticity resource" (analogous to synaptic proteins). Synapses that are active during behaviorally relevant periods claim more resource, leading to consolidation. The resource dynamics naturally implement something like credit assignment without explicit error backpropagation.

**Biological basis**: Synaptic tagging and capture, protein synthesis-dependent plasticity, heterosynaptic competition.

**Why potentially novel**: Recasting learning as resource competition rather than gradient descent; the optimization happens implicitly through competitive dynamics.

## 5. Experimental Designs

### Phase 1: Literature Survey
- Comprehensive survey of existing biologically plausible RNN learning rules
- Identify gaps and opportunities for genuinely novel contributions
- Verify novelty of proposed hypotheses

### Phase 2: Algorithm Design
- Formalize the most promising hypothesis into mathematical rules
- Prove/argue biological plausibility properties
- Analyze computational complexity

### Phase 3: Implementation & Testing
- Implement in PyTorch on vanilla RNN
- Test on: copy task, adding problem, sequential MNIST, Penn Treebank
- Compare against: BPTT, RTRL, e-prop, RFLO

### Phase 4: Analysis & Extension
- Ablation studies on components
- Extend to gated architectures
- Scale to longer sequences

## 6. Results Summary

*No results yet — research beginning.*

## 7. Open Questions & Confusions

1. How to verify true novelty? Need comprehensive literature search before committing to a direction.
2. What is the right trade-off between biological fidelity and task performance?
3. Are there fundamental impossibility results that constrain what local learning rules can achieve?
4. How important is the "surprise" criterion — is a genuinely novel combination of mechanisms acceptable if the combination itself yields unexpected emergent properties?
5. Which temporal tasks best discriminate biologically plausible rules from BPTT?

## 8. Next Steps

- **Immediate**: Conduct literature survey on existing biologically plausible RNN learning rules to (a) understand the landscape, (b) verify novelty of proposed hypotheses, (c) identify under-explored directions
- **Then**: Narrow to 1-2 most promising directions based on novelty + feasibility
- **Then**: Formalize and implement
