# Research State: Novel Biologically Plausible Learning Algorithm for RNNs

## 1. Research Question & Scope

**Core Question**: What new learning rule can we design for recurrent systems that avoids BPTT's biological implausibilities (weight transport, non-locality, non-causal credit assignment) while achieving competitive performance on ML tasks?

**Scope**: 
- Architecture: Vanilla RNN first, then gated (GRU/LSTM)
- Tasks: Short-sequence supervised tasks (length 10-50) first
- Constraints: Must be genuinely novel AND surprising (not obvious extension)

## 2. Operational Definitions

- **Biologically plausible**: No weight transport, only local information at synapses, online/near-online, no global error broadcast
- **Competitive performance**: Within reasonable range of BPTT on benchmark tasks (not necessarily matching it exactly)
- **Novel**: Not previously described in literature; verified via search
- **Surprising**: Not a trivial combination or minor modification of existing methods

## 3. Related Work

### Known Approaches (to be distinguished from)
- **BPTT**: Gold standard performance, biologically implausible
- **RTRL**: Local in time but O(n^4) compute; biologically expensive
- **e-prop** (Bellec et al., 2020): Eligibility traces for spiking networks, approximates RTRL
- **ModProp** (Liu et al., 2022): Neuromodulatory signals + eligibility traces for temporal credit
- **RFLO** (Murray, 2019): Random feedback + locality/causality constraints
- **Feedback Alignment** (Lillicrap et al., 2016): Random fixed backward weights
- **UORO/SnAp**: Unbiased/sparse approximations to RTRL
- **Predictive coding**: Hierarchical prediction error minimization
- **Equilibrium propagation**: Energy-based local learning

### Gap Analysis
Most existing approaches address one or two of the biological criticisms but rarely all simultaneously in a way that's both novel and surprising. The space between "local temporal credit" and "structural credit" remains underexplored — most methods handle one well but not both.

## 4. Hypotheses

### H1: Dendritic Prediction Error Hypothesis (Confidence: 40%)
**Idea**: Neurons with multi-compartment dendrites can compute LOCAL temporal prediction errors by comparing apical (top-down prediction) and basal (bottom-up input) compartments. These prediction errors, accumulated as eligibility traces, drive plasticity when modulated by a global "surprise" signal.

**Novelty angle**: Unlike predictive coding (which operates across layers), this operates WITHIN the recurrent dynamics — each neuron predicts its own future state via dendritic compartments, creating a self-supervised temporal signal that doesn't require backward weights.

### H2: Oscillatory Phase-Gated Credit Assignment (Confidence: 35%)
**Idea**: Use neural oscillations (theta/gamma) to temporally multiplex forward computation and credit assignment into different phases. During "forward" phases, the network processes input normally. During "credit" phases, lateral connections propagate local error estimates backward in time using the same physical connections (avoiding weight transport).

**Novelty angle**: The temporal multiplexing means the SAME weights serve dual roles depending on oscillatory phase — no separate backward pathway needed.

### H3: Competitive Lateral Inhibition as Implicit Error (Confidence: 30%)
**Idea**: Use lateral inhibitory circuits to compute an implicit error signal. When a neuron's activity deviates from what the inhibitory population "expects" (based on population statistics), this deviation drives plasticity. Combined with eligibility traces, this creates temporal credit without explicit error backpropagation.

**Novelty angle**: Error is never explicitly computed — it emerges from the dynamics of excitatory-inhibitory balance, which is fundamentally different from all gradient-based approaches.

### H4: Synaptic Tagging with Stochastic Neuromodulatory Replay (Confidence: 25%)
**Idea**: Synapses maintain fast Hebbian tags (marking co-activity). A stochastic neuromodulatory process randomly "replays" recent network states (via brief reactivation bursts), and during replay, tagged synapses are consolidated/modified based on how the replayed state relates to current reward. This performs temporal credit assignment through stochastic sampling rather than deterministic backward computation.

**Novelty angle**: Credit assignment happens through random temporal sampling rather than systematic backward computation — biologically maps to sharp-wave ripple replay events.

## 5. Experimental Designs

### Phase 1: Literature Validation
- Search literature for each hypothesis to confirm novelty
- Identify closest existing work and articulate clear distinction

### Phase 2: Mathematical Formalization
- Write formal update rules for the most promising hypothesis
- Prove/argue convergence properties or at least stability

### Phase 3: Implementation & Testing
- Implement in PyTorch on vanilla RNN
- Test on: copy task, sequential MNIST, adding problem
- Compare against BPTT, RFLO, e-prop baselines

### Phase 4: Analysis
- Ablation studies on components
- Biological plausibility scorecard
- Scaling behavior analysis

## 6. Results Summary

*No experiments run yet.*

## 7. Open Questions & Confusions

1. How to balance novelty requirement with practical performance? More novel ideas tend to be riskier.
2. Which hypothesis to pursue first? H1 (dendritic) seems most grounded in neuroscience but H2 (oscillatory) may be most surprising.
3. How much mathematical rigor is needed before empirical testing?
4. What counts as "within reasonable range" of BPTT? 90%? 80%? 70%?
5. Should we start with a literature search to validate novelty, or with mathematical formalization to see if the ideas are even coherent?

## 8. Next Steps

1. **PRIORITY**: Conduct literature search on H1 and H2 to assess novelty
2. If novel, formalize the most promising hypothesis mathematically
3. Implement and test on copy task as simplest benchmark
