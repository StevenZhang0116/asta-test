# Research State: Novel Biologically Plausible Learning Algorithm for RNNs

## 1. Research Question & Scope

**Core Question**: What new learning rule can we design for recurrent systems that avoids BPTT's biological implausibilities (weight transport, non-locality, non-causal credit assignment) while achieving competitive performance on ML tasks?

**Scope**: 
- Architecture: Vanilla RNN first, then gated (GRU/LSTM)
- Tasks: Short-sequence supervised tasks (length 10-50) first
- Constraints: Must be genuinely novel AND surprising (not obvious extension)

## 2. Operational Definitions

- **Biologically plausible**: No weight transport, only local information at synapses, online/near-online, no global error broadcast
- **Competitive performance**: Within reasonable range of BPTT on benchmark tasks
- **Novel**: Not previously described in literature; verified via search
- **Surprising**: Not a trivial combination or minor modification of existing methods

## 3. Related Work

### Key Papers Identified (Literature Search Complete)

**Dendritic / Structural Credit:**
- Sacramento et al. (2018): Dendritic cortical microcircuits approximate backpropagation — uses apical dendrites for error signals in feedforward hierarchies
- Rao et al. (2021): Normative framework for apical dendrite prediction learning — focuses on inter-layer predictions
- Payeur et al. (2021): Burst-dependent synaptic plasticity — burst coding coordinates learning across hierarchical circuits
- Meulemans et al. (2021): Deep Feedback Control — multi-compartment model with local voltage-dependent plasticity

**Temporal Credit Assignment:**
- Bellec et al. (2020): e-prop — eligibility propagation for recurrent spiking networks
- Liu et al. (2022): ModProp — neuromodulatory signals for temporal credit
- Liu et al. (2020): Cell-type-specific modulatory signals for temporal credit
- Barretto-Bittar et al. (2026): Diffusion of neuromodulators for temporal credit
- Temporal Predictive Coding (2026): Extends PC to RNNs for long-range temporal dependencies
- Predictive E-prop (Noè et al., 2026): Combines predictive coding + e-prop

**Predictive / Anticipatory Learning:**
- Saponati & Vinck (2023): Predictive learning rule — neurons predict synaptic input dynamics
- Kriener et al. (2024): ELiSe — dendritic compartments + scaffold for sequence learning

**Phase/Oscillatory:**
- Hasselmo et al. (2002, 2024): SPEAR model — theta phases separate encoding/retrieval
- Maes et al. (2019): Multiplexing neural oscillations for temporal sequence learning

### Gap Analysis (Updated)
The temporal credit assignment space is very active (multiple 2026 papers). Self-prediction within recurrent dynamics is partially explored but not via dendritic compartments. Oscillatory phase-multiplexing for credit assignment (not just encoding/retrieval) remains unexplored.

## 4. Hypotheses

### H1: Dendritic Prediction Error Hypothesis (Confidence: 30% → lowered after lit search)
**Idea**: Neurons predict their own future state via dendritic compartments; prediction errors drive eligibility traces gated by surprise.
**Status**: PARTIALLY NOVEL. The space is converging — temporal predictive coding (2026) and predictive e-prop (2026) are close. The specific "self-prediction" mechanism may still be distinct but is at risk of being scooped.

### H2: Oscillatory Phase-Gated Credit Assignment (Confidence: 45% → raised after lit search)
**Idea**: Theta/gamma oscillations multiplex forward computation and credit assignment through the SAME weights in different phases.
**Status**: MODERATELY NOVEL. No prior work proposes using oscillatory phases specifically for temporal credit assignment through shared weights. Hasselmo's SPEAR is closest but addresses encoding/retrieval, not credit. Mathematical feasibility is the main concern (weight transport in temporal guise).

### H3: Competitive Lateral Inhibition as Implicit Error (Confidence: 25%)
**Status**: Not yet assessed against literature.

### H4: Synaptic Tagging with Stochastic Neuromodulatory Replay (Confidence: 20%)
**Status**: Not yet assessed against literature.

## 5. Experimental Designs

### Phase 1: Literature Validation — COMPLETE for H1, H2
### Phase 2: Mathematical Formalization — NEXT PRIORITY
- Need to formalize H2 (oscillatory phase-gating) mathematically
- Key question: Under what constraints can the same weight matrix propagate credit backward in a different oscillatory phase?
- Consider: transpose approximation, feedback alignment, or emergent alignment

### Phase 3: Implementation & Testing (not yet started)
### Phase 4: Analysis (not yet started)

## 6. Results Summary

### Step 1: Literature Search (Complete)
- H1 (Dendritic Self-Prediction): ~55% novel; converging space, moderate risk
- H2 (Oscillatory Phase-Gating): ~65% novel; more surprising, less explored
- Recommendation: Pursue H2 as primary hypothesis due to higher novelty and surprise factor
- Full report: docs/novelty_assessment.md

## 7. Open Questions & Confusions

1. **Mathematical feasibility of H2**: Can the same weight matrix meaningfully propagate credit backward in a different phase? If it requires near-symmetric weights, it reduces to feedback alignment.
2. **How to handle the "transpose problem" in H2**: During credit phase, information needs to flow "backward" through the same weights — what mathematical trick makes this work?
3. Should we consider a HYBRID approach (H1+H2) where oscillatory phases gate between self-prediction and credit propagation?
4. H3 and H4 haven't been literature-checked yet — worth doing?

## 8. Next Steps

1. **PRIORITY**: Mathematically formalize H2 (oscillatory phase-gated credit assignment)
   - Define the forward dynamics (normal phase)
   - Define the credit dynamics (credit phase)  
   - Show under what conditions the combined system approximates gradient descent
   - Identify the biological constraints and what they cost in terms of approximation quality
2. Consider whether a hybrid H1+H2 approach adds value
3. If H2 math works out: implement and test on copy task
