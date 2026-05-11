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
- Sacramento et al. (2018): Dendritic cortical microcircuits approximate backpropagation
- Rao et al. (2021): Normative framework for apical dendrite prediction learning
- Payeur et al. (2021): Burst-dependent synaptic plasticity
- Meulemans et al. (2021): Deep Feedback Control

**Temporal Credit Assignment:**
- Bellec et al. (2020): e-prop — eligibility propagation for recurrent spiking networks
- Liu et al. (2022): ModProp — neuromodulatory signals for temporal credit
- Temporal Predictive Coding (2026): Extends PC to RNNs
- Predictive E-prop (Noè et al., 2026): Combines predictive coding + e-prop

**Phase/Oscillatory:**
- Hasselmo et al. (2002, 2024): SPEAR model — theta phases separate encoding/retrieval
- Maes et al. (2019): Multiplexing neural oscillations for temporal sequence learning

## 4. Hypotheses

### H2: Oscillatory Phase-Gated Credit Assignment — TESTED, PARTIALLY FAILED
**Idea**: Theta/gamma oscillations multiplex forward computation and credit assignment through the SAME weights in different phases.
**Status**: Implemented and tested. The basic version (using W without transpose for credit propagation) performs WORSE than feedback alignment with random B (62.4% vs 79.7% accuracy). This is a critical negative result.

**Key finding**: Using the forward weight matrix W for backward credit propagation creates *destructive interference* — the structured information in W that's useful for forward computation actually *hinders* credit propagation compared to random feedback. This is counterintuitive but important.

**OPCA-alpha variant** (learned damping): Achieves 80.6% accuracy by learning to reduce alpha from 0.9 to 0.047, effectively abandoning long-range temporal credit in favor of nearly-local learning. This suggests the "oscillatory credit propagation" part of H2 is not working as intended.

### H1: Dendritic Prediction Error Hypothesis (Confidence: 30%)
**Status**: Not tested experimentally. Given H2's failure, this may be worth revisiting.

### H3: Competitive Lateral Inhibition as Implicit Error (Confidence: 25%)
**Status**: Not yet assessed.

### H4: Synaptic Tagging with Stochastic Neuromodulatory Replay (Confidence: 20%)
**Status**: Not yet assessed.

### NEW HYPOTHESES EMERGING FROM H2 RESULTS:

### H5: Oscillatory Phase-Gated Credit with ASYMMETRIC Weights (Confidence: 40%)
**Insight from H2**: The problem isn't the oscillatory mechanism itself — it's that W is not a good approximation of W^T. But what if during the credit phase, the network uses a *learned but separate* backward weight matrix B that naturally aligns with W through training (like feedback alignment, but oscillatory-phase-gated and with alignment pressure)?

### H6: Predictive Self-Correction with Phase-Modulated Consolidation (Confidence: 45%)
**Insight combining H1+H2**: Instead of propagating credit backward through W, use FORWARD predictions of future error (self-prediction from H1) combined with oscillatory phase gating of WHEN plasticity occurs (from H2). Each neuron predicts whether it will contribute to future error, and the oscillatory cycle determines when these predictions are evaluated and turned into weight changes.

**Why this might be different**: It avoids the W-vs-W^T problem entirely by never backpropagating. Instead, credit is assigned via forward predictive signals that are consolidated during specific oscillatory windows.

## 5. Experimental Results

### Experiment 1: OPCA Copy Task (Step 2)

**Setup**: Copy task (8 bits, 10-step delay), vanilla RNN 128 hidden, 5000 iterations, Adam optimizer

| Method | Best LR | Final MSE | Bit Accuracy | Converge@0.01 |
|--------|---------|-----------|-------------|---------------|
| BPTT | 0.001 | 0.0004 | 100.0% | iter 650 |
| OPCA-alpha | 0.001 | 0.1416 | 80.6% | N/A |
| FA | 0.001 | 0.1453 | 79.7% | N/A |
| OPCA | 0.005 | 0.2339 | 62.4% | N/A |
| RFLO | 0.001 | 0.2679 | 51.6% | N/A |

**Key findings**:
1. BPTT >> all biologically plausible methods (huge gap)
2. OPCA (using W for credit) < FA (using random B) — using forward weights for credit is WORSE
3. OPCA-alpha learns to reduce alpha to ~0.047, effectively abandoning temporal credit propagation
4. All bio-plausible methods stuck around 60-80% accuracy, none converge

**Interpretation**: The "same weights for both directions" idea in H2 doesn't work because W is not a good surrogate for W^T. The oscillatory framing is biologically elegant but mathematically doesn't solve the weight transport problem.

## 6. Open Questions & Confusions

1. **Why does W perform worse than random B?** In feedback alignment theory, random B works because learning aligns W toward B over time. But W itself should at least carry SOME useful structural info about the gradient direction — why is it actively worse?
2. **Is the alpha→0 result an artifact of the task?** The copy task requires long-range memory but perhaps the gradient landscape is such that only the most recent time steps matter for output weight learning.
3. **Should we pivot to H6 (predictive + oscillatory consolidation)?** This avoids the weight transport problem entirely.
4. **Would H2 work better on tasks with shorter temporal dependencies?** Perhaps for very short sequences, W ≈ W^T approximation is better.

## 7. Next Steps

1. **PRIORITY**: Develop and test H6 (Predictive Self-Correction with Phase-Modulated Consolidation)
   - This avoids backward propagation entirely, using forward prediction + oscillatory gating
   - Key idea: neurons learn to predict their contribution to future error using LOCAL information
   - Oscillatory phase determines when these predictions are evaluated/consolidated
2. Alternatively, investigate WHY W performs worse than random B — analyze the angle between W*c and W^T*c for credit vectors
3. Consider whether a modified task (shorter delay, different structure) would reveal different dynamics
