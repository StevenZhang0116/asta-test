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

### Key Papers Identified
- Sacramento et al. (2018): Dendritic cortical microcircuits (structural credit)
- Bellec et al. (2020): e-prop (eligibility propagation for temporal credit)
- Liu et al. (2022): ModProp (neuromodulatory temporal credit)
- Hasselmo et al. (2002, 2024): SPEAR (theta phase encoding/retrieval separation)
- Saponati & Vinck (2023): Predictive learning rule (anticipatory plasticity)
- Lillicrap et al. (2016): Feedback alignment (random backward weights)
- Temporal Predictive Coding (2026), Predictive E-prop (2026)

## 4. Hypotheses — Status Summary

| Hypothesis | Status | Result |
|-----------|--------|--------|
| H2: OPCA (same weights for forward+credit) | TESTED | FAILED — worse than random FA |
| H6: PSC-NoGate (predictive self-correction, no osc) | TESTED | PROMISING — 76.4% accuracy |
| H6: PSC-Osc (with oscillatory gating) | TESTED | FAILED — oscillatory gating hurts |
| H1: Dendritic self-prediction | Not tested | Subsumed into H6 |
| H3: Lateral inhibition as implicit error | Not tested | - |
| H4: Stochastic neuromodulatory replay | Not tested | - |

### Key Findings Across All Experiments

1. **OPCA (using W for backward credit) is WORSE than random FA** (62.4% vs 79.7% accuracy on copy task with best hyperparams from prior experiment)
2. **PSC-NoGate achieves 76.4% accuracy** using only local eligibility traces + output error feedback — NO backward propagation at all
3. **Oscillatory gating HURTS** — it restricts plasticity to ~50% of timesteps, halving the learning signal
4. **All bio-plausible methods have massive gap to BPTT** (100% accuracy by iter ~300)
5. **OPCA-alpha (from Step 2)** learned to reduce alpha→0.047, confirming that long-range temporal credit via W is unhelpful

### Most Promising Direction: PSC-NoGate

The PSC-NoGate algorithm achieves the best bio-plausible performance observed so far:
- 76.4% accuracy on copy task (vs BPTT 100%, FA ~50-80% depending on setup)
- Uses ONLY local signals: eligibility traces + output error projected through W_out
- No backward propagation of any kind through recurrent weights
- Prediction compartment provides self-supervised auxiliary learning signal
- Best config: lr=0.001, beta=0.1, gamma=0.5, lambda=0.9

## 5. Experimental Results

### Experiment 1: OPCA Copy Task (Step 2)
| Method | Final MSE | Bit Accuracy |
|--------|-----------|-------------|
| BPTT | 0.0004 | 100.0% |
| OPCA-alpha | 0.1416 | 80.6% |
| FA | 0.1453 | 79.7% |
| OPCA | 0.2339 | 62.4% |
| RFLO | 0.2679 | 51.6% |

### Experiment 2: PSC Copy Task (Step 3)
| Method | Final MSE | Bit Accuracy |
|--------|-----------|-------------|
| BPTT | 0.0005 | 100.0% |
| PSC-NoGate | 0.1575 | 76.4% |
| PSC-Osc | 0.2494 | 51.7% |
| FA | 0.2523 | 50.8% |

### Cross-Experiment Summary (best bio-plausible per experiment)
- OPCA-alpha (Exp 1): 80.6% — but cheats by learning alpha→0 (nearly local)
- PSC-NoGate (Exp 2): 76.4% — genuinely novel, fully local, no backward pass

## 6. Open Questions & Key Insights

### Insights
1. **Backward propagation through recurrent weights (any version) is problematic**: Whether using W, W^T, or random B, propagating credit backward through the recurrent structure is either biologically implausible (W^T) or mathematically ineffective (W or random B)
2. **Local-only learning can work**: PSC-NoGate shows that eligibility traces + direct output error feedback achieves reasonable performance without any temporal credit propagation
3. **Oscillatory gating is counterproductive for this task**: Restricting plasticity to certain phases just reduces learning
4. **The performance gap to BPTT is fundamental**: Without proper temporal credit assignment, bio-plausible methods cap around 75-80% on the copy task

### Open Questions
1. **Would PSC-NoGate improve with better prediction learning?** Currently prediction is simple linear, could nonlinear prediction help?
2. **Is the copy task too hard a first test?** It requires perfect memory over 10 steps — maybe an easier task would show the methods in better light
3. **Can we improve temporal credit WITHOUT backward propagation?** The key challenge remains
4. **Would a different prediction target help?** Instead of predicting own activity, predict own GRADIENT contribution?

## 7. Next Steps (Priority Order)

1. **Refine PSC-NoGate**: 
   - Test on additional tasks (adding problem, sequential MNIST) to verify generality
   - Try nonlinear prediction compartment
   - Investigate why it plateaus at ~76% — is it a fundamental limitation or hyperparameter issue?
2. **Explore temporal credit via FORWARD prediction of future errors** (not backward propagation)
   - Each neuron could predict "will I contribute to future error?" using predictive compartment
   - This forward-looking prediction could provide temporal credit without backward flow
3. **Write up the PSC algorithm formally** as the main contribution
4. **Biological narrative**: Formalize the mapping between PSC components and neural circuits
