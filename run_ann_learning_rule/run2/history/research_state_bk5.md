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

## 4. Algorithm Status

### OPCR v1/v2 — FAILED TO DEMONSTRATE PHASE BENEFIT

**Critical finding from experiment_opcr_v2.py:**

| Method | Delay=5 Acc | Delay=10 Acc |
|--------|-------------|--------------|
| BPTT | 0.873 | 0.508 |
| OPCR (with phase) | 0.261 | 0.254 |
| No-Phase ablation | 0.271 | 0.260 |
| Random baseline | 0.125 | 0.125 |

**Key conclusions:**
1. OPCR learns (acc > random) but is far below BPTT
2. **Phase selectivity does NOT help** — the no-phase ablation slightly outperforms OPCR
3. The minimal learning comes from the basic eligibility + feedback alignment structure
4. The softmax-normalized phase kernels may be *hindering* learning by splitting eligibility across bins

### Diagnosis of Failure:

**Why phase isn't helping (hypothesis):**
1. **Phase as noise**: The phase gating fragments eligibility across M bins, effectively reducing the signal-to-noise ratio. Instead of concentrating eligibility, it disperses it.
2. **Phase-credit mismatch**: The phase at the time of error may not correctly "address" the phase at the time of activity — the temporal encoding may not be precise enough with these dynamics.
3. **Fixed frequencies don't match task structure**: The oscillator frequencies are distributed uniformly and don't adapt to the task's actual temporal structure.
4. **Softmax kills gradients**: Softmax normalization means most bins get near-zero eligibility, potentially zeroing out the correct one.

### Path Forward: Algorithm Redesign

Need to fundamentally rethink how phase helps. Two options:

**Option A: Phase as temporal filter, not address**
Instead of using phase to *select* which bin gets credit, use phase to create *frequency-selective filters* on the eligibility trace. Different neurons resonate at different frequencies and thus naturally accumulate credit at different temporal scales. This is closer to ModProp but with oscillatory structure.

**Option B: Phase-locked accumulation**
Instead of splitting eligibility into bins, use a single eligibility trace but gate its *accumulation rate* by phase. When pre-post phase alignment is good, eligibility accumulates faster. This concentrates rather than disperses the signal.

**Option C: Abandon phase for credit, use it for temporal segmentation**
Use oscillatory phase not for routing credit but for segmenting temporal sequences into chunks. Within each chunk, use standard eligibility. Between chunks, use phase transitions to reset/consolidate. This is more biologically motivated (theta-reset in hippocampus).

## 5. Experimental Plan

### Phase 1: Literature Survey ✅ COMPLETE
### Phase 2: Algorithm Design ✅ v1 COMPLETE (failed to validate)
### Phase 3: Initial Testing ✅ COMPLETE (weak results)
### Phase 4: Extended Testing + Ablation ✅ COMPLETE (phase not helping)
### Phase 5: Algorithm Redesign (CURRENT)
- Redesign the role of phase in the algorithm based on failure analysis
- Implement and test revised algorithm

## 6. Results Summary

| Step | What | Outcome |
|------|------|---------|
| 1 | Literature survey | 15+ methods catalogued; phase credit routing confirmed novel |
| 2 | OPCR formalization | Complete mathematical spec |
| 3 | Initial testing (3K steps) | Weak learning signal detected |
| 4 | Extended testing (10K steps) + ablation | **Phase does NOT help**; OPCR far below BPTT |

**Confidence in H1 (OPCR as originally conceived): 15%** — The phase-as-address mechanism does not work as designed. The core idea may still have merit but needs fundamental revision.

## 7. Open Questions & Confusions

1. **Is the problem phase or the whole approach?** The no-phase version also underperforms BPTT massively (0.27 vs 0.87). This suggests the feedback alignment + eligibility structure itself is weak, not just the phase component.
2. **Why does basic eligibility + FA underperform so much?** e-prop/RFLO papers report better results. Our implementation may have bugs or our hyperparameters may be bad.
3. **Should we pivot entirely?** Or fix the baseline (eligibility + FA) first, then re-add phase?
4. **Is the copy task too hard for this architecture?** Maybe test on a simpler task where eligibility + FA is known to work.

## 8. Next Steps

1. **IMMEDIATE**: Diagnose why our no-phase baseline underperforms. Compare implementation against e-prop. If we can get the baseline (eligibility + FA without phase) to work well, then phase improvements have a chance. Fix the foundation first.
2. **THEN**: Redesign phase mechanism using Option B (phase-locked accumulation) or Option C (temporal segmentation)
3. **THEN**: Re-run experiments with corrected baseline
