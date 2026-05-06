# Research State

## Research Question & Scope

**Goal:** Develop a **new learning algorithm** for recurrent neural networks that is biologically plausible and ML-capable. The deliverable is a novel method — not a comparison study of existing approaches. Existing methods (RFLO, ModProp, etc.) are starting points and inspiration, not endpoints.

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

Key methods surveyed:
1. **Feedback Alignment** (Lillicrap 2016): random backward weights solve weight transport in feedforward nets.
2. **RFLO** (Murray 2019): online, local, causal rule for vanilla RNNs. Rank-1 approx to influence matrix.
3. **e-prop** (Bellec 2020): eligibility propagation for spiking RNNs. Strong bio metaphor.
4. **KeRNL** (Roth 2019): learned temporal kernel. Moderate performance.
5. **OSTL** (Bohnstingl 2022): forward-mode + random feedback. O(n²) memory.
6. **ModProp** (Liu 2022): per-synapse learned temporal filters. Most expressive but complex.

**Critical finding: NO existing method has been convincingly extended to gated architectures (GRU/LSTM).**

## Hypotheses

**H1 (confidence: 45%, down from 60%):** A learning rule combining eligibility traces + neuromodulatory gating + random feedback can train vanilla RNNs on short-sequence tasks within reasonable range of BPTT. *Weakened by Exp01 results: RFLO achieves only 29.7% on copy task vs BPTT's 73.5%. The gap is larger than expected even for seq_len=10. Plain eligibility traces + random feedback are insufficient for the copy task.*

**H2 (confidence: 45%):** Gate activations in GRU/LSTM can serve as local modulatory signals. *Unchanged — not yet tested.*

**H3 (confidence: 70%, up from 65%):** Performance gap vs. BPTT grows with sequence length. *Confirmed directionally: even at seq_len=10 (copy) and 30 (adding), the gap is significant.*

**H4 (confidence: 55%):** Random feedback alignment + eligibility traces work better together than either alone. *Not yet directly tested via ablation.*

**H5 (confidence: 50%):** Multi-timescale eligibility traces can bridge RFLO and ModProp. *More important now given RFLO's poor copy task performance — single timescale may be the bottleneck.*

**H6 (new, confidence: 40%):** BPTT itself may need more iterations or hyperparameter tuning to solve these tasks well with a leaky vanilla RNN — the 73.5% BPTT accuracy on copy task suggests the architecture/hyperparams may also be limiting. Need to verify BPTT can reach ~100% before concluding RFLO's gap is fundamental.

## Experimental Designs

**Direction A (DONE): Implement RFLO baseline on copy task and adding problem.** ✓ Completed.

**Direction A2 (NEXT): Debug and improve baselines.** BPTT only reaches 73.5% on copy task — likely needs more training or tuning. RFLO's poor performance (29.7%) could be due to: (a) learning rate, (b) the rank-1 trace approximation, (c) random feedback alignment struggling with this task. Run hyperparameter sweep and longer training to establish true ceilings.

**Direction B: Design "Gated-RFLO" — extend RFLO to GRU.** Postponed until vanilla RNN baselines are solid.

**Direction C: Multi-timescale eligibility traces.** May help RFLO's copy task performance by allowing different timescales for different temporal dependencies.

**Direction D: Ablation and scaling.** After baselines are solid.

## Results Summary

### Experiment 01: RFLO vs BPTT (5000 iterations, hidden=128, alpha=0.2)

| Task | Method | Metric | Result | Time |
|------|--------|--------|--------|------|
| Copy (T=10) | BPTT (lr=1e-3) | Accuracy | 0.735 | 91.9s |
| Copy (T=10) | RFLO (lr=1e-2) | Accuracy | 0.297 | 98.2s |
| Copy (T=10) | Random | Accuracy | 0.125 | - |
| Adding (T=30) | BPTT (lr=1e-3) | MSE | 0.142 | 77.3s |
| Adding (T=30) | RFLO (lr=1e-2) | MSE | 0.174 | 52.7s |
| Adding (T=30) | Random | MSE | 0.167 | - |

**Key observations:**
- BPTT moderately solves copy task (73.5%) but hasn't converged — likely needs more iterations or tuning
- RFLO barely beats random on copy task (29.7% vs 12.5%) — large gap vs BPTT
- Neither method has clearly solved the adding problem — BPTT slightly below random baseline (0.142 vs 0.167) but RFLO is at random level (0.174)
- Training times are comparable (RFLO slightly faster on adding due to no backprop)

**Code:** `experiments/exp01_rflo_vs_bptt.py`
**Plots:** `results/exp01/learning_curves.png`

## Open Questions & Confusions

1. **[Priority]** Why is BPTT only at 73.5% on copy task with seq_len=10? This should be solvable near-perfectly. Suspect: alpha=0.2 may be too low (information decays too fast), or need more training iterations, or learning rate needs tuning.
2. **[Priority]** Is RFLO's poor copy task performance due to the learning rule itself or hyperparameters? Need to sweep lr, alpha, and hidden size.
3. The adding problem results are puzzling — BPTT at 0.142 MSE is barely below the random baseline of 0.167. The architecture may not be well-suited without tuning.
4. Should we use a different optimizer for BPTT? (Adam vs SGD, learning rate scheduling)
5. Is the rank-1 approximation in RFLO the main bottleneck, or is it the random feedback?
6. What exactly happens mathematically when you try to derive an RFLO-like rule for GRU?
7. For multi-timescale traces: how many timescales are needed? (K=2? K=5?)

## Suggested Next Step

**Direction A2: Hyperparameter tuning and longer training.** Run BPTT with higher alpha (0.5, 0.8), lower learning rate or learning rate decay, and 10000-20000 iterations to establish a near-perfect baseline. Then tune RFLO similarly. We need to know the true ceiling of each method before concluding anything about their relative capabilities.
