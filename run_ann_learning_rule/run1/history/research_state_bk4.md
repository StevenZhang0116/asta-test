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

**H1 (confidence: 30%, down from 45%):** A learning rule combining eligibility traces + neuromodulatory gating + random feedback can train vanilla RNNs on short-sequence tasks within reasonable range of BPTT. *Further weakened by Exp02: RFLO saturates at ~34.5% on copy task (seq_len=10) across ALL hyperparameter settings, while BPTT reaches 100%. The rank-1 eligibility trace approximation appears to be fundamentally insufficient for memory tasks. Plain RFLO is not a viable starting point without significant modification.*

**H2 (confidence: 45%):** Gate activations in GRU/LSTM can serve as local modulatory signals. *Unchanged — not yet tested. However, if the basic trace mechanism is this weak, gating alone won't fix it.*

**H3 (confidence: 75%, up from 70%):** Performance gap vs. BPTT grows with sequence length. *Strongly confirmed: even at seq_len=10, the gap is 100% vs 34.5%.*

**H4 (confidence: 55%):** Random feedback alignment + eligibility traces work better together than either alone. *Need ablation: is the bottleneck the random feedback or the trace approximation?*

**H5 (confidence: 60%, up from 50%):** Multi-timescale eligibility traces (or richer trace dynamics) are needed to bridge the gap. *Strengthened: single-timescale traces saturate at ~34% regardless of the timescale (alpha). The rank-1 approximation loses too much temporal structure.*

**H6 (RESOLVED):** BPTT needs tuning — confirmed that with alpha=1.0, lr=1e-3, it reaches 100% on copy task. The architecture is fine; the issue was alpha=0.2 causing information decay.

**H7 (new, confidence: 55%):** The key bottleneck in RFLO is the rank-1 approximation to the influence matrix, not the random feedback. A higher-rank trace (rank-K, K=3-5) could dramatically improve performance while remaining local and online. This is where our novel contribution might lie.

## Experimental Designs

**Direction A (DONE): Implement RFLO baseline.** ✓ 
**Direction A2 (DONE): Hyperparameter sweep.** ✓ Confirmed RFLO ceiling at ~34.5%, BPTT at 100%.

**Direction A3 (NEXT): Ablation — separate the effect of random feedback from trace approximation.** Run RFLO with exact transpose (W_out^T) instead of random B. If performance jumps significantly, the bottleneck is random feedback. If it stays at ~34%, the bottleneck is the trace. This is critical for knowing what to improve.

**Direction B: Design novel rule with richer traces.** Based on ablation results, design a rule with:
- Option 1: Higher-rank eligibility traces (maintain K rank-1 traces with different timescales)
- Option 2: Use recurrent dynamics in the trace itself (second-order traces)
- Option 3: Combine with gated architecture where gates modulate trace dynamics

**Direction C: Gated architecture extension.** After vanilla RNN rule is competitive.

**Direction D: Scaling.** After method works.

## Results Summary

### Experiment 01: RFLO vs BPTT Initial (5000 iters, hidden=128, alpha=0.2)

| Task | Method | Result |
|------|--------|--------|
| Copy (T=10) | BPTT (lr=1e-3) | 73.5% acc |
| Copy (T=10) | RFLO (lr=1e-2) | 29.7% acc |
| Adding (T=30) | BPTT (lr=1e-3) | 0.142 MSE |
| Adding (T=30) | RFLO (lr=1e-2) | 0.174 MSE |

### Experiment 02: Hyperparameter Sweep on Copy Task (10000 iters, hidden=128)

**BPTT Results (best → worst):**
| Config | Final Accuracy |
|--------|---------------|
| alpha=1.0, lr=1e-3 | **1.000** |
| alpha=0.5, lr=3e-3 | **1.000** |
| alpha=0.5, lr=1e-3 | 0.998 |
| alpha=0.2, lr=3e-3 | 0.986 |
| alpha=0.2, lr=1e-3 | 0.927 |
| alpha=1.0, lr=3e-3 | 0.699 |

**RFLO Results (best → worst):**
| Config | Final Accuracy |
|--------|---------------|
| alpha=0.2, lr=0.02 | **0.345** |
| alpha=0.2, lr=0.01 | 0.328 |
| alpha=0.3, lr=0.01 | 0.325 |
| alpha=0.1, lr=0.02 | 0.320 |
| alpha=0.5, lr=0.02 | 0.320 |
| alpha=0.5, lr=0.01 | 0.316 |
| alpha=0.3, lr=0.02 | 0.314 |
| alpha=0.1, lr=0.005 | 0.304 |
| alpha=0.1, lr=0.01 | 0.302 |
| alpha=0.2, lr=0.005 | 0.300 |

**Key findings:**
- BPTT achieves 100% with proper alpha (1.0 or 0.5); the Exp01 issue was alpha=0.2 causing information decay
- RFLO saturates at ~30-35% regardless of alpha and lr — this is a **fundamental limitation of the rule**, not hyperparameters
- The gap between BPTT (100%) and RFLO (34.5%) is enormous and consistent
- RFLO's rank-1 trace approximation cannot capture the temporal structure needed for the copy task

**Code:** `experiments/exp02_hyperparam_sweep.py`
**Plots:** `results/exp02/hyperparam_sweep.png`

## Open Questions & Confusions

1. **[Priority]** Is RFLO's bottleneck the rank-1 trace or the random feedback? Ablation needed.
2. Would a higher-rank trace (K=3-5 independent traces with different decay rates) break through the 34% ceiling while remaining local?
3. Can we formulate a trace that uses recurrent information without violating locality? (e.g., a trace that evolves with its own dynamics)
4. The copy task requires exact memory — is this too hard for approximate methods? Should we also test on "softer" tasks (pattern generation, classification)?
5. Murray (2019) reports RFLO working on various tasks — are our results consistent with the original paper, or is there a bug? (The original paper uses different tasks, mostly continuous outputs, not discrete memory.)
6. What exactly makes the copy task hard for RFLO? Is it that the relevant input happens long before the output, and the rank-1 trace has decayed by then?

## Suggested Next Step

**Direction A3: Ablation experiment.** Replace random feedback B with exact W_out^T in our RFLO implementation. If accuracy jumps significantly (e.g., to 60%+), the random feedback is a major bottleneck and we should focus on better feedback mechanisms. If it stays at ~35%, the trace approximation is the bottleneck and we should focus on richer trace dynamics (Direction B).
