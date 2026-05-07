# Research State

## Research Question & Scope

**Goal:** Develop a **new learning algorithm** for recurrent neural networks that is biologically plausible and ML-capable.

**Primary Question:** What new learning rule can we invent for recurrent systems that avoids BPTT's biological implausibilities while achieving competitive task performance?

**Scope:**
- Start with vanilla RNN, then extend to gated architectures (GRU, LSTM)
- Supervised learning first, then RL and unsupervised
- Short sequences initially (10-50 steps)
- Clear biological metaphor required
- Computationally feasible

**Environment:**
- Python: `/home/zihan.zhang/.conda/envs/panda/bin/python`
- Install: `/home/zihan.zhang/.conda/envs/panda/bin/pip install <package>`

## Operational Definitions

- **Weight transport**: backward weights = transposes (implausible)
- **Non-locality**: updates need non-local info
- **RTRL**: Real-Time Recurrent Learning — exact forward-mode gradient, O(n³), online
- **Three-factor rule**: ΔW = f(pre, post, modulator)
- **Decayed RTRL**: full RTRL Jacobian propagation with multiplicative decay factor each step

## Related Work

See `docs/literature_review.md`.

## Hypotheses — Updated After Exp06

**H7 (CONFIRMED, 95%):** Trace quality is THE bottleneck. Full RTRL + random FB = 100%; rank-1 = 28%.

**H13 (PARTIALLY CONFIRMED, nuanced):** Temporal windowing helps — but ONLY with exponential decay, NOT hard reset. Decay=0.8 (eff window ~5 steps) gives 95.1%; decay=0.9 (~10 steps) gives 99.8%. Hard reset at any period fails (23-57% for K=5-15). The key insight: even heavily attenuated long-range info is useful; hard truncation destroys it.

**H15 (confidence: 55%):** Combined spatial-temporal approximation may still work. Now we know the temporal side must use decay (not reset). A combined approach would be: block-diagonal Jacobian with exponential decay. But given that decay=0.8 on FULL Jacobian gives 95%, the question is whether decay can compensate for spatial approximation too.

**H16 (confidence: 55%):** Network size effects still untested.

**H17 (new, confidence: 70%):** The winning formula for our novel algorithm is: **Full-spatial RTRL with exponential decay + random feedback.** This is:
- Online and causal ✓ (forward-mode, no future info needed)
- No weight transport ✓ (random feedback)
- Uses local info ✓ (each entry J[i,j,k] depends on local activities and the recurrent weights — though the full Jacobian itself is non-local in the sense that it's n³ entries)
- Biologically interpretable: the decay maps to natural eligibility trace decay timescales

**THE REMAINING PROBLEM:** Full RTRL is O(n³) memory and O(n⁴) compute per step. This is NOT computationally feasible for large networks. We need to find a way to reduce the spatial complexity while keeping the temporal decay property.

**H18 (new, confidence: 60%):** Since exponential decay already helps enormously (decay=0.8 → 95%), perhaps block-diagonal + decay can synergize. Block-diagonal K=16 alone gives 41%. With decay=0.8-0.9, the decayed long-range info from WITHIN the block might partially compensate for missing cross-block paths. Testing: block-diagonal K=16 with decay=0.9.

## Experimental Designs

**Direction A (DONE):** Baseline experiments (Exp01-04). ✓
**Direction B1 (DONE):** Block-diagonal and random projection RTRL. ✓ Both fail at small sizes.
**Direction B2 (DONE): Truncated forward-RTRL.** ✓ Exponential decay works beautifully; hard reset fails.

**Direction B4 (NEXT): Combined spatial-temporal approximation.** Test block-diagonal RTRL (K_spatial = {8, 16, 32}) WITH exponential decay (trace_decay = {0.8, 0.9, 0.95}). The hypothesis: decay on block-diagonal traces might recover some of what's lost by ignoring cross-block paths. Sweep the 2D grid. Compare against: decay-only (full Jacobian), block-only (no decay), and full RTRL.

**Direction B5: Network size scaling.** Test at n={32, 64, 128, 256}.

**Direction C: Gated architecture extension.** After vanilla RNN method is solid.

**Direction E (new): Formalize the algorithm.** Once we find the right approximation, write it up as a clean algorithm with:
- Mathematical formulation
- Biological narrative (what each component corresponds to)
- Complexity analysis
- Extension to GRU/LSTM

## Results Summary

### Key Principles Established:
1. **Trace quality is everything** (Exp04). Random FB works. Optimizer irrelevant.
2. **Exponential decay on full Jacobian works excellently** (Exp06). Decay=0.8→95%, 0.9→99.8%.
3. **Hard reset fails** (Exp06). Even K=11 (matching task delay) only gets 28%.
4. **Block-diagonal alone fails at small K** (Exp05). K=4→27%, K=8→31%, K=16→41%, K=32→79%.
5. **Random projection completely fails** (Exp05). 12-17% regardless of R.

### Exp06: Truncated Forward-RTRL (10000 iters, hidden=64, copy task T=10)

**Exponential Decay (soft window):**
| Decay | Eff. Window | Accuracy |
|-------|-------------|----------|
| 0.80 | ~5 steps | **95.1%** |
| 0.90 | ~10 steps | **99.8%** |
| 0.95 | ~20 steps | **100%** |
| 0.99 | ~100 steps | **100%** |
| 1.00 | infinite | **100%** |

**Periodic Hard Reset:**
| Reset Period | Accuracy |
|-------------|----------|
| K=5 | 23.0% |
| K=7 | 20.2% |
| K=11 | 28.3% |
| K=15 | 57.1% |
| K=21 (full) | 100% |

**Code:** `experiments/exp06_truncated_rtrl.py`
**Plots:** `results/exp06/truncated_rtrl.png`

## Open Questions & Confusions

1. **[Priority]** Can block-diagonal + decay synergize? If BlockDiag K=16 with decay=0.9 reaches >80%, we have a viable O(n × K²) algorithm.
2. Why does hard reset fail so badly even when K matches the delay? Because the reset alignment is random w.r.t. when relevant info arrives — some output steps have nearly zero Jacobian depth. Exponential decay avoids this by always having SOME info from all past steps.
3. The full-Jacobian-with-decay approach (the current winner) is still O(n³). How to make it practical for n>100?
4. Is the decay rate a critical hyperparameter? For copy task, 0.8-0.9 works. For longer-delay tasks, would we need higher decay? This could limit generality.
5. Biological interpretation of decayed RTRL: each synapse maintains an influence trace that naturally decays — this IS an eligibility trace, just high-dimensional (full Jacobian). Can biology maintain this much information? Likely no — need to find the essential low-dimensional structure.

## Suggested Next Step

**Direction B4: Block-diagonal + exponential decay.** Test BlockDiag K={8, 16, 32} combined with decay={0.8, 0.9, 0.95} on copy task. This is the critical test: if spatial approximation + temporal decay synergize, we have an algorithm with complexity O(n × K² ) that is both biologically motivated (local circuits with decaying eligibility) and potentially performant.
