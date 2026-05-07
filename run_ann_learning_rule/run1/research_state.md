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

## Hypotheses — Updated After Exp07 (partial)

**H7 (CONFIRMED, 95%):** Trace quality is THE bottleneck.

**H13 (CONFIRMED):** Temporal decay on full Jacobian works excellently. Decay=0.8→95%, 0.9→99.8%.

**H15 (RESOLVED — FALSE):** Combined spatial-temporal approximation does NOT synergize. *Fully confirmed by Exp07: decay hurts block-diagonal at ALL block sizes. K=8: -1.7 to -8%, K=16: -4.7 to -11%, K=32: -7.9 to -35%. Only K=16+decay=0.95 shows marginal +6.6%. The two approximations compound errors.*

**H18 (RESOLVED — FALSE):** Block-diagonal + decay synergy does not exist. *Even K=32 (half network) + decay=0.95 only reaches 70.9% vs 78.8% without decay. Decay attenuates the already-incomplete block-diagonal signal.*

**H19 (new, confidence: 65%):** The core insight emerging: **full spatial resolution is essential, temporal decay is the ONLY safe approximation.** The algorithm must propagate influence through ALL neurons (full n×n Jacobian) but can tolerate temporal decay. This means our novel algorithm is essentially "Decayed RTRL + Random Feedback" — which is O(n³) per step. To make it practical, we need:
- Efficient GPU implementation (the Jacobian propagation is a batched matrix-matrix multiply)
- Accept O(n³) for moderate n (n=64-256) where it's still feasible
- Or find a DIFFERENT spatial compression (not block-diagonal, not random projection) that works

**H20 (new, confidence: 50%):** For practical networks (n=128-512), the decayed full RTRL might actually be feasible with modern GPU hardware if implemented as efficient tensor operations. The key operation (W_rec @ Jacobian) is just a batch matmul. For n=256, the Jacobian is 256³ = 16M entries per sample — challenging but potentially feasible with batching tricks.

## Experimental Designs

**Direction A (DONE):** Baseline experiments (Exp01-04). ✓
**Direction B1 (DONE):** Block-diagonal and random projection RTRL. ✓ Both fail at small sizes.
**Direction B2 (DONE):** Truncated forward-RTRL. ✓ Decay works; hard reset fails.
**Direction B4 (IN PROGRESS): Block-diagonal + decay.** K=8 results show NO synergy (decay hurts). K=16 and K=32 results pending.

**Direction B5: Network size scaling.** Test decayed full RTRL at n={32, 64, 128, 256}. Quantify the wall-clock time and memory. Determine at what n it becomes infeasible.

**Direction E (NEXT after B4 completes): Formalize the algorithm.** Based on all findings, formalize "Decayed RTRL with Random Feedback" as a clean algorithm:
- Mathematical formulation
- Complexity analysis (identify which operations dominate)
- Biological narrative: decaying eligibility traces + random feedback pathways
- Properties: online, causal, no weight transport, approximate (controlled by decay)
- Extension path to GRU/LSTM

**Direction C: Gated architecture extension.** After formalization.

## Results Summary

### Established Principles:
1. **Trace quality is everything** (Exp04).
2. **Exponential decay on full Jacobian works** (Exp06): 0.8→95%, 0.9→99.8%.
3. **Hard reset fails** (Exp06): destroys information catastrophically.
4. **Block-diagonal fails at small K** (Exp05): need K≥32 for meaningful improvement.
5. **Random projection fails completely** (Exp05).
6. **Block-diagonal + decay does NOT synergize for small K** (Exp07 partial): decay HURTS K=8.

### Exp07: Block-Diagonal + Decay (COMPLETE)

| Config | Accuracy | vs no-decay (from Exp05) |
|--------|----------|--------------------------|
| K=8, decay=0.8 | 22.5% | -8.0% (worse) |
| K=8, decay=0.9 | 25.9% | -4.6% (worse) |
| K=8, decay=0.95 | 28.8% | -1.7% (worse) |
| K=16, decay=0.8 | 30.1% | -11.0% (worse) |
| K=16, decay=0.9 | 36.4% | -4.7% (worse) |
| K=16, decay=0.95 | **47.7%** | **+6.6% (slight improvement)** |
| K=32, decay=0.8 | 43.6% | -35.2% (much worse) |
| K=32, decay=0.9 | 57.9% | -20.9% (worse) |
| K=32, decay=0.95 | 70.9% | -7.9% (worse) |

**Conclusion: Block-diagonal + decay does NOT synergize.** Decay almost universally HURTS block-diagonal performance. Only K=16 with decay=0.95 (minimal decay) shows marginal improvement (+6.6%). For K=32, even the best decay (0.95) reduces accuracy from 78.8% to 70.9%. The two approximations compound errors rather than compensating.

**This definitively establishes: full spatial resolution is non-negotiable. The ONLY viable approximation to RTRL for this task is temporal decay on the FULL Jacobian.**

**Code:** `experiments/exp07_blockdiag_decay.py`
**Plots:** `results/exp07/blockdiag_decay.png`
**Plots:** `results/exp07/` (will be generated on completion)

## Open Questions & Confusions

1. Will K=32 + decay show synergy? K=32 alone gives 78.8%; with decay=0.9 it might approach the full RTRL decay result (99.8%) since K=32 is already half the network.
2. **The big question now:** Is full decayed RTRL (O(n³)) actually our novel algorithm? It's online, causal, uses random feedback (no weight transport), and performs excellently. The "novelty" would be: (a) showing that decay makes RTRL practical by reducing the effective temporal window, (b) showing random feedback works perfectly with RTRL, (c) biological interpretation as decaying eligibility traces.
3. Can we make O(n³) practical with efficient GPU tensor operations? Key op: batched matmul W_rec @ J (shapes n×n and n×n×n → batch matmul of n matrices of size n×n).
4. What's the biological interpretation? "Each synapse maintains a local eligibility trace that decays exponentially and propagates through the recurrent network." This is a multi-dimensional eligibility trace — more complex than standard 1D traces but potentially implementable via dendritic computation.
5. Should we move to formalizing the algorithm and testing on more tasks before further approximation attempts?

## Suggested Next Step

Wait for Exp07 to complete (K=16, K=32 results), then either:
- If K=32+decay matches full RTRL: the combined approach works at O(n × K²) = O(n × n²/4) = O(n³/4) — modest savings.
- If not: accept full decayed RTRL as the algorithm and move to Direction E (formalization) + B5 (scaling).
