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
- **Block-diagonal RTRL**: only track within-block Jacobians, ignore cross-block influences

## Related Work

See `docs/literature_review.md`.

## Hypotheses — Updated After Exp05

**H7 (CONFIRMED, 95%):** Trace quality is THE bottleneck. Full RTRL + random FB = 100%; rank-1 = 28%.

**H10 (confidence: 50%, down from 70%):** A cheap RTRL approximation can work. *Complicated by Exp05: block-diagonal needs K=32 (half the network) to reach 78.8%, and random projection fails completely. Simple approximations to RTRL lose critical cross-neuron interaction information.*

**H11 (confidence: 30%, down from 60%):** Block-diagonal maps to cortical columns. *Weakened: K=4 and K=8 blocks (realistic column sizes) only achieve 27-30%, worse or equal to simple RFLO. The biological column size (5-20 neurons) is too small to capture the needed Jacobian structure.*

**H12 (new, confidence: 65%):** The key information that RTRL captures and rank-1 misses is the **recurrent propagation of influence** — how weight changes affect hidden states not just at the current step but through the recurrent connections over multiple steps. Block-diagonal fails because most influence paths cross block boundaries. A successful approximation must somehow capture these cross-block paths.

**H13 (new, confidence: 55%):** Instead of approximating RTRL directly, a better approach may be to use **truncated RTRL with a short window** (e.g., propagate the Jacobian for K steps then reset). This is like BPTT with truncation length K but computed forward in time. It's online, causal, and captures K-step influence paths. Bio metaphor: short-term synaptic plasticity mechanisms that integrate over a few hundred ms.

**H14 (new, confidence: 45%):** A **hierarchical** approach could work: fast local traces within groups + slower global trace that communicates between groups. Bio metaphor: local circuits (fast GABA-ergic) + global neuromodulation (slow, diffuse).

**H15 (new, confidence: 60%):** Spatial and temporal approximations need not be applied in isolation — a **combined spatial-temporal approximation** (block-diagonal RTRL within a truncated time window) could achieve good performance at cost O(n × K_spatial² × K_temporal), which is far cheaper than full RTRL (O(n³)) while potentially capturing enough structure from both dimensions. For example, block size K=8 alone gives only 30%, and truncated window alone may need K_temporal≥11, but combining them (K_spatial=16, K_temporal=10) might synergize because partial spatial coverage over a longer window captures influence paths that neither alone does. Bio metaphor: local circuits (minicolumns) that integrate over oscillation-bounded time windows (theta/gamma cycles).

**H16 (new, confidence: 55%):** The relative effectiveness of RTRL approximations may depend on **network size n**. All experiments so far use n=64. As n grows, (a) the ratio of within-block to cross-block connections shrinks for fixed K, potentially making block-diagonal worse; (b) the network becomes more redundant/overparameterized, potentially making approximations MORE effective since the Jacobian becomes lower-rank; (c) the computational gap between full RTRL (O(n³)) and approximations becomes more important, changing the practical tradeoffs. Testing at n={32, 64, 128, 256} will reveal whether our conclusions are specific to n=64 or general.

## Experimental Designs

**Direction A (DONE):** Baseline experiments (Exp01-04). ✓
**Direction B1 (DONE):** Block-diagonal and random projection RTRL. ✓ Both fail at realistic sizes.

**Direction B2 (NEXT): Truncated forward-RTRL.** Propagate the full Jacobian forward but reset every K steps. Test K={3, 5, 7, 10, 15, 20} steps. This keeps the algorithm online and causal with O(K × n²) memory. If K=10 suffices (the delay in copy task is 11 steps), this confirms that local temporal windows are the key.

**Direction B3: Hierarchical trace.** Fast within-group RTRL (small K) + slow between-group eligibility trace with random connectivity. Combines the strengths of block-diagonal (local detail) with global information flow.

**Direction B4: Combined spatial-temporal approximation.** Apply block-diagonal RTRL (K_spatial = {8, 16}) within a truncated time window (K_temporal = {5, 10, 15}). The Jacobian is block-diagonal (only within-group) AND reset every K_temporal steps. Memory: O(n × K_spatial² × K_temporal). Sweep the 2D grid of (K_spatial, K_temporal) to find the Pareto frontier of accuracy vs compute. This tests whether partial information from both dimensions is better than full information from one dimension alone.

**Direction B5: Network size scaling.** Test how approximation methods behave as n varies: n={32, 64, 128, 256}. For each size, compare full RTRL, block-diagonal (K=n/4, n/2), and the best method from B2-B4. Key questions: (a) does the minimum required rank/block size grow with n? (b) does overparameterization at large n make approximations easier? (c) at what n does full RTRL become computationally infeasible and approximations become necessary?

**Direction C: Gated architecture extension.** After vanilla RNN works.

## Results Summary

### Key Principle (from Exp04):
**Trace quality is everything.** Random FB works perfectly. Optimizer is irrelevant.

### Exp05: Approximate RTRL (10000 iters, hidden=64, copy task T=10)

| Method | Accuracy | Memory | Time |
|--------|----------|--------|------|
| Full RTRL | 100% | 262,144 | ~550s |
| BlockDiag K=32 | **78.8%** | 65,536 | 642s |
| BlockDiag K=16 | 41.1% | 16,384 | 763s |
| BlockDiag K=8 | 30.5% | 4,096 | 978s |
| **Rank-1 (RFLO)** | **28.0%** | 4,096 | ~115s |
| BlockDiag K=4 | 26.6% | 1,024 | 1417s |
| RandProj R=2 | 17.3% | 8,192 | 156s |
| RandProj R=4 | 16.2% | 16,384 | 156s |
| RandProj R=8 | 16.2% | 32,768 | 156s |
| RandProj R=16 | 14.9% | 65,536 | 156s |
| RandProj R=32 | 12.1% | 131,072 | 156s |

**Key findings:**
1. **Block-diagonal**: improves monotonically with block size but needs K≈32 (half network) for meaningful improvement (78.8%). Realistic "column" sizes (K=4-8) give no benefit over RFLO.
2. **Random projection**: completely fails (~12-17%, worse than random). The projected trace recursion approximation (using effective transition matrix Q @ diag(phi') @ W_rec @ Q^T) destroys the gradient signal. More projections = worse (noise accumulates).
3. **Cross-neuron influence paths are critical**: block-diagonal fails because influence propagates across block boundaries. Any successful approximation must capture inter-group dynamics.

**Code:** `experiments/exp05_approx_rtrl.py`
**Plots:** `results/exp05/approx_rtrl.png`

## Open Questions & Confusions

1. **[Priority]** Can truncated forward-RTRL (propagate K steps then reset) work? The copy task has delay ~11 steps — if K≥11, it should work. But shorter K might also help by capturing partial influence paths.
2. Why does random projection fail so badly? The effective transition matrix Q@diag(phi')@W_rec@Q^T loses information about the phase of oscillations in the Jacobian. Can this be fixed?
3. Is there a way to maintain CROSS-block information cheaply? E.g., track a low-rank approximation of the between-block Jacobian while keeping full within-block?
4. Hierarchical idea: within-group full RTRL (K=8, captures local dynamics) + between-group random feedback (captures global credit). Would this combine the 30% of K=8 with additional global information?
5. The copy task delay is 11 steps. Does truncated RTRL with window=11 perfectly solve it? If so, is the window length a tunable hyperparameter or a fundamental limitation?
6. Biological plausibility of truncated RTRL: what is the neural mechanism for "resetting" the trace every K steps? Could be tied to oscillatory rhythms (theta cycles ≈ 100-200ms, ~5-10 timesteps).

## Suggested Next Step

**Direction B2: Truncated forward-RTRL.** Propagate the full n×n×n Jacobian forward but reset it to zero every K steps. Test K = {3, 5, 7, 10, 15, 21}. The copy task delay is ~11 steps, so K≥11 should solve it. If K=7-10 already helps significantly, this suggests partial temporal windows are valuable and could form the basis of a biologically plausible rule (tied to neural oscillation cycles).
