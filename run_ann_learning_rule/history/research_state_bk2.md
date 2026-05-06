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
1. **Feedback Alignment** (Lillicrap 2016): random backward weights solve weight transport in feedforward nets. Foundational but feedforward-only.
2. **RFLO** (Murray 2019): online, local, causal rule for vanilla RNNs. Simple (rank-1 approx to influence matrix), but limited temporal credit and no gated architecture support.
3. **e-prop** (Bellec 2020): eligibility propagation for spiking RNNs. Strong bio metaphor (three-factor rule), but targets spiking networks, not standard RNNs. No gated arch support.
4. **KeRNL** (Roth 2019): learned temporal kernel approximates influence of past on present. Moderate performance, weaker bio metaphor, partial locality.
5. **OSTL** (Bohnstingl 2022): forward-mode differentiation + random feedback. Online and causal but O(n²) memory cost.
6. **ModProp** (Liu 2022): per-synapse learned temporal filters + neuromodulatory gating. Most expressive but most complex. No gated arch.

**Critical finding: NO existing method has been convincingly extended to gated architectures (GRU/LSTM).** This is the clearest gap for a novel contribution.

## Hypotheses

**H1 (confidence: 60%, up from 55%):** A learning rule combining eligibility traces + neuromodulatory gating + random feedback can train vanilla RNNs on short-sequence tasks within reasonable range of BPTT. *Strengthened by literature review: RFLO already achieves this partially, and the framework is well-established.*

**H2 (confidence: 45%, up from 40%):** Gate activations in GRU/LSTM can serve as local modulatory signals — specifically, the GRU update gate maps to "how much to consolidate" (plasticity gating) and the reset gate maps to "what eligibility to maintain" (trace decay). *This is the novel contribution opportunity. No prior work has attempted this mapping.*

**H3 (confidence: 65%):** Performance gap vs. BPTT grows with sequence length but remains bounded for short-range dependencies. *Consistent with all prior work showing this tradeoff.*

**H4 (confidence: 55%, up from 50%):** Random feedback alignment + eligibility traces work better together than either alone. *Supported by e-prop results showing that the combination enables temporal credit assignment that neither component achieves alone.*

**H5 (new, confidence: 50%):** A method simpler than ModProp but more expressive than RFLO can be achieved by using a small set of fixed (not learned) temporal filter timescales rather than per-synapse learned filters. *This would sit in the gap between RFLO's single timescale and ModProp's full per-synapse learning.*

## Experimental Designs

*Candidate directions refined by literature review:*

**Direction A (NEXT): Implement RFLO baseline on copy task and adding problem.** This establishes a working local/online baseline and builds hands-on intuition. Short sequences (10-20 for copy, 20-50 for adding). Compare against BPTT. Quantify the performance gap.

**Direction B: Design "Gated-RFLO" — extend RFLO to GRU.** The key novelty. Map GRU gates to biological roles: update gate → plasticity modulation, reset gate → eligibility trace reset. Derive the local update rule. Test on same tasks as Direction A.

**Direction C: Multi-timescale eligibility traces.** Instead of RFLO's single decay rate or ModProp's learned filters, use K fixed timescales (e.g., K=3: fast, medium, slow). The modulatory signal selects which timescale is relevant. Simpler than ModProp, more expressive than RFLO.

**Direction D: Ablation and scaling.** Once a method works, ablate components and scale to longer sequences.

## Results Summary

*No experiments conducted yet. Literature review completed (Step 1).*

## Open Questions & Confusions

1. **[Priority]** What exactly happens mathematically when you try to derive an RFLO-like rule for GRU? Where does it break down? (Need to work through the math.)
2. Is the GRU reset gate really analogous to eligibility trace decay, or is this a forced metaphor?
3. For Direction C (multi-timescale): should the timescales be fixed hyperparameters or slowly learned? Fixed = simpler, learned = more flexible.
4. How do we fairly compare against BPTT? Same number of parameters? Same training time? Same total compute?
5. Should we target rate-based networks (standard ML) or spiking networks (more bio-plausible but less ML-practical)? Current decision: rate-based first.
6. What's the minimal task complexity that discriminates between RFLO and a better method? Copy task might be too easy.
7. Can the method naturally accommodate reward signals (for RL) in place of supervised error? If the modulatory signal = reward prediction error, this might work for free.

## Suggested Next Step

**Direction A: Implement RFLO on copy task (seq len 10-20) and adding problem (seq len 20-50) as baselines, compared against BPTT.** This gives us concrete numbers and working code to extend in Direction B.
