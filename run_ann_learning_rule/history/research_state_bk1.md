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
- **Non-locality**: weight updates require information not locally available at the synapse
- **Non-causality**: future states needed to compute present gradients (BPTT unrolling)
- **Eligibility trace**: local synaptic variable tracking recent co-activity, decays over time
- **Neuromodulation**: diffuse chemical signal (e.g., dopamine) that gates plasticity
- **Feedback alignment**: fixed random backward weights replacing transposed forward weights
- **Online learning rule**: updates depend only on past and present, not future

## Related Work

1. **Lillicrap et al. (2016)** — "Random synaptic feedback weights support error backpropagation for deep learning" (Nature Communications). Shows fixed random backward weights can replace symmetric transport in feedforward nets. Foundational.

2. **Murray (2019)** — "Local online learning in recurrent networks with random feedback" (eLife). RFLO: online, local, causal learning rule for vanilla RNNs using random feedback. Key prior art. Limitation: limited extension to gated architectures.

3. **Liu et al. (2022)** — "Biologically-plausible backpropagation through arbitrary timespans via local neuromodulators" (NeurIPS 2022). ModProp: neuromodulatory signals + synapse-specific filters for temporal credit. More flexible than RFLO but more complex.

*We haven't yet done a comprehensive literature survey. There may be other relevant recent work (e.g., e-prop, OSTL, KeRNL) worth examining.*

## Hypotheses

**H1 (confidence: 55%):** Eligibility traces + neuromodulatory gating + random feedback can train vanilla RNNs on short-sequence tasks within reasonable range of BPTT. Rationale: RFLO already shows this partially works; adding neuromodulatory gating might improve temporal credit assignment. Uncertainty: unclear how much the neuromodulatory component adds over plain RFLO for short sequences.

**H2 (confidence: 40%):** Gate activations in GRU/LSTM can serve as natural local modulatory signals for the learning rule — the gating mechanism itself is the biological metaphor for neuromodulation. Rationale: gates already control information flow, which is conceptually similar to neuromodulatory gating of plasticity. Uncertainty: this is speculative; gating in ML and neuromodulation in biology may be too different mechanistically.

**H3 (confidence: 65%):** Performance gap vs. BPTT grows with sequence length but remains bounded for short-range dependencies. Rationale: all local/online rules lose information about long-range dependencies; this is a known tradeoff.

**H4 (confidence: 50%):** Random feedback alignment + eligibility traces work better together than either alone. Rationale: traces provide temporal smoothing that might stabilize noisy random feedback signals. Uncertainty: could also interact badly (smoothing noise = still noise).

## Experimental Designs

*These are candidate directions, not a locked-in plan. The next step should refine one of these into a concrete experiment.*

**Direction A: Reproduce and understand RFLO baseline.** Implement RFLO on a simple task (e.g., copy task, seq len 10-20) to establish a working baseline and build intuition for how local online rules behave in practice.

**Direction B: Design candidate rule.** Starting from RFLO, explore adding neuromodulatory gating or modified eligibility traces. Compare variants on the same simple task.

**Direction C: Gated architecture extension.** Once a vanilla RNN rule works, attempt to adapt it for GRU. This is where H2 gets tested.

**Direction D: RL and unsupervised (future).** Much later — only after supervised case is solid.

## Results Summary

*No experiments conducted yet.*

## Open Questions & Confusions

1. Is RFLO already "good enough" for short sequences, making neuromodulation unnecessary at this scale? We might need to find tasks where RFLO clearly fails before the added complexity is justified.
2. What exactly breaks when applying RFLO to gated architectures? Is it a fundamental issue or an implementation detail?
3. ModProp's synapse-specific filters seem powerful but complex — is there a simpler middle ground between RFLO and ModProp?
4. Should we care about exact gradient approximation quality, or is task performance the only metric that matters?
5. Are there other recent methods (e-prop, KeRNL, OSTL) that we should review before committing to a design direction?
6. For biological plausibility: what level of abstraction is appropriate? Single-neuron biophysics? Circuit-level? Algorithmic-level?
7. How do we define "within reasonable range of BPTT"? 90% of BPTT performance? 80%?

## Suggested Next Step

Start with a literature deep-dive (especially e-prop, KeRNL, OSTL, and any 2023-2024 work) to understand the current landscape before designing a new rule. Alternatively, jump straight to implementing RFLO as a baseline to build hands-on intuition. Either is a reasonable first move.
