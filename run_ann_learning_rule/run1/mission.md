# Mission: Develop a New Biologically Plausible Learning Algorithm for Recurrent Neural Networks

## Goal

**Develop a novel learning algorithm** for recurrent neural networks that is both biologically plausible and capable of solving machine learning tasks. This is an algorithm design project — the primary deliverable is a new method, not a survey or analysis of existing ones.

## Research Question

What new learning rule can we design for recurrent systems that avoids classical criticisms of BPTT (weight transport, non-locality, non-causal credit assignment) while achieving competitive performance on ML tasks — and what biological principles should it be grounded in?

## Motivation

BPTT is the dominant algorithm for training RNNs but suffers from several biological implausibilities:
- **Weight transport problem**: requires symmetric forward/backward weights
- **Non-locality**: weight updates depend on information not locally available at the synapse
- **Non-causality**: future information is needed to compute present gradients
- **Global error signals**: all neurons must have access to a global loss

These issues make BPTT unlikely as a model of biological learning. We seek learning rules that address these criticisms while retaining the ability to solve temporal credit assignment in practice.

## Approach

1. **Start with vanilla RNN** as the base architecture, then extend to gated architectures (GRU, LSTM)
2. **Balance biological plausibility and computational efficiency** — the algorithm should have a clear metaphor/connection to existing biological theories (e.g., eligibility traces, neuromodulation, synaptic tagging) while being able to solve standard ML benchmarks
3. **Begin with short temporal tasks** to reduce computational burden during development, then scale to longer sequences
4. **Start with supervised learning**, then extend to reinforcement learning and unsupervised learning paradigms

## Key Literature

1. **Liu et al. (2022)** — "Biologically-plausible backpropagation through arbitrary timespans via local neuromodulators" (NeurIPS 2022). Proposes ModProp: local neuromodulatory signals propagate credit through time via synapse-specific filters interacting with eligibility traces.

2. **Murray (2019)** — "Local online learning in recurrent networks with random feedback" (eLife). Introduces RFLO: enforces locality and causality constraints, uses random feedback weights instead of symmetric weight transport for RNN learning.

3. **Lillicrap et al. (2016)** — "Random synaptic feedback weights support error backpropagation for deep learning" (Nature Communications). Demonstrates feedback alignment: random fixed backward weights can transmit teaching signals effectively, resolving the weight transport problem.

## Biological Grounding

The learning rule should connect to one or more of:
- **Eligibility traces**: synaptic tags that mark recently active synapses for later reinforcement
- **Neuromodulation**: diffuse signals (dopamine, acetylcholine, etc.) that gate plasticity
- **Hebbian/anti-Hebbian learning**: local correlation-based rules
- **Dendritic computation**: compartmentalized processing enabling local error computation
- **Synaptic tagging and capture**: multi-timescale consolidation mechanisms

## Success Criteria

- The learning rule avoids weight transport (no symmetric backward weights)
- Weight updates use only locally available information (pre/post-synaptic activity, local modulatory signals)
- The rule is online or near-online (does not require storing full activity history)
- Performance on short-sequence supervised tasks is within reasonable range of BPTT
- Clear biological narrative explaining what each component corresponds to in neural circuits

## Constraints

- Start from vanilla RNN before extending to gated architectures
- Begin with short temporal tasks (sequence length ~10-50) before scaling
- Supervised learning first, then RL and unsupervised
- Prioritize clarity of biological metaphor alongside performance
