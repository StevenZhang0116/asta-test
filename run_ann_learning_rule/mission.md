# Mission: Biologically Plausible Learning Rules for Recurrent Neural Networks

## Objective

Design and evaluate biologically plausible learning rules for recurrent neural networks (RNNs) that address the classical criticisms of Backpropagation Through Time (BPTT), including:

- **Weight transport problem**: BPTT requires symmetric forward and backward weights
- **Non-locality**: weight updates depend on information not locally available at the synapse
- **Non-causality**: BPTT requires future information (backward pass through time)
- **Global error signals**: precise global error must be communicated to all neurons

## Experimental Plan

- **Phase 1**: Supervised learning tasks (sequential MNIST, copy task, pattern generation)
- **Phase 2**: Reinforcement learning tasks (control problems, simple games)
- **Phase 3**: Unsupervised learning (sequence prediction, generative modeling)

## Success Criteria

- The learning rule is demonstrably more biologically plausible than BPTT (satisfies locality, causality, no weight transport)
- Performance on benchmarks is competitive with or within reasonable range of BPTT
- The rule scales from vanilla RNN to gated architectures without ad-hoc modifications
- Clear biological interpretation exists for each component of the algorithm
