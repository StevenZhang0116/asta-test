# Mission: Develop a New Biologically Plausible Learning Algorithm for Recurrent Neural Networks

## Goal

**Develop a novel learning algorithm** for recurrent neural networks that is biologically plausible and capable of solving machine learning tasks. This is an algorithm design project — the primary deliverable is a new method itself.

## Research Question

What new learning rule can we design for recurrent systems that makes meaningful progress on the biological implausibilities of BPTT while achieving competitive performance on ML tasks, and what biological principles should it be grounded in?

The agent must identify, during cold drafting, **which** biological implausibilities of BPTT it considers most important to address, **why**, and **how** the proposed rule addresses them. The mission deliberately does not pre-select these — that selection is part of the design.

## Motivation

BPTT is the dominant algorithm for training RNNs but is widely regarded as biologically implausible for several reasons. The agent should derive its own characterization of those reasons during cold drafting, then design a rule that improves on whichever ones it judges most tractable and most important. Both the diagnosis (which problems matter most) and the prescription (how to address them) are part of the contribution.

The mission does not enumerate the implausibilities, name candidate biological mechanisms, or prescribe a solution family. Doing so would seed the design.

## Approach

1. **Architecture flexibility**: Begin with a recurrent architecture that enables rapid iteration. Treat the learning rule as the primary contribution and the architecture choice as secondary; switch architectures if early experiments reveal limitations of the initial choice.

2. **Biological grounding is part of the design, not given**: The rule should connect to *some* biological theory or mechanism, but the choice of which theory is part of cold drafting. Do not select a mechanism family before the hypothesis exists; do not assume a particular solution shape (feedback pathway, trace-based, energy-based, predictive, contrastive, target-propagation, three-factor, or otherwise).

3. **Begin with short temporal tasks** to reduce computational burden during development; scale to longer sequences once the rule is working. The specific tasks are chosen as part of `experiment_design`, not the mission.

4. **Learning paradigm**: Pick the paradigm (supervised, reinforcement, unsupervised, self-supervised, continual learning ...) that best matches the biological mechanism the hypothesis proposes. Do not default to a paradigm before the mechanism is chosen.

5. **Calibrate training budget to BPTT**: For each task in the experimental design, first run BPTT under matched hyperparameters (architecture, sequence length, batch size, optimizer where applicable) and record `T_bptt` — the iteration count at which BPTT first reaches the task's success threshold (or plateaus, if the threshold cannot be reached in a reasonable budget). Train the proposed rule on the same task for `k × T_bptt` iterations, with `k` a small fixed multiplier (default `k=2`; use up to `k=3` only if early experiments demonstrate materially slower per-iteration convergence of the proposed rule). This anchors training duration to a known reference: under-training gives the rule no fair chance, while over-training wastes compute and can mask whether the rule has truly converged versus simply running out of headroom. Record `T_bptt`, `k`, and the resulting iteration budget in `experiment_design.variables.controls`; report final-iteration metrics (and best-iteration if learning is non-monotonic) in `analysis.output.reasoning`.

## Subfield Disambiguation (intentionally absent)

This mission **does not** name prior work, anchor papers, or related-work clusters. Naming specific papers — even as "pointers to a neighborhood" — leaks the solution shape and defeats the cold-draft requirement. The agent constructs the relevant subfield characterization itself during the post-hoc novelty audit, after the hypothesis is closed.

If you (the human) need to disambiguate the subfield to a downstream reader, do it outside `mission.md` (e.g., in a separate `subfield_notes.md` that the agent is instructed not to read during cold drafting).

## Success Criteria

### Design constraints (by-construction; checked at hypothesis time)

The rule must satisfy these properties **by its own definition** — verifiable from the update equations alone, without running an experiment. They are stated as *what the rule must achieve*, not *how* it achieves them; the mechanism is the agent's choice.

- The rule must avoid weight transport (i.e., must not require that weight matrices used during forward computation be reused symmetrically during credit assignment). The mechanism is unspecified.
- The rule must not rely on information that is unavailable, in principle, at the synapse where the update is applied. "Locally available" is the agent's to operationalize, with an explicit definition in `hypothesis.output.rationale`.
- The rule must offer a clear biological narrative: each component of the update has a stated correspondence to a process plausibly realized in neural circuits. The narrative may be speculative but must be explicit.

A hypothesis whose proposed rule fails any of these by construction should not advance to `experiment_design`; revise the rule first.

Notes on what the mission deliberately does **not** require:
- It does not require an asymmetric *feedback pathway* (that prescribes one solution family). Replace with: the rule must address weight transport in some way, mechanism-agnostic.
- It does not require online operation or bounded-memory traces. If a hypothesis is online and trace-based, fine; if it is offline, energy-based, contrastive, or batch-equilibrium, also fine — provided weight transport and locality are addressed and the biological narrative is coherent.
- It does not name biological mechanisms (eligibility traces, neuromodulation, synaptic tagging, predictive coding, equilibrium states, ...). The agent picks the mechanism family.

### Empirical success criteria (verdicted by `analysis`)

These are outcomes that depend on running experiments. They feed `analysis.output.verdict` per the Evidence Standard.

- **Competitive with BPTT**: on every task in the experimental design, the proposed rule's final-iteration metric (and best-iteration if learning is non-monotonic) is within a stated tolerance of BPTT's at the matched `k × T_bptt` budget. Default tolerance: **within 10% (relative) of BPTT's metric**, or within 2 percentage points absolute for accuracy-style metrics — whichever is looser. The first `experiment_design` may refine this tolerance and must record the chosen value in `variables.controls`; subsequent designs inherit it unless replanned.
- A claim of "competitive performance" is `verdict=supported` only when the tolerance holds on every qualitatively distinct task; per-task disagreement → `verdict=inconclusive` per the Evidence Standard.

## Evidence Standard

> **Every numerical claim recorded in an `analysis` task's `metadata.research_step.output` (or surfaced from there into `summary.md`) MUST be supported by at least two *qualitatively distinct* tasks.** Treat a single-task result as preliminary; record it inside the analysis `reasoning`, but set `verdict=inconclusive` until a second qualitatively distinct task reproduces it.

- **Definition.** "Different tasks" means qualitatively distinct problems. The mission deliberately does not name candidate tasks; the `experiment_design` step justifies each chosen task against the properties the rule is supposed to demonstrate. Treat multiple seeds, hyperparameters, or sequence lengths of one task as *within*-task robustness, and treat a second qualitatively distinct task as the cross-task evidence required here.
- **Encode in `experiment_design.output`.** The `procedure` must enumerate the two (or more) distinct tasks as separate ordered steps; `variables.controls` should call out matched conditions across them; `artifacts_expected` must list a path per task. The `procedure` (or `method`) must also state *why* each task was chosen — what property of the rule it stresses.
- **Encode in `evidence_gathering.output`.** The `artifacts[]` array must contain at least one entry per task, with the task name in the `description` field; `log_path` should record runs for both.
- **Encode in `analysis.output`.** Report per-task numbers explicitly in `reasoning`, list any per-task asymmetries in `caveats`, and set `verdict=supported` only when the claim holds on **every** task included in the design. Disagreement between tasks → `verdict=inconclusive` with both numbers stated in `reasoning`.
- **BPTT-anchored training budget.** Per Approach item 5, every task must include a BPTT calibration run that determines `T_bptt`, and the proposed rule must be trained for `k × T_bptt` iterations on the same task under matched hyperparameters. `experiment_design.variables.controls` must include the matched-hyperparameter list and the chosen `k`; `evidence_gathering.artifacts[]` must include both the BPTT run and the proposed-rule run for each task (clearly labeled in `description`); `analysis.output.reasoning` must report final-iteration metrics for both, and best-iteration metrics when learning is non-monotonic. A claim of competitive performance is only `verdict=supported` when the proposed rule is within the stated tolerance of BPTT at the matched iteration budget on every task; ambiguous outcomes → `verdict=inconclusive`.
- **Single-task results.** Acceptable as an interim checkpoint inside `evidence_gathering`, but the corresponding `analysis` must remain `inconclusive` until a `replan` adds a second distinct-task evidence_gathering and re-runs analysis.

## Environment

- Run all Python code under the **conda environment `panda`** (activate with `conda activate panda`)
- For any required Python package that is missing, install it before running (e.g., `pip install <package>` within the `panda` environment)
- **Use GPU when available**: detect at runtime (e.g., `torch.cuda.is_available()`) and move the model, parameters, and batches to the GPU device; fall back to CPU automatically when a GPU is absent. Code should run correctly in both cases with the same source. Log which device is being used at the start of each run.
- **Independent run**: This run (`run1_cold/`) is an independent replicate. Restrict all reads and references to files **inside this directory** — `mission.md`, `summary.md`, `background_knowledge.txt`, `references.bib`, `.beads/`, and every other artifact must come from `run1_cold/`, with sibling `runN/` directories (including `../run1/`) treated as out of scope. Derive all hypotheses, literature search, novelty audits, and design decisions from within this directory only. The goal is to produce an independent trajectory for comparison across runs.

## Constraints

- **Architecture and paradigm choice are part of the hypothesis, not the mission.** The mission does not order the agent to start with vanilla RNNs, gated RNNs, supervised learning, or any other specific shape. Pick the architecture and paradigm that best fit the biological mechanism the hypothesis proposes; justify the choice in `hypothesis.output.rationale`.
- **Prioritize clarity of biological metaphor alongside performance.**
- **Hypothesis-first, post-hoc novelty audit**: To reduce literature bias on the design of new learning rules, this run defers literature exposure. The bootstrap `literature_review` task is closed with a deferred-audit placeholder (no real literature work), and each hypothesis is drafted **cold** — from first principles, without consulting external literature, semantic-scholar, web search, or prior-art comparisons. The cold-drafting agent must also not consult `mission.md` for solution-shape hints — this mission file is intentionally written to avoid leaking such hints, and the agent should treat any solution vocabulary it recognizes as coming from its own training rather than from the mission. The hypothesis is then exercised through the standard `experiment_design → evidence_gathering → analysis` chain, with the `analysis.verdict` based on experimental evidence alone. **Only after the `analysis` task closes** is a `literature_review` novelty-audit task created, with `inputs[]` listing both the `hypothesis` and the closed `analysis` (and any revised `experiment_design`). This audit records: the queries used in the long-form `summary_path` (`background_knowledge.txt`); the top hits in `citations[]` (each with a `relevance` string stating novel / near-match / overlap); the explicit verdict in `key_findings` (lead bullet: "Novelty audit verdict: novel|near-match|overlap"); and any unresolved residual concerns in `gaps[]`. A `near-match` or `overlap` verdict is **not** grounds for retroactively rewriting the closed experiments — record it for the eventual `synthesis.open_questions` and let it shape the *next* hypothesis via fresh `replan`. The audit must still be re-run whenever the design changes materially after this point (new mechanism added, component replaced or reformulated, biological metaphor shifts, update equations modified).
- **Surprise requirement (enforced via the post-hoc novelty audit)**: The algorithm should reach beyond a trivial or obvious extension of existing methods (i.e., something deeper than simply combining two known techniques, adding a decay term to an existing rule, or swapping one component for another). The design should involve a conceptual insight or unexpected connection that an expert would reach only after several iterations.

  This requirement is operationalized through the post-hoc novelty audit `literature_review` described above; it does not introduce a separate check. Mapping:
    - Audit verdict `novel` → surprise requirement **passes**.
    - Audit verdict `near-match` or `overlap` → surprise requirement **fails**, even if the empirical Success Criteria are met. The closed experiments are not retroactively rewritten; instead, the audit's `key_findings` must call out the failure (e.g., "Surprise requirement: failed (near-match to [paper])"), `gaps[]` must record what would have to change to pass, and the next `replan`-spawned hypothesis must address the overlap rather than incrementally extend the cold-drafted rule.
    - A `synthesis` step may only mark the mission's overall research question as answered if at least one supporting hypothesis has both `analysis.verdict=supported` *and* a `literature_review` audit with verdict `novel`.

## What this mission deliberately omits (and why)

This mission is paired with a more detailed sibling, `mission.md`, that specifies a particular subfield, candidate mechanisms, and anchor references. `mission_cold.md` exists to test whether a less-seeded mission produces meaningfully different cold-drafted hypotheses. Specifically, this version omits:

- A list of BPTT's biological implausibilities (the agent must derive its own).
- Named anchor papers (RFLO, ModProp, etc.) — naming any paper, even as a "neighborhood pointer," seeds the agent's solution.
- Candidate biological mechanisms (eligibility traces, neuromodulation, synaptic tagging, ...) — these are the agent's to choose.
- Solution-shape constraints framed as mechanisms ("asymmetric feedback pathway," "online bounded-memory traces") — replaced with property-level constraints (no weight transport, locality) that the agent satisfies by mechanism of its own choosing.
- Pre-named benchmarks (copy-task, sequential-MNIST, adding-task) — `experiment_design` justifies its own task choices.
- Architecture and paradigm ordering (vanilla → gated; supervised → RL → unsupervised) — the agent picks based on the hypothesis.

The trade-off: this version is harder to compare across runs (since runs may diverge into different subfields), and harder for the agent to make rapid progress (since it must do more upstream framing work itself). Use this mission when the goal is to test what an agent invents under minimal seeding; use `mission.md` when the goal is to converge on results within a chosen subfield.
