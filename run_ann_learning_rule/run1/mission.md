# Mission: Develop a New Biologically Plausible Learning Algorithm for Recurrent Neural Networks

## Goal

**Develop a novel learning algorithm** for recurrent neural networks that is both biologically plausible and capable of solving machine learning tasks. This is an algorithm design project — the primary deliverable is a new method itself.

## Research Question

What new learning rule can we design for recurrent systems that addresses known biological implausibilities of BPTT — such as weight transport, non-locality, non-causal credit assignment, global error signals, or other criticisms (e.g., memory requirements, lack of online operation, separate forward/backward phases, continuous-time incompatibility, or any additional concerns identified during the research) — while achieving competitive performance on ML tasks, and what biological principles should it be grounded in?

## Motivation

BPTT is the dominant algorithm for training RNNs but suffers from biological implausibilities. Common criticisms include:
- **Weight transport problem**: requires symmetric forward/backward weights
- **Non-locality**: weight updates depend on information not locally available at the synapse
- **Non-causality**: future information is needed to compute present gradients
- **Global error signals**: all neurons must have access to a global loss
- **Other concerns**: memory requirements, lack of online operation, separate forward/backward phases, continuous-time incompatibility

Treat this list as illustrative. The research should identify which implausibilities are most tractable to address and most important for biological realism, and focus on those. We seek learning rules that make meaningful progress on biological plausibility while retaining the ability to solve temporal credit assignment in practice.

## Approach

1. **Architecture flexibility**: Begin with a recurrent architecture that enables rapid iteration. Vanilla RNNs offer simplicity for initial development; if early experiments reveal fundamental limitations, pivot promptly to gated architectures (GRU, LSTM, etc.). Treat the learning rule as the primary contribution and the architecture choice as secondary.

2. **Balance biological plausibility and computational efficiency** — the algorithm should have a clear metaphor/connection to existing biological theories (e.g., eligibility traces, neuromodulation, synaptic tagging) while being able to solve standard ML benchmarks

3. **Begin with short temporal tasks** (sequence length ~100–200 timesteps) to reduce computational burden during development, then scale to longer sequences once the rule is working

4. **Learning paradigm**: Start with the paradigm that best matches the biological mechanism being explored. Supervised learning is a natural starting point for many credit assignment problems; if the design suggests a reinforcement or unsupervised formulation would be more natural or biologically grounded, explore that direction early.

5. **Calibrate training budget to BPTT**: For each task in the experimental design, first run BPTT under matched hyperparameters (architecture, sequence length, batch size, optimizer where applicable) and record `T_bptt` — the iteration count at which BPTT first reaches the task's success threshold (or plateaus, if the threshold cannot be reached in a reasonable budget). Train the proposed rule on the same task for `k × T_bptt` iterations, with `k` a small fixed multiplier (default `k=2`; use up to `k=3` only if early experiments demonstrate materially slower per-iteration convergence of the proposed rule). This anchors training duration to a known reference: under-training gives the rule no fair chance, while over-training wastes compute and can mask whether the rule has truly converged versus simply running out of headroom. Record `T_bptt`, `k`, and the resulting iteration budget in `experiment_design.variables.controls`; report final-iteration metrics (and best-iteration if learning is non-monotonic) in `analysis.output.reasoning`.

## Scope Anchors (not a reading list)

This mission lives in the research tradition exemplified by work on local online RNN learning with random feedback (Murray 2019, *eLife* — RFLO) and neuromodulatory credit propagation through time (Liu et al. 2022, *NeurIPS* — ModProp). Treat these references **only as a way to disambiguate the subfield** the mission targets. Treat them as pointers to the neighborhood; build the curated bibliography, baseline list, novelty audit, and Related Work section of `research_state.md` independently.

The research loop should build its own Related Work from fresh literature searches, and the novelty audit must evaluate the current design against the full body of relevant work, going well beyond these anchors.

## Success Criteria

### Design constraints (by-construction; checked at hypothesis time)

These are properties the rule must satisfy by its definition — verifiable from the update equations alone, without running an experiment. They belong in `hypothesis.output.statement` / `rationale`, not in any `analysis.verdict`.

- The learning rule uses asymmetric (e.g., random or learned-forward-only) feedback pathways in place of weight transport
- Weight updates rely on locally available information at each synapse
- The rule operates online or near-online, using bounded-memory traces in place of full activity history
- Clear biological narrative explaining what each component corresponds to in neural circuits

A hypothesis whose proposed rule fails any of these by construction should not advance to `experiment_design`; revise the rule first.

### Empirical success criteria (verdicted by `analysis`)

These are outcomes that depend on running experiments. They feed `analysis.output.verdict` per the Evidence Standard.

- **Competitive with BPTT**: on every task in the experimental design, the proposed rule's final-iteration metric (and best-iteration if learning is non-monotonic) is within a stated tolerance of BPTT's at the matched `k × T_bptt` budget. Default tolerance: **within 10% (relative) of BPTT's metric**, or within 2 percentage points absolute for accuracy-style metrics — whichever is looser. The first `experiment_design` may refine this tolerance and must record the chosen value in `variables.controls`; subsequent designs inherit it unless replanned.
- A claim of "competitive performance" is `verdict=supported` only when the tolerance holds on every qualitatively distinct task; per-task disagreement → `verdict=inconclusive` per the Evidence Standard.

## Evidence Standard

> **Every numerical claim recorded in an `analysis` task's `metadata.research_step.output` (or surfaced from there into `summary.md`) MUST be supported by at least two *qualitatively distinct* tasks.** Treat a single-task result as preliminary; record it inside the analysis `reasoning`, but set `verdict=inconclusive` until a second qualitatively distinct task reproduces it.

- **Definition.** "Different tasks" means qualitatively distinct problems (e.g., copy-task + sequential-MNIST + adding-task). Treat multiple seeds, hyperparameters, or sequence lengths of one task as *within*-task robustness, and treat a second qualitatively distinct task as the cross-task evidence required here.
- **Encode in `experiment_design.output`.** The `procedure` must enumerate the two (or more) distinct tasks as separate ordered steps; `variables.controls` should call out matched conditions across them; `artifacts_expected` must list a path per task.
- **Encode in `evidence_gathering.output`.** The `artifacts[]` array must contain at least one entry per task, with the task name in the `description` field; `log_path` should record runs for both.
- **Encode in `analysis.output`.** Report per-task numbers explicitly in `reasoning`, list any per-task asymmetries in `caveats`, and set `verdict=supported` only when the claim holds on **every** task included in the design. Disagreement between tasks → `verdict=inconclusive` with both numbers stated in `reasoning`.
- **BPTT-anchored training budget.** Per Approach item 5, every task must include a BPTT calibration run that determines `T_bptt`, and the proposed rule must be trained for `k × T_bptt` iterations on the same task under matched hyperparameters. `experiment_design.variables.controls` must include the matched-hyperparameter list and the chosen `k`; `evidence_gathering.artifacts[]` must include both the BPTT run and the proposed-rule run for each task (clearly labeled in `description`); `analysis.output.reasoning` must report final-iteration metrics for both, and best-iteration metrics when learning is non-monotonic. A claim of competitive performance is only `verdict=supported` when the proposed rule is within a stated tolerance of BPTT at the matched iteration budget on every task; ambiguous outcomes → `verdict=inconclusive`.
- **Single-task results.** Acceptable as an interim checkpoint inside `evidence_gathering`, but the corresponding `analysis` must remain `inconclusive` until a `replan` adds a second distinct-task evidence_gathering and re-runs analysis.

## Environment

- Run all Python code under the **conda environment `panda`** (activate with `conda activate panda`)
- For any required Python package that is missing, install it before running (e.g., `pip install <package>` within the `panda` environment)
- **Use GPU when available**: detect at runtime (e.g., `torch.cuda.is_available()`) and move the model, parameters, and batches to the GPU device; fall back to CPU automatically when a GPU is absent. Code should run correctly in both cases with the same source. Log which device is being used at the start of each run.
- **Independent run**: This run (`run1/`) is an independent replicate. Restrict all reads and references to files **inside this directory** — `mission.md`, `summary.md`, `background_knowledge.txt`, `references.bib`, `.beads/`, and every other artifact must come from `run1/`, with sibling `runN/` directories treated as out of scope. Derive all hypotheses, literature search, novelty audits, and design decisions from within this directory only. The goal is to produce an independent trajectory for comparison across runs.

## Constraints

- Start from vanilla RNN, then extend to gated architectures
- Tackle supervised learning first, then move on to RL and unsupervised
- Prioritize clarity of biological metaphor alongside performance
- **Hypothesis-first, post-hoc novelty audit**: To reduce literature bias on the design of new learning rules, this run defers literature exposure. The bootstrap `literature_review` task is closed with a deferred-audit placeholder (no real literature work), and each hypothesis is drafted **cold** — from first principles, without consulting external literature, semantic-scholar, web search, or prior-art comparisons. The hypothesis is then exercised through the standard `experiment_design → evidence_gathering → analysis` chain, with the `analysis.verdict` based on experimental evidence alone. **Only after the `analysis` task closes** is a `literature_review` novelty-audit task created, with `inputs[]` listing both the `hypothesis` and the closed `analysis` (and any revised `experiment_design`). This audit records: the queries used in the long-form `summary_path` (`background_knowledge.txt`); the top hits in `citations[]` (each with a `relevance` string stating novel / near-match / overlap); the explicit verdict in `key_findings` (lead bullet: "Novelty audit verdict: novel|near-match|overlap"); and any unresolved residual concerns in `gaps[]`. A `near-match` or `overlap` verdict is **not** grounds for retroactively rewriting the closed experiments — record it for the eventual `synthesis.open_questions` and let it shape the *next* hypothesis via fresh `replan`. The audit must still be re-run whenever the design changes materially after this point (new mechanism added, component replaced or reformulated, biological metaphor shifts, update equations modified).
- **Surprise requirement (enforced via the post-hoc novelty audit)**: The algorithm should reach beyond a trivial or obvious extension of existing methods (i.e., something deeper than simply combining two known techniques, adding a decay term to an existing rule, or swapping one component for another). The design should involve a conceptual insight or unexpected connection that an expert would reach only after several iterations.

  This requirement is operationalized through the post-hoc novelty audit `literature_review` described above; it does not introduce a separate check. Mapping:
    - Audit verdict `novel` → surprise requirement **passes**.
    - Audit verdict `near-match` or `overlap` → surprise requirement **fails**, even if the empirical Success Criteria are met. The closed experiments are not retroactively rewritten; instead, the audit's `key_findings` must call out the failure (e.g., "Surprise requirement: failed (near-match to [paper])"), `gaps[]` must record what would have to change to pass, and the next `replan`-spawned hypothesis must address the overlap rather than incrementally extend the cold-drafted rule.
    - A `synthesis` step may only mark the mission's overall research question as answered if at least one supporting hypothesis has both `analysis.verdict=supported` *and* a `literature_review` audit with verdict `novel`.
