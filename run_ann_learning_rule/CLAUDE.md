# Conventions for `run_ann_learning_rule/`

These rules apply to every `runN/` subdirectory. Each run is an independent replicate of the same mission (see `runN/mission.md`).

## Tool usage

- **Coding/running experiments: use the `run-experiment` (Panda) skill, not ad-hoc Python.**
  Reason: Panda produces standardized experiment artifacts (`experiment.py`, `experiment-trace.txt`, `experiment.html`, `experiment-artifacts.py`) in `.asta/experiment/` that downstream Asta tools consume. Writing the experiment directly skips these artifacts and breaks interoperability even if the numeric results are correct.
  How to apply: any research step whose concrete task is "implement + run a training / evaluation script" goes through `run-experiment`. Only bypass it for trivial utility scripts (e.g. a one-off plot from existing JSON).

- **Literature reports: render with the `preview` skill after writing.**
  Reason: grep-verifying that `[@key]` entries exist in `.bib` confirms the necessary condition for citations but not that they render. Preview catches formatting, encoding, and citation-processor issues before the report is used downstream.
  How to apply: every time `literature_report*.md` is created or materially updated.

- **One-off paper lookups: prefer the `semantic-scholar-lookup` skill over ad-hoc `asta papers get` shell loops.**
  Reason: matches the Asta ecosystem and the skill handles edge cases (missing fields, search ranking) that ad-hoc scripts silently mishandle.
  How to apply: when fetching metadata for ≤ ~10 papers. For bulk cluster searches, `asta literature find` is still correct.

## Project layout

- **`references.bib` lives inside each `runN/`, not in a shared parent.**
  Reason: `mission.md` defines each run as an independent replicate that must not read sibling runs' files. A shared parent `references.bib` would violate that independence.
  How to apply: the literature report's YAML frontmatter should read `bibliography: references.bib` (same dir), not `../references.bib`.

- **Standard per-run artifacts:**
  - `mission.md` — research mission (given, not edited by the research loop).
  - `research_state.md` — living research document.
  - `logbook.md` — one paragraph per step.
  - `background_knowledge.txt` — context for the current step (overwritten each step).
  - `history/research_state_bk<N>.md` — backups made at the end of each step.
  - `references.bib` — BibTeX for all cited work.
  - `literature_report_step<N>.md`, `design_step<N>.md`, etc. — step-specific artifacts.

## Research-loop discipline

- **Follow the research-step skill's prescribed tools.** If a skill says "use X for Y", invoke X. Don't bypass it to save time — the skill exists to produce artifacts other tools consume.

- **If you deviate, say so up-front before acting.** Describe the deviation and the reason in a user-visible message. Don't silently pick a different tool and justify it after the fact.

- **Negative results are results.** When an experiment falsifies a hypothesis, document what was learned and let `research_state.md` reflect it — don't keep tuning in the same step trying to produce a positive outcome.

## Environment

- Python code runs under the `panda` conda environment.
- Detect and use GPU at runtime; fall back to CPU automatically. Log the device at start of each run.
- If a package is missing, `pip install` it within `panda`.

## Independence of runs

- Each `runN/` must not read, reference, or build on sibling runs' files. The purpose is an independent trajectory for comparison across runs.
- This convention file (CLAUDE.md) is intentionally the *only* shared context. Everything downstream of it is per-run.
