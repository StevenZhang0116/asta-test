---
name: analyze-belief-evolution
description: Summarize what was done in each research step and its outcome, tracking how beliefs, hypotheses, and conclusions evolved across snapshots. Produces a PDF report with a visual flowchart of the research trajectory.
---

You are analyzing the evolution of a research agent's work across all steps. Your goals are:
1. For EACH research step, summarize (a) what was done and (b) what the outcome was.
2. Track how hypotheses/beliefs evolved step-by-step — including any self-corrections, flipped conclusions, or bug-induced artifacts that were later rejected.
3. Summarize the correct scientific conclusions that survived scrutiny.
4. Produce a PDF report with a visual flowchart of the research reasoning path.

## Skills to use

- **`academic-plotting`**: use this to generate the research-trajectory flowchart as a publication-quality figure (architecture/flow diagram). Pass it the per-step summaries plus the branch/correction structure; let it pick the right renderer (Gemini diagram or matplotlib).
- **`preview`**: use this to render the final Markdown report to PDF. Do NOT hand-roll HTML→PDF conversion; `preview` handles citations, formatting, and embedded figures correctly and matches the rest of this project's document pipeline.

## Instructions

1. **Read the full history.** Load every `history/research_state_bk*.md` (in numerical order), the current `research_state.md`, and `logbook.md`. Also skim `experiments/` and any `design_step*.md` / `literature_report_step*.md` files for per-step context.

2. **Per-step summary (primary output).** For each research step N (numbered by the backup files and logbook entries), produce:
   - **What was done:** the concrete actions — literature review, experiment designed, code written, analysis performed. Reference the relevant experiment ID(s) (e.g. `Exp01`).
   - **Outcome:** the result in plain language — what was measured, what hypothesis it bore on, and whether it confirmed, disconfirmed, or left open the question.
   - **Belief delta:** how the agent's picture of the problem changed at the end of this step (e.g., confidence in H1 went 55% → 30%; new hypothesis H3 introduced).
   - **Links forward:** what the outcome motivated next.

3. **Hypothesis trajectory table.** Across all steps, track each hypothesis's:
   - Step it was introduced
   - Confidence trajectory across snapshots (e.g., 55% → 45% → 30% → FALSE)
   - Final status (confirmed, disconfirmed, open)

4. **Self-corrections and bug-induced artifacts.** Call out:
   - Hypotheses that FLIPPED (the agent believed something, then reversed)
   - Experimental results later attributed to bugs or artifacts, and what false conclusions were briefly drawn
   - How the agent detected and corrected the error

5. **Correct scientific conclusions.** Summarize the final validated picture:
   - Which hypotheses were ultimately CONFIRMED and what they mean
   - How the understanding built up incrementally (first X, which led to Y, which revealed Z)
   - Key quantitative results that are trustworthy

6. **Flowchart.** Use the `academic-plotting` skill to generate `history/belief_flowchart.pdf` (or `.png` if PDF isn't supported by the chosen renderer — the `preview` step will embed it either way). The flowchart should show:
   - The reasoning path through experiments (Exp01 → Exp02 → … → ExpN)
   - At each node: what was tested and what was concluded
   - Branch points where the agent's belief diverged (wrong path vs. correct path)
   - Dead ends (disconfirmed hypotheses) in red/gray
   - The correct path in green
   - Bug-induced detours in a distinct style (dashed orange)
   - Arrows showing how one conclusion led to the next experiment

7. **Write the Markdown report** to `history/belief_evolution_summary.md` with this structure:
   - Title: "Belief Evolution Analysis"
   - Introduction: what this document tracks
   - **Step-by-Step Summary** (the primary section): for each step, the "what was done / outcome / belief delta / links forward" block from (2)
   - Embedded flowchart figure from (6)
   - **Scientific Conclusions** section summarizing validated findings
   - **Hypothesis Trajectory** table from (3)
   - **Self-Corrections** section with narrative from (4)
   - **Bug-Induced Artifacts** section from (4)
   - **Lessons Learned** section on research epistemics

8. **Render to PDF.** Invoke the `preview` skill on `history/belief_evolution_summary.md` to produce `history/belief_evolution_summary.pdf`. Verify the PDF was written before reporting done.

## Constraints

- Do NOT modify any existing files (`research_state_bk*.md`, `research_state.md`, `logbook.md`, `experiments/*`, etc.).
- Only CREATE or OVERWRITE: `history/belief_evolution_summary.md`, `history/belief_evolution_summary.pdf`, and `history/belief_flowchart.{pdf,png}`.
- The PDF is the deliverable; the Markdown source is kept alongside so the report can be re-rendered or edited.
- Python code, if any ad-hoc scripting is still needed outside the skills, runs under the `panda` conda env (`/home/zihan.zhang/.conda/envs/panda/bin/python`).
