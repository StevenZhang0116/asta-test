---
name: analyze-belief-evolution
description: Analyze the belief trajectory across all research_state snapshots to identify self-corrections, flipped hypotheses, bug-induced artifacts, and summarize the correct scientific conclusions with a visual flowchart.
---

You are analyzing the evolution of scientific beliefs in a research agent's reasoning. Your goals are:
1. Identify cases where the agent drew conclusions that it later realized were wrong (due to bugs, artifacts, or flawed reasoning).
2. Summarize the CORRECT scientific conclusions that survived scrutiny and how they evolved through iterations.
3. Produce a structured HTML report with a visual flowchart of the research reasoning path.

## Instructions

1. Read ALL snapshot files in `history/research_state_bk*.md` (in numerical order) plus the current `research_state.md` and `logbook.md`.

2. For each hypothesis (H1 through H11+), track:
   - Which snapshot introduced it
   - Confidence trajectory across snapshots (e.g., 55% → 45% → 30% → FALSE)
   - Final status (confirmed, disconfirmed, open)

3. Identify SELF-CORRECTIONS:
   - Hypotheses that FLIPPED (agent believed something, then reversed)
   - Experimental results later attributed to bugs or artifacts
   - Cases where the agent's reasoning changed direction

4. Identify BUG-INDUCED ARTIFACTS:
   - Experimental results initially trusted but later found to be implementation errors
   - What false conclusions were drawn from buggy results
   - How the agent detected and corrected the error

5. Summarize CORRECT SCIENTIFIC CONCLUSIONS:
   - Which hypotheses were ultimately CONFIRMED and what they mean
   - How the understanding built up incrementally (e.g., "first we learned X, which led us to test Y, which revealed Z")
   - The final validated scientific picture — what is now known to be true
   - Key quantitative results that are trustworthy

6. Create a VISUAL FLOWCHART by writing and running a Python script:
   - Use `/home/zihan.zhang/.conda/envs/panda/bin/python` to run the script
   - Use `matplotlib` and `networkx` (install with pip if needed) to generate the flowchart as SVG
   - The flowchart should show:
     - The reasoning path through experiments (Exp01 → Exp02 → ... → Exp04)
     - At each node: what was tested, what was concluded
     - Branch points where the agent's belief diverged (wrong path vs correct path)
     - Dead ends (hypotheses that were disconfirmed) shown in red/gray
     - Correct path shown in green
     - Bug-induced detours shown with a distinct style (e.g., dashed orange)
     - Arrows showing how one conclusion led to the next experiment
   - Save the figure as `history/belief_flowchart.svg`
   - Also embed it as base64 in the final HTML

7. Write the output as a standalone HTML file saved to `history/belief_evolution_summary.html` with:
   - Title: "Belief Evolution Analysis"
   - Introduction explaining what this document tracks
   - A "Scientific Conclusions" section summarizing validated findings
   - The visual flowchart (embedded as base64 SVG/PNG)
   - A table showing each hypothesis's confidence across all snapshots
   - A "Self-Corrections" section with detailed narrative
   - A "Bug-Induced Artifacts" section
   - A "Lessons Learned" section about research epistemics
   - Clean inline CSS styling (professional, readable)

## Environment

- Python: `/home/zihan.zhang/.conda/envs/panda/bin/python`
- Pip: `/home/zihan.zhang/.conda/envs/panda/bin/pip`
- Install any missing packages (e.g., networkx, matplotlib) via pip before running

## Constraints

- Do NOT modify any existing files (research_state_bk*.md, research_state.md, logbook.md, experiments/*, etc.)
- Only CREATE or OVERWRITE: `history/belief_evolution_summary.html` and `history/belief_flowchart.svg`
- Temporary Python scripts can be written to a temp location and cleaned up after
- The final HTML must be self-contained (the SVG should be embedded inline or as base64 so it renders without external files)
