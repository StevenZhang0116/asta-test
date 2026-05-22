"""Defer the bootstrap literature_review in research-step.

Minimal intervention: after `init` + `plan` bootstrap have created the epic
and the three frontier tasks (scope, definitions, literature_review), this
script closes the bootstrap `literature_review` (LR0) with a schema-valid
placeholder output that records the deferral. Everything else follows the
standard research-step pipeline:

  - replan from LR0 spawns a hypothesis (per the standard rule)
  - the agent drafts the hypothesis cold per mission.md Constraints
  - the experiment_design -> evidence_gathering -> analysis chain runs as usual
  - the post-hoc novelty-audit literature_review is created and run as
    mission.md mandates

There is nothing else to orchestrate here — the deferral policy itself lives
in mission.md, and the agent driving research-step honors it.

Prereqs:
    bd, jq installed; .beads/ initialized; bootstrap `plan` already run so
    LR0 exists; scope and definitions tasks have been executed and closed
    (so LR0 is the next ready task). Running this while LR0 is still blocked
    is technically possible but skips intended ordering — don't.

Run:
    python defer_lr0.py

Practical workflow (from this run's directory, with `conda activate panda`):

    # 1. Open Claude Code; drive research-step through Phase 2.
    #    In the Claude Code session, prompt the agent to:
    #      - init the research              (creates .beads/, installs bd/jq)
    #      - run plan                       (bootstrap: epic + scope +
    #                                        definitions + LR0)
    #      - execute the next ready task    (closes scope)
    #      - execute the next ready task    (closes definitions)
    claude

    # 2. With scope and definitions closed, run the intervention. This can
    #    be done from a separate shell or from Claude Code's Bash tool in
    #    the same session — either works.
    python defer_lr0.py

    # 3. Resume research-step in Claude Code:
    #      - run plan                       (replan from LR0 → spawns one
    #                                        hypothesis, left OPEN by the
    #                                        DEFERRED-LR sentinel gap)
    #      - execute the hypothesis         (drafted cold per mission.md)
    #      - continue: experiment_design →
    #                  evidence_gathering →
    #                  analysis →
    #                  post-hoc novelty-audit literature_review →
    #                  ... → synthesis
    claude

Do NOT rerun this script for the post-hoc novelty-audit literature_review.
That LR is a real execute, not a deferral. The stage guard in find_lr0()
refuses once any hypothesis exists, which is exactly the right behavior —
this script is only for the bootstrap LR.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent


def _resolve_skill_root() -> Path:
    """Locate the research-step skill directory.

    Resolution order:
      1. RESEARCH_STEP_SKILL_ROOT env var (explicit override)
      2. Claude Code plugin install under ~/.claude/plugins/marketplaces/
      3. Project-tree clone (walk up from RUN_DIR looking for asta-plugins/)

    Raises if none of these resolve to an existing validate-output.sh.
    """
    candidates: list[Path] = []

    env_root = os.environ.get("RESEARCH_STEP_SKILL_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    candidates.append(
        Path.home()
        / ".claude/plugins/marketplaces/asta-plugins"
        / "plugins/asta/skills/research-step"
    )

    for ancestor in [RUN_DIR, *RUN_DIR.parents]:
        candidates.append(
            ancestor / "asta-plugins/plugins/asta/skills/research-step"
        )

    for path in candidates:
        if (path / "scripts/validate-output.sh").is_file():
            return path

    raise RuntimeError(
        "Could not locate research-step skill. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
        + "\nSet RESEARCH_STEP_SKILL_ROOT to override."
    )


SKILL_ROOT = _resolve_skill_root()

PLACEHOLDER_BACKGROUND = (
    "Bootstrap literature_review intentionally deferred.\n\n"
    "Per mission.md Constraints (hypothesis-first, post-hoc novelty audit), "
    "this run drafts hypotheses cold from first principles and runs the "
    "novelty audit only after the corresponding analysis closes. An upfront "
    "literature_review would bias hypothesis generation toward existing "
    "methods.\n"
)

PLACEHOLDER_OUTPUT = {
    "summary_path": "background_knowledge.txt",
    "key_findings": [
        "[DEFERRED-LR] Bootstrap literature_review deferred to post-hoc "
        "novelty audit (mission.md Constraints: hypothesis-first). Cold "
        "hypotheses must be drafted from first principles, with no "
        "literature exposure and no use of literature-driven hypothesis "
        "skills (e.g., generate-theories). Each cold hypothesis is audited "
        "downstream after its analysis closes, with inputs[] including "
        "both the hypothesis and its closed analysis. Downstream tooling "
        "should treat the [DEFERRED-LR] prefix in key_findings[0] as the "
        "authoritative signal that this is the bootstrap deferral and not "
        "a real audit."
    ],
    "gaps": [
        "[DEFERRED-LR sentinel — do not auto-resolve.] One cold hypothesis "
        "must be drafted from first principles, without literature exposure, "
        "per mission.md Constraints. This gap intentionally provides no "
        "statement, rationale, falsifiable prediction, or expected-evidence "
        "material; plan must leave the spawned hypothesis task open for "
        "cold drafting via the standard 'too thin to auto-resolve' fallback "
        "(see plan.md). The novelty audit is created post-hoc, after the "
        "corresponding analysis closes, with inputs[] referencing both."
    ],
    "citations": [],
}


def bd(*args: str) -> str:
    return subprocess.run(
        ["bd", *args], cwd=RUN_DIR, capture_output=True, text=True, check=True
    ).stdout


def find_lr0() -> tuple[str, list[str]]:
    """Return (LR0 id, its inputs[]) for the open bootstrap literature_review.

    Stage guards, in order:
      1. No hypothesis task may exist yet — if any does, the only open
         literature_review is a post-hoc novelty audit, not the bootstrap LR.
      2. The candidate LR's inputs[] must be exactly {scope_id, definitions_id}
         (one of each, total length 2). Bootstrap `plan` always emits this
         shape (plan.md:49); a different shape signals a non-bootstrap LR.
      3. Both input tasks (scope and definitions) must already be closed.
         The bd graph's `blocks` edges enforce this for `bd ready`, but this
         script touches metadata directly, so we re-verify here.
    """
    issues = json.loads(bd("list", "--all", "--json"))
    by_id = {i["id"]: i for i in issues}

    def task_type(issue: dict) -> str | None:
        return ((issue.get("metadata") or {}).get("research_step") or {}).get(
            "task_type"
        )

    if any(task_type(i) == "hypothesis" for i in issues):
        raise RuntimeError(
            "Refusing to defer: hypothesis tasks already exist, so any open "
            "literature_review is a post-hoc novelty audit, not the bootstrap LR."
        )

    scope_ids = {i["id"] for i in issues if task_type(i) == "scope"}
    definitions_ids = {i["id"] for i in issues if task_type(i) == "definitions"}
    bootstrap_ids = scope_ids | definitions_ids

    candidates: list[tuple[str, list[str]]] = []
    for issue in issues:
        meta = (issue.get("metadata") or {}).get("research_step") or {}
        inputs = meta.get("inputs", [])
        if (
            meta.get("task_type") == "literature_review"
            and issue.get("status") != "closed"
            and len(inputs) == 2
            and set(inputs) == bootstrap_ids
            and any(i in scope_ids for i in inputs)
            and any(i in definitions_ids for i in inputs)
        ):
            candidates.append((issue["id"], inputs))

    if not candidates:
        raise RuntimeError(
            "No open bootstrap literature_review found. Has `plan` bootstrap "
            "run yet (creating LR with inputs == {scope, definitions}), or "
            "has the bootstrap LR already been closed?"
        )
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous: multiple open bootstrap literature_review candidates "
            f"({[c[0] for c in candidates]})."
        )

    lr0_id, inputs = candidates[0]
    not_closed = [
        iid
        for iid in inputs
        if (by_id.get(iid) or {}).get("status") != "closed"
    ]
    if not_closed:
        raise RuntimeError(
            f"Refusing to defer LR0 ({lr0_id}): input tasks must be closed "
            f"first, but these are still open: {not_closed}. Run `execute` "
            "on scope and definitions before invoking this script."
        )

    return lr0_id, inputs


def main() -> None:
    lr0_id, inputs = find_lr0()
    print(f"[lr0] {lr0_id} inputs={inputs}")

    (RUN_DIR / "background_knowledge.txt").write_text(PLACEHOLDER_BACKGROUND)

    meta = {
        "research_step": {
            "task_type": "literature_review",
            "inputs": inputs,
            "output_schema_version": 1,
            "output": PLACEHOLDER_OUTPUT,
        }
    }
    meta_file = RUN_DIR / ".tmp_lr0_placeholder.json"
    meta_file.write_text(json.dumps(meta, indent=2))

    subprocess.run(
        [str(SKILL_ROOT / "scripts/validate-output.sh"),
         "literature_review", str(meta_file)],
        cwd=RUN_DIR, check=True,
    )

    bd("update", lr0_id, "--metadata", f"@{meta_file}")
    bd("close", lr0_id)
    meta_file.unlink()
    print(f"[done] {lr0_id} closed with deferred-audit placeholder. "
          "Resume research-step normally (replan from LR0).")


if __name__ == "__main__":
    main()
