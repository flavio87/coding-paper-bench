# AGENTS.md

---

## RULE 1 – ABSOLUTE (DO NOT EVER VIOLATE THIS)

You may NOT delete any file or directory unless I explicitly give the exact command **in this session**.

- This includes files you just created (tests, tmp files, scripts, etc.).
- You do not get to decide that something is "safe" to remove.
- If you think something should be removed, stop and ask. You must receive clear written approval **before** any deletion command is even proposed.

Treat "never delete files without permission" as a hard invariant.

---

## Irreversible Git & Filesystem Actions

Absolutely forbidden unless you give the **exact command and explicit approval** in the same message:

- `git reset --hard`
- `git clean -fd`
- `rm -rf`
- Any command that can delete or overwrite code/data

Rules:

1. If you are not 100% sure what a command will delete, do not propose or run it. Ask first.
2. Prefer safe tools: `git status`, `git diff`, `git stash`, copying to backups, etc.
3. After approval, restate the command verbatim, list what it will affect, and wait for confirmation.
4. When a destructive command is run, record in your response:
   - The exact user text authorizing it
   - The command run
   - When you ran it

If that audit trail is missing, then you must act as if the operation never happened.

---

## 🚨 MANDATORY: READ THIS FIRST 🚨

### Before ANY Code Changes

```bash
bd prime                    # Load beads workflow context
bd ready --json             # Check for ready work to claim
```

### Before Saying "Done" or "Complete"

```bash
git status                  # Check what changed
git add <files>             # Stage code changes
ubs --only=python <changed-files>   # UBS scan (exit 0 = safe)
bd sync                     # Commit beads changes
git commit -m "..."         # Commit code
bd sync                     # Commit any new beads changes
git push                    # Push to remote
```

**⚠️ Work is NOT done until pushed. NEVER skip this checklist.**

### Quick Command Reference

| Action | Command |
|--------|---------|
| Find ready work | `bd ready` or `bd ready --json` |
| Claim work | `bd update <id> --status=in_progress` |
| Create issue | `bd create "Title" -t task -p 2` |
| Close issues | `bd close <id1> <id2> ...` |
| Sync to git | `bd sync` |

---

## Project Overview

**coding-paper-bench** is a research project for testing LLM-guided evolutionary code optimization using the ShinkaEvolve framework. The project benchmarks how well frontier LLMs can improve algorithmic code through evolutionary search.

### Key Components

| Directory | Purpose |
|-----------|---------|
| `ShinkaEvolve/` | Modified ShinkaEvolve framework (LLM + evolutionary algorithms) |
| `experiments/` | Experiment configurations and results |
| `paperbench/` | Paper benchmark tasks |

### Running ShinkaEvolve Experiments

```bash
# Activate the ShinkaEvolve environment
cd ShinkaEvolve
source .venv/bin/activate

# Run an experiment (e.g., TSP optimization)
cd ../experiments/shinka_tsp
python run_evo.py --generations 10 --islands 2

# Visualize results
shinka_visualize results_new/ --port 8888
```

---

## UBS (Ultimate Bug Scanner)

**Golden rule:** run `ubs <changed-files>` before every commit. Exit 0 = safe; exit >0 = fix & re-run.

### Commands (fast path)

```bash
ubs --only=python file.py file2.py              # Specific files (< 1s) — USE THIS
ubs --only=python $(git diff --name-only --cached)  # Staged files — before commit
ubs --only=python src/                          # Language filter (fast)
ubs --ci --fail-on-warning --only=python .      # CI mode — before PR
```

### Fix workflow
1. Read finding → category + suggested fix
2. Navigate `file:line:col`
3. Verify it's real (not false positive)
4. Fix root cause (not symptom)
5. Re-run `ubs <file>` → exit 0
6. Commit

**Speed critical:** scope to changed files. Never full scan for small edits.

---

## Issue Tracking with bd (beads)

**IMPORTANT**: This project uses **bd (beads)** for ALL issue tracking. Do NOT use markdown TODOs, task lists, or other tracking methods.

### Why bd?

- **Dependency-aware**: Track blockers and relationships between issues
- **Git-friendly**: Auto-syncs to JSONL for version control
- **Agent-optimized**: JSON output, ready work detection, discovered-from links
- **Prevents duplicate tracking systems** and confusion

### Session Bootstrap

- Run `bd prime` at session start to inject workflow context
- Use `bd ready --json` to list candidate tasks
- Use `bd list`, `bd show`, `bd dep tree` etc. for deeper inspection

---

## Essential bd Commands

### Finding Work

```bash
bd ready                           # Show issues ready to work (no blockers)
bd ready --json                    # Same, but JSON for programmatic use
bd list --status=open              # All open issues
bd list --status=in_progress       # Your active work
bd show <id>                       # Detailed issue view with dependencies
bd blocked                         # Show all blocked issues
```

### Creating Issues

```bash
bd create --title="..." --type=task|bug|feature --priority=2 --json
bd create "Issue title" -t bug -p 1 --json                    # Short form
bd create "Subtask" --parent <epic-id> --json                 # Hierarchical subtask
bd create "Found bug" -p 1 --deps discovered-from:bd-123      # Link to parent issue
bd q "Quick capture"                                           # Quick capture, outputs only ID
```

**Issue Types:**
- `bug` - Something broken
- `feature` - New functionality
- `task` - Work item (tests, docs, refactoring)
- `epic` - Large feature with subtasks
- `chore` - Maintenance (dependencies, tooling)

**Priorities (use numbers 0-4, NOT "high"/"medium"/"low"):**
- `0` - Critical (security, data loss, broken builds)
- `1` - High (major features, important bugs)
- `2` - Medium (default, nice-to-have)
- `3` - Low (polish, optimization)
- `4` - Backlog (future ideas)

### Updating Issues

```bash
bd update <id> --status=in_progress    # Claim work
bd update <id> --status=open           # Unclaim
bd update <id> --priority=1            # Change priority
```

### Completing Work

```bash
bd close <id>                          # Mark complete
bd close <id> --reason="explanation"   # Close with reason
bd close <id1> <id2> ...               # Close multiple issues at once
```

### Sync & Collaboration

```bash
bd sync                                # Sync with git remote (run at session end!)
bd sync --status                       # Check sync status without syncing
```

---

## Using bv (Beads Viewer)

bv is a graph-aware triage engine for Beads projects. Use robot flags for deterministic, dependency-aware outputs.

**CRITICAL: Use ONLY `--robot-*` flags. Bare `bv` launches an interactive TUI that blocks your session.**

### The Workflow: Start With Triage

```bash
bv --robot-triage        # THE MEGA-COMMAND: start here
bv --robot-next          # Minimal: just the single top pick + claim command
bv --robot-plan          # Parallel execution tracks
```

### All Robot Commands

| Command | Returns |
|---------|---------|
| `--robot-triage` | Everything: quick_ref, recommendations, quick_wins, blockers, health |
| `--robot-next` | Single top pick with claim command |
| `--robot-plan` | Parallel execution tracks with `unblocks` lists |
| `--robot-insights` | Full metrics: PageRank, betweenness, critical path, cycles |

---

## Package Management & Environments

- Always use `uv` to ensure a separate clean environment for each project
- Use Python 3.11+ for ShinkaEvolve compatibility
- Use `.env` file to pull in secrets

### Environment Setup

```bash
# For ShinkaEvolve experiments
cd ShinkaEvolve
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

---

## LLM Usage

- Use OpenRouter for LLM calls
- Load `OPENROUTER_API_KEY` from `.env` file

### 🚨 ALWAYS Run LLM Calls in Parallel 🚨

**CRITICAL**: When making multiple LLM API calls, ALWAYS use `asyncio.gather()` to run them in parallel.

```python
# ❌ BAD - Sequential calls (SLOW!)
results = []
for item in items:
    result = await call_llm(item)
    results.append(result)

# ✅ GOOD - Parallel calls (FAST!)
tasks = [call_llm(item) for item in items]
results = await asyncio.gather(*tasks)
```

---

## Web Searches & Documentation

### 🚨 CRITICAL: Always Use Current Date Context 🚨

**Before performing any web search for documentation:**
1. **First**, note today's date
2. **Then**, include the year (2025+) in your search query

```bash
# ❌ BAD - No date context
"ShinkaEvolve documentation"

# ✅ GOOD - Include year
"ShinkaEvolve evolutionary optimization 2025"
```

---

## MCP Agent Mail: coordination for multi-agent workflows

### Session startup workflow

```bash
# 1. Ensure project and register
ensure_project(human_key="/data/projects/coding-paper-bench")
register_agent(project_key="...", program="claude-code", model="claude-opus-4.5", task_description="Your task")

# 2. Check inbox for any coordination messages
fetch_inbox(project_key="...", agent_name="YourAgentName", include_bodies=true)

# 3. Optional: Reserve files you plan to edit
file_reservation_paths(project_key="...", agent_name="YourAgentName", paths=["experiments/**"], ttl_seconds=3600, exclusive=true, reason="Running experiment")
```

### Project key for this repo
`/data/projects/coding-paper-bench`

---

## Integrating Beads with Agent Mail

- **Single source of truth**: Use **Beads** for task status/priority/dependencies; use **Agent Mail** for conversation and audit trail
- **Shared identifiers**: Use the Beads issue id as the Mail `thread_id`
- **Reservations**: When starting a task, reserve affected paths

---

## Code Editing Discipline

- Do **not** run scripts that bulk-modify code
- Large mechanical changes: break into smaller, explicit edits
- The bar for adding new files is very high. Prefer editing existing files.
- No "compat shims" or "v2" file clones

---

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

---

## cass — Cross-Agent Search

`cass` indexes prior agent conversations so we can reuse solved problems.

**Rules:** Never run bare `cass` (TUI). Always use `--robot` or `--json`.

```bash
cass doctor --json                              # Diagnose index + connectors
cass index                                      # Refresh local index
cass search "TSP optimization" --robot --limit 5
cass capabilities --json
```

---

## Memory System: cass-memory (cm)

The Cass Memory System gives agents procedural memory.

```bash
# Get context before starting a task
cm context "<task description>" --json

# Manage playbook
cm playbook list
cm playbook add "Your rule content"
```

---

## ShinkaEvolve Experiment Workflow

### Running an Experiment

```bash
cd /data/projects/coding-paper-bench/experiments/shinka_tsp
source ../../ShinkaEvolve/.venv/bin/activate

# Run evolution
python run_evo.py --generations 20 --islands 2 --results_dir results_experiment_name

# Visualize (accessible via Tailscale)
shinka_visualize results_experiment_name/ --port 8888
```

### Experiment Reports

After each experiment, create a report in the results directory:
- `report_<experiment_name>.md`
- Include: hypothesis, baseline, evolved solution, comparison to SOTA

### Key Metrics

| Metric | Description |
|--------|-------------|
| Score | Aggregate fitness (higher = better) |
| Validity Rate | % of generations producing correct code |
| Gap from Optimal | How far from theoretical/known optimal |
| API Cost | Total OpenRouter spend |

---

## Morph Warp Grep — AI-Powered Code Search

Use `mcp__morph-mcp__warpgrep_codebase_search` for "how does X work?" discovery.

```
mcp__morph-mcp__warpgrep_codebase_search(
  repo_path: "/data/projects/coding-paper-bench",
  search_string: "How does ShinkaEvolve select parents?"
)
```

---

## Important Rules Summary

- ✅ Use bd for ALL task tracking
- ✅ Always use `--json` flag for programmatic use
- ✅ Run `bd sync` at end of sessions
- ✅ Push to remote before ending session
- ✅ Create experiment reports for each run
- ❌ Do NOT create markdown TODO lists
- ❌ Do NOT delete files without explicit permission
- ❌ Do NOT skip the landing-the-plane checklist
