# Agenthon 2026 Track 1 agent

One Docker image that solves unseen quantitative-finance coding units with the organizers'
House model at run time: `solve --task-dir /input --out /app/output`.

## Facts the design is built on (verified against the kit, not assumed)

| Fact | Consequence in the agent |
|---|---|
| House model = `nvidia/nemotron-3-super-120b-a12b`, thinks by default; reasoning shares the 4000-token output cap | plan/repair/review run with thinking ON (with salvage if thinking eats the cap); code generation runs with thinking OFF; truncated code is continued |
| 25 admitted requests/unit, retries and upstream failures are charged, 401/403 are not | client charges before sending, caps itself at 16 by default, never retries blindly |
| Graders mostly pin oracle numbers (1e-6 tolerances common), exact key sets / column order, JSON bool/int types, exact RNG streams | plan has a mandatory "traps and decisions" ledger; validator flags missing/extra keys, wrong types, row counts; reviewer audits literal compliance |
| 40/86 instructions name inputs at their unit-image path (`/app/x.csv`); evaluation mounts the unit at `/input` | path map rebuilt from `environment/Dockerfile` COPY lines; string literals in generated code are rewritten |
| `checks/` is sealed on the platform | agent never depends on it; benchmark hides it by default |
| Canary GUIDs in instruction/checks; any canary in output fails the organisers' output gate (g2) | canary lines scrubbed from prompts; output tree scanned and scrubbed |
| Read-only root FS, uid 65534, 64 MiB noexec `/tmp`, 256 PIDs, no HOME | HOME/caches in /tmp, thread caps, scripts interpreted (not exec'd) from /tmp |
| 12 h stage clock for ~86 units (~8 min each) is the binding limit | per-unit budget `min(card - 90 s, 560 s)` (full-v9 ran at 450 s), best attempt always promoted to the output |
| Output tree limits (<=256 files, <=64 MiB, no links, valid names) | attempts run in a stage dir; only the best attempt's files are promoted; final sanitizer |

## Pipeline

```
ingest (instruction, card, inventory, path map, canaries)
 -> profile every input deterministically (csv/parquet/xlsx/json/zip/text)
 -> deliverable contract from the instruction (files, JSON key paths + types, columns, rows)
 -> PLAN   (House, thinking)   deliverables / inputs / method / traps / self-checks
 -> CODE   (House, no thinking) complete program; continuation if truncated
 -> loop:  run in stage dir -> validate (errors / self-check fails / warnings) -> keep best
           -> REPAIR (House, thinking, SEARCH/REPLACE) | REWRITE on stagnation or failed edits
 -> REVIEW (House, thinking) of clean outputs vs the instruction -> fixes re-validated
 -> sanitize output tree -> exit 0 (never crashes)
```

## Rules compliance

* Rule 1/network: only `$MODEL_ENDPOINT/v1/chat/completions` with `$MODEL_TOKEN`; proxies read from
  env; no vendor tools; generated code may not use the network (static check, credentials removed
  from its environment).
* Rule 8: the casebook (`assets/casebook`) holds only the agent's own passing solutions; retrieval
  excludes the target unit by id AND by instruction similarity (content match), so a unit can never
  receive its own solution.
* Rule 9: every solution is written by the House model during the run; there is no stored answer.
* No task data is vendored into the image (`.dockerignore` admits only `agent/`, `assets/`, `LICENSE`).

## Commands

```bash
# base image (once) and agent image
docker build -t finance-bench-sandbox:latest -f ../track1-coding-public/docker/sandbox.Dockerfile ../track1-coding-public
docker build --platform linux/amd64 -t t1-agent:dev .

# benchmark (platform-identical container flags, checks hidden, graded by each unit's real checker)
python dev/bench.py --image t1-agent:dev --units panel            # 16-unit dev panel
python dev/bench.py --image t1-agent:dev --units all --jobs 2     # full public roster

# offline tools
python dev/eval_contract.py ../track1-coding-public/units          # deliverable extraction accuracy
python dev/eval_profile.py  ../track1-coding-public/units          # profiler speed / prompt size / canary leak
python dev/build_casebook.py benchmarks/runs/<tag>                 # rule-8 casebook from passing runs (run dir created by dev/bench.py)

# submission (keys are read by path from ../extra_info/keys, never printed)
python scripts/package_submission.py --registry docker.io --repository <user>/agenthon-t1 --digest sha256:<...>
```

Environment knobs (all optional): `AGENT_MAX_SECONDS` (560), `AGENT_REQUEST_CAP` (16),
`AGENT_EXEC_TIMEOUT` (300), `AGENT_MAX_REPAIRS` (6), `AGENT_REVIEW_PASSES` (1),
`AGENT_USE_CHECKS` (1), `AGENT_USE_CASEBOOK` (1), `AGENT_DEBUG_DIR` (unset).
