# Nemotron Quant Coding Agent

A Dockerised agent that solves unseen quantitative-finance coding tasks (option pricing, curve
bootstrapping, VaR, backtests) using a fixed, organiser-hosted LLM. It runs under a hard request
budget and a read-only sandbox.

## Context

- Event: Agenthon 2026, Track 1 ("Coding"). I built the agent from 2 to 9 October 2026; the
  competition's Development phase closed on 12 October 2026.
- Built solo. Only Development-phase submissions were made. Two Development-phase platform runs
  scored 18/86 (v6) and 16/86 (v7), as noted in `track1-agent/agent/config.py`; no final-phase
  score or rank exists.
- Organiser material is not included in this repository. See the
  [Track 1 public kit](https://github.com/Agenthon-2026/track1-coding-public) (86 public units,
  checkers, sandbox image) and the
  [competition hub and toolkit](https://github.com/Agenthon-2026/Agenthon2026-public).

## Problem

Each "unit" is a quant-finance task: a natural-language instruction, input files (csv, parquet,
xlsx, json, zip), and a list of required deliverables. Hidden pytest checkers grade each unit.
They usually pin oracle numbers at 1e-6 tolerance, exact JSON key sets and types, and exact
column order. The metric is mean pass@1 with a fixed denominator, where a unit passes only if
every check passes. The constraints are:

| Constraint | Value |
|---|---|
| Model | organiser "House" endpoint only: `nvidia/nemotron-3-super-120b-a12b` (reasoning shares the output cap) |
| Model budget | 25 admitted requests per unit, at most 4,000 output tokens per request; failures and retries are charged |
| Clock | 12 h for the whole 86-unit roster, about 8 min per unit |
| Sandbox | read-only root FS, uid 65534, 64 MiB noexec `/tmp`, no network except the model route |

## Approach

The pipeline is in `track1-agent/agent/` (19 modules, about 3.2k lines of Python):

```
ingest      instruction, card, file inventory, unit-image -> /input path map, canary scrub
profile     deterministic profile of every input (schema, dtypes, ranges, samples)    profile.py
contract    deliverable contract from the instruction: files, JSON key paths + types,
            columns, row counts                                                       contract.py
PLAN        House model, thinking on: method, traps-and-decisions ledger, self-checks
CODE        House model, thinking off: full program; continued if truncated
loop        run in a stage dir -> validate against the contract -> keep best attempt
            -> REPAIR (SEARCH/REPLACE edits) or REWRITE on stagnation                 validate.py, codeio.py
REVIEW      literal-compliance audit of clean outputs vs the instruction
finalize    promote the best attempt, sanitise the output tree, always exit 0         finalize.py
```

Design points:

- **Budget accounting.** The client charges each request before sending it, caps itself at 16 of
  the 25 allowed requests, and never retries blindly (`llm.py`, `config.py`).
- **Contract-driven validation.** Missing or extra keys, wrong JSON types, wrong column order and
  wrong row counts are found deterministically before any model call is spent on review
  (`contract.py`, `validate.py`).
- **Casebook.** Retrieval returns only the agent's own passing solutions to other public units
  (`assets/casebook`). It excludes the target unit by id and also by instruction similarity, as
  competition rule 8 requires (`casebook.py`, `dev/build_casebook.py`).
- **Platform-faithful harness.** `dev/bench.py` reproduces the platform's container flags, hides
  `checks/` from the agent, and grades each output with the unit's own checker.

## Results

All results in the table are local pass@1 on the public units, with checkers hidden from the
agent. They are not official leaderboard scores. On the platform itself, two Development-phase
submissions scored 18/86 = 0.209 (v6, 450 s per-unit cap) and 16/86 = 0.186 (v7, 400 s cap); these
figures are recorded in a comment in `track1-agent/agent/config.py`, and the platform logs are not
committed.

| Run | Units | pass@1 | Mean agent time | Log |
|---|---|---|---|---|
| panel-v3 | 16-unit dev panel | 2/16 = 0.125 | 328 s | `benchmarks/panel-v3.log` |
| panel-v4 | 16-unit dev panel | 3/16 = 0.188 | 488 s | `benchmarks/panel-v4.log` |
| panel-v5 | 16-unit dev panel | 5/16 = 0.312 | 383 s | `benchmarks/panel-v5.log` |
| **full-v9** | **all 86 public units** | **17/86 = 0.198** | 313 s | `benchmarks/full-v9.log` |
| set-v10 (with casebook) | 21-unit hard/mixed subset | 6/21 = 0.286 | 468 s | `benchmarks/set-v10.log` |

How to read these numbers:

- The full-roster local figure, **0.198**, is the representative one and sits between the two
  Development-phase platform scores (0.186 and 0.209). The 16-unit panel was used for
  development, so its 0.312 is optimistic.
- full-v9 ran before the casebook existed. The casebook was then built from v9's passing
  solutions, so set-v10 is not a clean held-out measurement.
- Passes cluster in well-specified textbook tasks (backtests 3/3, tool-using data tasks 2/2);
  derivatives pricing was at the roster average (3/15) and cross-domain units scored 0/19. These counts come from joining
  `full-v9.log` with `unit_survey.csv`.
- Most failures are partial, not crashes. A typical failed unit passes many checks but misses a
  few oracle values or an exact schema detail, and pass@1 gives no credit for that.
- `panel-v2`, `full-v8` and `full-v9-450` are aborted or partial runs, kept for completeness.
  full-v9 was resumed once after 6 units, and one unit was re-run after a network error (it failed
  both times).

## Repository layout

```
track1-agent/
  agent/          the agent (controller, llm client, contract, profile, validate, codeio, casebook, ...)
  assets/casebook rule-8 casebook: the agent's own passing solutions + index (no unit data)
  dev/            benchmark harness, mock House server, offline evaluators, casebook builder
  scripts/        submission packaging and registry push (credentials read from local files, not in repo)
  tests/          unit tests for the deterministic components
  benchmarks/     summary logs of local runs (per-unit run dumps are not committed)
  Dockerfile      agent image on the organisers' sandbox base
  README.md       design notes: platform facts and how each one is handled
```

## How to run

You need Docker and a clone of the organiser kit next to `track1-agent/` (as `../track1-coding-public`).

```bash
cd track1-agent
python -m pytest tests -q                      # 13 unit tests, no model or Docker needed

docker build -t finance-bench-sandbox:latest -f ../track1-coding-public/docker/sandbox.Dockerfile ../track1-coding-public
docker build --platform linux/amd64 -t t1-agent:dev .

python dev/bench.py --image t1-agent:dev --units panel            # 16-unit dev panel
python dev/bench.py --image t1-agent:dev --units all --jobs 2     # full public roster
```

`dev/bench.py` reads the model endpoint and key from `../extra_info/keys/` (not included). It
defaults to the NVIDIA API with the same Nemotron model. `dev/mock_house.py` provides an offline
mock endpoint for smoke tests.

## Notes and limitations

- 0.198 on the public roster is a modest score. The binding constraints were the 4,000-token output
  cap, which reasoning tokens also consume, and strict oracle tolerances. Many failed units pass
  most of their checks but miss a few exact values.
- The numbers come from a single run each, with no confidence intervals. Run-to-run variance from
  the model is material at this sample size.
- Agent code is Apache-2.0 (`track1-agent/LICENSE`). The task material it was developed against is
  CC BY-NC 4.0 and remains in the organisers' repositories.
