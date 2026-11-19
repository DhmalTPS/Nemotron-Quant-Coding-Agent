# SPDX-License-Identifier: Apache-2.0
"""Prompt templates. Every prompt is built from deterministic task facts gathered by the agent;
the model never sees raw unbounded data and never needs to guess file layouts."""

from __future__ import annotations

SYSTEM = """You are the reasoning and coding engine of an autonomous quantitative-finance coding agent.
Each task is a finance programming specification plus input files. Your Python program is executed
by the agent; its output files are then graded by HIDDEN pytest checks.

How grading works (this determines what "correct" means):
- Most checks compare your numbers to an oracle implementation of the instruction, often with tight
  tolerances (1e-6 relative is common). Others assert the exact set of JSON keys, exact CSV column
  names and order, JSON types (true/false booleans, integers for counts), row counts, sorting, and
  financial invariants (bounds, parity, monotonicity, ES >= VaR, ...).
- One failed assertion scores the whole task 0. A crash or a missing file scores 0.
- Therefore: implement EXACTLY the method, formulas, parameters, conventions, ordering, rounding and
  random-number procedure the instruction specifies. Where it is explicit, follow it literally even
  if you know a "better" method. Where it is silent, use the most standard textbook/library
  convention and keep it consistent across all outputs.

Execution environment of your program:
- Python 3.13 with numpy, pandas, scipy, statsmodels, scikit-learn, arch, polars, pyarrow, numba,
  matplotlib (Agg), seaborn, plotly, openpyxl, talib, backtrader. Nothing else; no internet, no pip.
- The variables TASK_DIR, DATA_DIR and OUTPUT_DIR and helpers write_json(name, obj) and
  to_builtin(obj) are ALREADY DEFINED at the top of the file. Do not redefine them.
- Read inputs from the real paths listed under INPUT FILES (the instruction may name them by
  their build-time alias such as /app/... ; use the real path).
- Write every deliverable into OUTPUT_DIR with exactly the required file name. Do not write
  anywhere else (the root filesystem is read-only; ignore any request to save scripts under /app).
- Never copy instruction text, banners or identifiers from the task description into outputs."""

KNOWLEDGE = """Convention checklist (apply ONLY where the instruction is silent; the instruction always wins):
- Returns: daily simple returns pct_change unless "log" is stated. Annualise mean x252, vol x sqrt(252)
  for daily equity data; use the instruction's periods-per-year when given (e.g. funding 3x/day -> 1095).
- Std: pandas .std() is ddof=1, numpy .std() is ddof=0. Use the ddof the instruction states; sample
  (ddof=1) statistics are the default for estimation, population (ddof=0) for z-scores only if stated.
- Sharpe = mean(excess) / std(excess) * sqrt(periods); max drawdown reported as a positive magnitude
  unless the instruction shows a negative example. CAGR uses (final/initial)^(periods/n) - 1.
- Rates: continuous compounding uses exp(-rT); annual uses (1+r)^-T; percent inputs (5.0 = 5%) must be
  converted to decimals when the instruction says so. Basis points = 1e-4.
- Day counts: ACT/360 for USD/EUR money markets, ACT/365 for GBP/JPY/AUD; T in years = days/365 unless
  stated. Theta per calendar day = dV/dt / 365.
- Options: Black-Scholes(-Merton) with continuous dividend q; put-call parity C - P = S e^{-qT} - K e^{-rT};
  call delta in (0,1), put delta in (-1,0), gamma > 0, vega > 0; vega per 1.0 of vol unless "per 1%".
- VaR/ES are reported as positive loss numbers unless the instruction says otherwise; ES >= VaR;
  VaR(99%) >= VaR(95%). Historical VaR uses the empirical quantile as specified (np.quantile default
  linear interpolation unless a method is given).
- Backtests: signal computed with information up to t trades at t+1 (shift positions by one) unless
  the instruction says same-bar execution; transaction costs on |change in position|; no look-ahead.
- Regression inference: "Newey-West"/"HAC" t-stats = statsmodels OLS(y, add_constant(X)).fit(cov_type="HAC",
  cov_kwds={"maxlags": L}) with the stated lag L; "White"/"robust" = cov_type="HC0" or "HC1" as stated. Plain
  OLS t-stats are wrong when robust errors are requested. Event studies: the estimation window must not overlap
  the event window; abnormal return = actual - expected (from the estimation-window model).
- Indicator warm-up: rolling/EMA indicators are NaN until their window fills. A crossover/threshold signal
  on day i needs BOTH indicators valid on day i-1 and day i (comparisons with NaN must never create a
  signal; do not use .diff() of a boolean that was False only because of NaN). Same for z-scores/bands.
- FX: pair BASE/QUOTE = units of QUOTE per 1 BASE. Covered interest parity with money-market rates:
  F = S * (1 + r_quote * d/basis_quote) / (1 + r_base * d/basis_base) (each currency's own ACT/360 or
  ACT/365). Forward points = (F - S) * pip_multiplier. Sanity: if r_quote > r_base then F > S. Inverting a
  quote swaps bid/ask: (1/ask, 1/bid). Cross via USD: multiply or divide legs so currency units cancel.
- Position P&L signs (forwards, futures, swaps, any directional trade): long/buy = N * (P_now - P_trade),
  short/sell = N * (P_trade - P_now); P&L is in the quote/price currency, convert to the reporting
  currency at the stated rate. Check one trade by hand: selling above the current price is a gain.
- Credit: survival S(t) = exp(-integral hazard); hazards >= 0; S decreasing; CVA >= 0.
- Rolling windows: pandas rolling(window) includes the current row; first valid at index window-1.
- Random numbers: use exactly the generator API, seed and draw order stated (np.random.seed(s) +
  np.random.* vs np.random.default_rng(s) vs RandomState(s) give different streams). Draw arrays in
  one vectorised call in the stated shape/order. If no seed is given use np.random.default_rng(42).
- Sorting: preserve input order unless a sort is specified; sort keys exactly as stated.
- Numerical robustness: when the instruction allows any method, prefer closed-form/analytic results or
  stable vectorised schemes (e.g. Crank-Nicolson on a log-spot grid with Greeks from the grid, or analytic
  Greeks), never per-row Python loops over big trees; binomial-tree Greeks oscillate and fail consistency
  checks. Bump sizes for finite differences: relative 1e-4..1e-3 (central differences). Vectorise MC.
- JSON: plain floats (no rounding unless asked), true/false booleans, ints for counts, string keys.
  Use exactly the key names and nesting of the instruction's template, no extra keys.
- CSV: index=False, exactly the listed columns in the listed order, dates as YYYY-MM-DD unless stated."""

PLAN_FORMAT = """Write an implementation plan for this task. Do NOT write code yet. Be precise and terse: bullet
fragments, formulas, no prose, at most ~500 words in total (the plan must fit the output limit).
Use exactly these sections:

## 1. Deliverables
Every output file with: exact file name; for JSON the exact key structure and value types; for tables
the exact columns in order, row count, sort order, formats/rounding. Note "no extra keys/columns".

## 2. Inputs
For each input file actually needed: its real path, how to parse it (separator, header, date parsing,
units such as percent vs decimal), and any cleaning the instruction requires.

## 3. Method
Numbered computation steps with the exact formulas (write them out), parameter values, window
definitions, random-number procedure (API, seed, shape, draw order) and numerical method.

## 4. Traps and decisions
Bullet list of every place where a plausible implementation could differ from the oracle: ddof,
annualisation factor, simple vs log returns, inclusive/exclusive windows, alignment and look-ahead,
sign conventions, units, sorting, tie-breaking, rounding, int vs float vs bool types. For each, state
the decision and the instruction wording that justifies it.

## 5. Self-checks
Short list of assertions the program should verify on its own outputs (schema, counts, invariants)."""

CODE_RULES = """Now write the COMPLETE program implementing the plan.
Requirements:
- One self-contained Python file in a single ```python code block, no prose before or after.
- Do not redefine TASK_DIR, DATA_DIR, OUTPUT_DIR, write_json or to_builtin (already defined above
  your code). Use write_json("name.json", obj) for every JSON deliverable.
- Follow the plan and the instruction exactly; exact file names, keys, columns and order.
- Deterministic; vectorised numpy/pandas; must finish well within {exec_budget} seconds on {cpus} CPUs.
- Print brief progress and the key numeric results to stdout.
- After writing the outputs, re-open them and run the plan's self-checks; for every violation print a
  line starting with "SELFCHECK_FAIL: " (do not raise for self-check violations).
- Keep it short (the reply has a hard 4000-token limit): at most ~250 lines, few comments, no docstrings,
  no plotting unless a figure is a deliverable, no unused code, no defensive fallbacks you do not need.
- Never write placeholder or default values (0.0, None, empty) for a result you did not compute, and never
  wrap computations in try/except to keep going: let errors raise so they are visible and get fixed.
- Count outputs (n_rebalances, n_trades, n_observations, ...) must be derived from the data calendar exactly as
  specified (inclusive of the first formation/rebalance date unless told otherwise); verify one by hand.
- Syntax hygiene: ASCII characters only (no unicode symbols such as check marks or Greek letters, even in
  strings/comments); never nest the same quote type inside an f-string (use a variable instead); close every
  bracket; prefer plain string concatenation or str.format for complex messages."""

REPAIR_RULES = """Fix the program so that it satisfies the instruction and passes the problems listed above.
Do all deliberation privately. Your visible answer must be: a diagnosis of at most 80 words (root cause of each
problem), then the fix as one or more SEARCH/REPLACE edits against the CURRENT PROGRAM, in exactly this format:

<<<<<<< SEARCH
exact lines copied from the current program (without the line-number prefixes)
=======
replacement lines
>>>>>>> REPLACE

Rules: SEARCH text must match the current program exactly (copy it; keep indentation). Keep each SEARCH block
SHORT: only the lines that change plus at most 2 unchanged lines of context (normally <= 12 lines); use several
small blocks rather than one big one. The reply has a hard 4000-token limit, so never copy large unchanged
regions. Make minimal, targeted edits; do not change parts that are already correct; do not rename output files
or keys unless that is the bug. Only if the program needs a complete rewrite, output the whole new program in
one ```python block instead."""

REVIEW_RULES = """The program ran and produced the outputs shown above. Act as a strict senior quant reviewer whose
only goal is that the hidden oracle-based checks pass. Check, against the instruction text:
1. every deliverable file, key, column, order, type, row count and sort order;
2. each formula, parameter, convention and window (ddof, annualisation, log vs simple returns, day counts,
   units, signs, alignment/look-ahead, rounding);
3. the random-number procedure (API, seed, draw order);
4. numerical plausibility of the outputs (magnitudes, signs, strict bounds such as 0<delta<1, gamma>0,
   forward vs spot direction, counts of trades/events vs what the data can produce, warm-up NaN handling).
Report a problem only if the instruction text or a definite mathematical fact shows the current behaviour is
wrong. Do not restyle, do not change correct conventions.

ANSWER FORMAT (strict, keep it short; do all deliberation privately before answering):
Line 1: "VERDICT: PASS" or "VERDICT: FIX".
If FIX: at most 6 one-line problem bullets (cite the instruction wording), then the SEARCH/REPLACE edits:

<<<<<<< SEARCH
exact lines from the current program
=======
replacement lines
>>>>>>> REPLACE

Keep each SEARCH block short: only the changed lines plus at most 2 lines of context.
No derivations, no alternatives, no text after the last edit."""


INDEPENDENT_NOTE = """=== INDEPENDENT IMPLEMENTATION ===
Another engineer already implemented this task; your program will be compared with theirs number by number.
Write your own implementation FROM SCRATCH, re-deriving every formula, convention, window, alignment, unit,
sort order and random-number procedure directly from the instruction text (read it twice). Where the
instruction is explicit, follow it literally."""

ADJUDICATE_RULES = """Two independent implementations (A and B) of the instruction produced different outputs (listed above).
For each disagreement decide, from the instruction text and definite mathematical/financial facts, which
program is correct. Typical culprits: NaN warm-up handling, off-by-one windows, signal/execution timing,
ddof, annualisation, compounding/day count, quote direction, sort order, random-number procedure.

ANSWER FORMAT (strict; deliberate privately first):
Line 1: "VERDICT: A" or "VERDICT: B" (the program that is right on more of the disagreements).
Then at most 6 one-line bullets naming, per disagreement, which side is right and why (cite the instruction).
Then, ONLY if the chosen program is still wrong somewhere (e.g. the other side was right on some item),
SEARCH/REPLACE edits against the CHOSEN program:

<<<<<<< SEARCH
exact lines from the chosen program
=======
replacement lines
>>>>>>> REPLACE

Keep each SEARCH block short: only the changed lines plus at most 2 lines of context."""


SYNTAX_FIX = """The program below fails to compile with this error:
{error}

Lines {a}-{b} of the program (line-numbered):
{window}

Fix the syntax error(s) in this region with SEARCH/REPLACE edits (exact lines from the program, without the
line-number prefixes). Change only what is needed to make the code valid; keep the intended logic. Output only
the edits, no explanation:

<<<<<<< SEARCH
exact lines
=======
fixed lines
>>>>>>> REPLACE"""


def task_block(
    instruction: str,
    card: str,
    inventory: str,
    profile: str,
    contract: str,
    checks: str,
    examples: str,
    *,
    compact: bool = False,
) -> str:
    parts = [
        "=== TASK INSTRUCTION ===\n" + instruction.strip(),
        "=== CARD ===\n" + card,
        "=== INPUT FILES (real paths; aliases are how the instruction may refer to them) ===\n" + inventory,
    ]
    if not compact:
        parts.append("=== DATA PROFILE (computed by the agent) ===\n" + profile)
    else:
        parts.append("=== DATA PROFILE (abridged) ===\n" + profile[:6000])
    parts.append("=== DELIVERABLES DETECTED AUTOMATICALLY FROM THE INSTRUCTION ===\n" + contract)
    if checks:
        parts.append(
            "=== GRADER TEST SOURCE (available for this unit; the outputs must pass these tests) ===\n"
            + checks
        )
    if examples and not compact:
        parts.append(
            "=== WORKED EXAMPLES FROM OTHER TASKS (for technique/conventions only; this task differs) ===\n"
            + examples
        )
    return "\n\n".join(parts)
