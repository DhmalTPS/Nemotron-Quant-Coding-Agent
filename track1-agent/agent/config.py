# SPDX-License-Identifier: Apache-2.0
"""Runtime configuration. Every knob can be overridden by an environment variable so the
same image can be benchmarked under different budgets without a rebuild."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- House budget (competition: 25 admitted requests/unit, 4000 output tokens/request) ---
    hard_request_cap: int = 25
    request_cap: int = field(default_factory=lambda: min(_int("AGENT_REQUEST_CAP", 16), 25))
    max_output_tokens: int = 4000
    # --- wall clock ---------------------------------------------------------------------------
    # Development roster: 12 h for ~86 units (about 8 min each). The card ceiling is an upper
    # bound only; the shared stage clock is what binds, so cap our own spend per unit.
    # 12 h stage clock / 86 units. Evidence: v7 (400 s) 16/86 < v6 (450 s) 18/86 on the platform, and the full
    # local run at 450 s used mean 313 s/unit (7.5 h projected) while p90 hit the cap with 34 units cut off
    # mid-repair. 560 s adds <= ~1.3 h worst case (~9 h total) and gives struggling units more repair time.
    max_unit_seconds: float = field(default_factory=lambda: _float("AGENT_MAX_SECONDS", 560.0))
    card_margin_seconds: float = 90.0
    exec_timeout_cap: float = field(default_factory=lambda: _float("AGENT_EXEC_TIMEOUT", 300.0))
    http_timeout: float = field(default_factory=lambda: _float("AGENT_HTTP_TIMEOUT", 240.0))
    # --- sampling (pinned for reproducibility) -------------------------------------------------
    temperature_think: float = field(default_factory=lambda: _float("AGENT_TEMP_THINK", 0.6))
    temperature_code: float = field(default_factory=lambda: _float("AGENT_TEMP_CODE", 0.2))
    top_p: float = 0.95
    # --- behaviour ------------------------------------------------------------------------------
    # Nemotron reasoning shares the 4000-token cap; NIM honours chat_template_kwargs.reasoning_budget
    reasoning_budget_plan: int = field(default_factory=lambda: _int("AGENT_RB_PLAN", 1500))
    reasoning_budget_fix: int = field(default_factory=lambda: _int("AGENT_RB_FIX", 1500))
    stream: bool = field(default_factory=lambda: _bool("AGENT_STREAM", True))  # SSE: robust to idle-timeout proxies
    use_checks_dir: bool = field(default_factory=lambda: _bool("AGENT_USE_CHECKS", True))
    max_repairs: int = field(default_factory=lambda: _int("AGENT_MAX_REPAIRS", 6))
    dual: bool = field(default_factory=lambda: _bool("AGENT_DUAL", True))  # differential self-consistency
    review_passes: int = field(default_factory=lambda: _int("AGENT_REVIEW_PASSES", 1))
    use_casebook: bool = field(default_factory=lambda: _bool("AGENT_USE_CASEBOOK", True))
    work_root: str = field(default_factory=lambda: os.environ.get("AGENT_WORK", "/tmp/agent"))
    debug_dir: str = field(default_factory=lambda: os.environ.get("AGENT_DEBUG_DIR", ""))
    seed: int = field(default_factory=lambda: _int("QFBENCH_SEED", 0))
