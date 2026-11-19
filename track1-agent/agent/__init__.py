# SPDX-License-Identifier: Apache-2.0
"""Agenthon 2026 Track 1 quant-coding agent.

Pipeline: ingest -> profile -> contract -> plan (House) -> code (House) -> execute ->
validate -> repair/review (House) -> finalize. The House model writes every solution at run
time; this package only supplies orchestration, inspection, verification and output hygiene.
"""

__version__ = "0.1.0"
