"""MAROS-SEI Stage 15 single entry point.

Normal use:
    python run_stage15.py

Quick smoke run:
    python run_stage15.py --samples-per-class 4

All CLI arguments are forwarded to the Stage-15 experiment runner.
"""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "scripts" / "experiments" / "run_stage15_llm_multiagent.py"

if not RUNNER.exists():
    raise FileNotFoundError(f"Stage-15 runner not found: {RUNNER}")

runpy.run_path(str(RUNNER), run_name="__main__")
