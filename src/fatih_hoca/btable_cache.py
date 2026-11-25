"""Module-level B-table cache (learned per-step token estimates).

Read by selection (ranking / requirements_builder) — hence it lives here, in
fatih_hoca. Written by general_beckman's hourly ``btable_rollup`` cron, which
is the downward direction (orchestration → selection).
"""
from __future__ import annotations

_BTABLE: dict[tuple[str, str, str], dict] = {}


def get_btable() -> dict:
    return _BTABLE


def set_btable(rows: dict) -> None:
    global _BTABLE
    _BTABLE = rows
