"""Speed demotion must be recoverable.

Regression: a single low tok/s reading demoted a local model, the flag was
persisted to data/model_speeds.json, restored at every startup and never
cleared — and since the selector skips demoted models the model was never
re-measured either. Healthy models stayed disabled for months.
"""
import json
import time

import pytest

from fatih_hoca.registry import ModelInfo, ModelRegistry
from fatih_hoca.selector import Selector


@pytest.fixture(autouse=True)
def _isolated_speed_cache(tmp_path, monkeypatch):
    path = tmp_path / "model_speeds.json"
    monkeypatch.setattr(ModelRegistry, "_SPEED_CACHE_PATH", path)
    return path


def _local(name="m", tps=0.0, **kw):
    return ModelInfo(
        name=name, location="local", provider="llama_cpp",
        litellm_name=f"openai/{name}", capabilities={},
        context_length=8192, max_tokens=4096, tokens_per_second=tps, **kw,
    )


def _registry(*models):
    reg = ModelRegistry()
    reg.models = {m.name: m for m in models}
    return reg


def test_slow_reading_demotes_with_expiry():
    info = _local()
    reg = _registry(info)
    before = time.time()
    reg.update_measured_speed("m", 1.0)
    assert info.demoted is True
    assert info.demoted_until >= before + ModelRegistry._SPEED_DEMOTION_TTL - 1


def test_ema_recovery_clears_speed_demotion():
    info = _local()
    reg = _registry(info)
    reg.update_measured_speed("m", 1.5)
    assert info.demoted
    # Real decode readings come back healthy; the EMA climbs over the floor.
    for _ in range(10):
        reg.update_measured_speed("m", 6.0)
    assert info.tokens_per_second >= ModelRegistry._MIN_USABLE_TPS
    assert info.demoted is False
    assert info.demoted_until == 0.0


def test_selector_ignores_expired_speed_demotion():
    info = _local(demoted=True, demoted_until=time.time() - 1)
    assert Selector._is_demoted(info) is False
    live = _local(demoted=True, demoted_until=time.time() + 3600)
    assert Selector._is_demoted(live) is True
    # No expiry recorded = not a speed demotion: stays demoted.
    assert Selector._is_demoted(_local(demoted=True)) is True


def test_expired_demotion_rearms_on_new_slow_reading():
    info = _local(tps=1.0, demoted=True, demoted_until=time.time() - 1)
    reg = _registry(info)
    reg.update_measured_speed("m", 1.0)
    assert info.demoted is True
    assert info.demoted_until > time.time()


def test_active_demotion_survives_restart(_isolated_speed_cache):
    reg = _registry(_local())
    reg.update_measured_speed("m", 1.0)
    reg._save_speed_cache(force=True)

    fresh_info = _local()
    _registry(fresh_info)._load_speed_cache()
    assert fresh_info.demoted is True
    assert fresh_info.demoted_until > time.time()


def test_expired_demotion_not_restored(_isolated_speed_cache):
    _isolated_speed_cache.write_text(json.dumps({
        "m": {"tps": 1.2, "demoted": True, "demoted_until": time.time() - 10},
    }), encoding="utf-8")
    info = _local()
    _registry(info)._load_speed_cache()
    assert info.demoted is False


def test_legacy_permanent_demotion_not_restored(_isolated_speed_cache):
    """Pre-fix entries (no expiry) came from the wall-clock measure: drop
    both the flag and the bogus tps so the model is re-measured fresh."""
    _isolated_speed_cache.write_text(json.dumps({
        "m": {"tps": 1.3, "demoted": True, "updated": "2026-04-15"},
        "ok": {"tps": 30.0, "demoted": False},
    }), encoding="utf-8")
    info, ok = _local("m"), _local("ok")
    _registry(info, ok)._load_speed_cache()
    assert info.demoted is False
    assert info.tokens_per_second == 0.0
    assert ok.tokens_per_second == 30.0
