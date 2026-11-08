import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gateway"))
sys.path.insert(0, str(ROOT / "tools"))

import pytest  # noqa: E402

from app.auth import check_key, extract_bearer, hash_key, key_id, new_key  # noqa: E402
from app.metrics import Metrics  # noqa: E402
from app.ratelimit import RateLimiter  # noqa: E402
import vram_estimator as ve  # noqa: E402


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


# --- auth -------------------------------------------------------------------

def test_new_key_is_random_and_prefixed():
    a, b = new_key(), new_key()
    assert a != b and a.startswith("llm_") and len(a) > 40


def test_check_key_accepts_only_known_hashes():
    key = new_key()
    allowed = [hash_key("other"), hash_key(key)]
    assert check_key(key, allowed) == hash_key(key)
    assert check_key("wrong", allowed) is None
    assert check_key(None, allowed) is None
    assert check_key(key, []) is None


@pytest.mark.parametrize("header,expected", [
    ("Bearer abc", "abc"),
    ("bearer  abc ", "abc"),
    ("Basic abc", None),
    ("Bearer", None),
    (None, None),
])
def test_extract_bearer(header, expected):
    assert extract_bearer(header) == expected


def test_key_id_is_short_and_not_the_key():
    h = hash_key("secret")
    assert key_id(h) == h[:8] and "secret" not in key_id(h)


# --- rate limit -------------------------------------------------------------

def test_rate_limiter_blocks_then_refills():
    clock = FakeClock()
    rl = RateLimiter(per_minute=2, clock=clock)
    assert rl.allow("k") and rl.allow("k")
    assert not rl.allow("k")
    assert rl.retry_after("k") >= 1
    clock.t += 30  # 2 per minute -> 1 token every 30 s
    assert rl.allow("k")


def test_rate_limiter_is_per_key():
    rl = RateLimiter(per_minute=1, clock=FakeClock())
    assert rl.allow("a") and rl.allow("b")
    assert not rl.allow("a")


def test_rate_limiter_rejects_bad_config():
    with pytest.raises(ValueError):
        RateLimiter(per_minute=0)


# --- metrics ----------------------------------------------------------------

def test_metrics_render_prometheus_text():
    m = Metrics()
    m.record("abcd1234", "llama3.1:8b", 200, 1.5, prompt_tokens=10, completion_tokens=20)
    text = m.render()
    assert 'gateway_requests_total{key="abcd1234",model="llama3.1:8b",status="200"} 1' in text
    assert 'gateway_tokens_total{model="llama3.1:8b",kind="completion"} 20' in text


# --- VRAM estimator -----------------------------------------------------------

def test_weights_8b_at_4bit_is_about_4gb():
    assert 3.5 < ve.weights_gb(8.0, 4) < 4.0


def test_bigger_context_needs_more_memory():
    spec = ve.CATALOG["llama3.1:8b"]
    assert ve.estimate_gb(spec, 4, 32768) > ve.estimate_gb(spec, 4, 4096)


def test_fitting_models_sorted_largest_first_and_within_budget():
    fits = ve.fitting_models(12, bits=4, context=8192)
    names = [n for n, _ in fits]
    assert "llama3.1:8b" in names and "llama3.3:70b" not in names
    assert all(gb <= 12 for _, gb in fits)
    sizes = [ve.CATALOG[n].params_b for n in names]
    assert sizes == sorted(sizes, reverse=True)


def test_cli_runs(capsys):
    assert ve.main(["--vram", "24"]) == 0
    assert "Models that fit" in capsys.readouterr().out
