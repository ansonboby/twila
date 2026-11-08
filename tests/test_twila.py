"""Fast tests for gate + scoring that don't need llama-server.

Laya checkpoint downloads on first run (~1.7GB) — cache is per-user, after that
these run in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from twila.eval import extract_answer, same_answer
from twila.twil import split_thinking


# --- pure-function tests: run even without laya cached ----------------------

def test_http_normalize_and_plan(monkeypatch):
    """The Jev/hospital path: no local checkpoint, no network — stub _predict."""
    from twila.gate import HTTPGate

    g = HTTPGate.__new__(HTTPGate)  # skip __init__ (needs env vars)
    g.last_decision = None
    fake = {
        "domain": {"choice": "math", "probabilities": {"math": 0.91, "code": 0.05}},
        "difficulty": {"score": 2.4, "probabilities": {"2": 0.7}},
        "needs_thinking": {"noul": 0.96},
        "policy_violation": {"noul": 0.02},
    }
    monkeypatch.setattr(HTTPGate, "_predict", lambda self, s, q: HTTPGate._normalize(fake, q))
    plan = g.plan("integrate $\\int x^2 dx$ and prove convergence details")
    assert plan.decision.domain == "math"
    assert plan.decision.difficulty == 2 and plan.thinking and plan.budget == 16384
    assert not plan.refused

@pytest.mark.parametrize("raw,delivery", [
    ("thinke/visual>\nstep 1...\n>\n\nThe answer is 4.", "The answer is 4."),
    ("plain answer, no thinking block", "plain answer, no thinking block"),
])
def test_split_thinking(raw, delivery):
    reasoning, content = split_thinking(raw)
    assert delivery in content
    if raw.startswith("thinke/visual>"):
        assert "step 1" in reasoning


def test_extract_boxed():
    assert extract_answer("we get $\\boxed{267}$ total") == "267"
    assert extract_answer("...final $\\boxed{\\frac{288}{17}}$") == "(288/17)"


@pytest.mark.parametrize("exp,got,ok", [
    ("267", "The count is 267.", True),
    ("0.5", "So we get 0.5", True),
    ("1/2", "answer 0.5000", True),
    ("267", "answer 268", False),
    ("\\frac{288}{17}", "x = 288/17 ??", True),
])
def test_same_answer(exp, got, ok):
    assert same_answer(exp, got) == ok


# --- laya live tests: real decisions, real checkpoint ------------------------

LAYA_CACHED = (Path.home() / ".cache" / "huggingface" / "hub").glob("models--convaiinnovations-laya*")
pytestmark_laya = pytest.mark.skipif(not any(LAYA_CACHED), reason="laya checkpoints not cached yet")


@pytest.mark.usefixtures()
class TestGate:
    @pytest.fixture()
    def gate(self):
        from twila.gate import LayaGate
        return LayaGate()

    def test_trivial_gets_low_level(self, gate):
        d = gate.gate("What's 2 + 2?")
        assert d.difficulty in (0, 1)

    def test_contest_gets_high_level(self, gate):
        d = gate.gate("Let $S$ be the set of positive integers $n$ such that $n^4 - 7n - 29$ is prime. Find the sum of all $n$ in $S$.")
        assert d.difficulty >= 1

    def test_plan_matches_budget(self, gate):
        plan = gate.plan("What's 2 + 2?")
        assert 0 <= plan.budget <= 16384 and plan.system_prompt

    def test_verdict_shape(self, gate):
        v = gate.verify("What is 2+3?", "The answer is 5.")
        assert set(v) == {"answered", "confidence", "raw"}
