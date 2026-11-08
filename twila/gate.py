"""System 1: Laya typed decisions -> a Plan that decides how much TwIL you buy.

Every decision here is one forward pass (~50ms CPU). No text is generated; the
outputs feed code branches and prompt fragments directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

# Budgets are CPU-tuned: 16k tokens of reasoning is already ~30+ min on this
# machine. Raise on real hardware; the policy stays shape-equal.

LEVELS = {0: dict(thinking=False, budget=2048, retry=False),   # one-liners
          1: dict(thinking=True, budget=8192, retry=False),    # standard steps
          2: dict(thinking=True, budget=16384, retry=True)}    # contest-class

DOMAIN_PROMPTS = {
    "math": "Work through the problem step by step, verify each step, then give a final numeric or symbolic answer.",
    "code": "Provide complete, runnable code with a brief explanation. Handle edge cases.",
    "instruction": "Follow every stated constraint exactly; compliance beats eloquence.",
    "general": "Be concise and direct.",
}

PRE_QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which kind of request is this?",
        "criteria": {
            "math": "arithmetic, algebra, calculus, probability, logic puzzles with numbers",
            "code": "writing, explaining or debugging source code",
            "instruction": "following explicit constraints, formatting or summarizing rules",
            "general": "anything else: facts, chat, opinions",
        },
    },
    "difficulty": {
        "type": "score",
        "instructions": "How much sustained reasoning does a correct answer need? 0 = one-liner, 1 = a few clear steps, 2 = contest-class, long derivation.",
        "criteria": {0: "one-liner", 1: "a few steps", 2: "contest-class"},
    },
    "needs_thinking": {
        "type": "noul",
        "instructions": "Does correctness require extended step-by-step reasoning, rather than recall?",
    },
    "policy_violation": {
        "type": "noul",
        "instructions": "Is the request asking for clearly harmful, dangerous or malicious content (weapons, malware)?",
    },
}

VERIFY_QUESTIONS = {
    "answered": {"type": "noul",
                 "instructions": "Does this answer actually try to answer the question asked?"},
    "high_confidence": {"type": "noul",
                        "instructions": "Is the answer complete and confident (not truncated, hedged, or self-contradicting)?"},
}


@dataclass
class Decision:
    domain: str
    difficulty: int     # 0..2 after normalization
    needs_thinking: bool
    policy_violation: bool
    raw: dict = field(default_factory=dict)
    latency_ms: int = 0


@dataclass
class Plan:
    decision: Decision
    thinking: bool
    budget: int
    system_prompt: str
    can_retry: bool
    refused: bool = False
    reason: str = ""

    def escalated(self) -> "Plan":
        return Plan(self.decision, True, max(self.budget, 8192) * 2,
                    self.system_prompt, False, reason="escalated")


def _score_to_level(raw) -> int:
    """Laya score labels arrive as e.g. '1.4/2' or a criterion string."""
    s = str(raw)
    try:
        return min(2, max(0, round(float(s.split("/")[0]) * 1.0)))
    except ValueError:
        low = s.lower()
        for key, hint in (("one", 0), ("two", 2)):
            if hint and low.startswith(str(hint)):
                return hint
        return 1


class LayaGate:
    def __init__(self, checkpoint: str | None = None, preload: bool = True):
        self._checkpoint = checkpoint
        self.last_decision: Decision | None = None
        if preload:
            self._load()

    def _load(self):
        from laya import Router

        self.router = (Router(strict=self._checkpoint) if self._checkpoint
                       else Router(preload=True))

    def _predict(self, state: dict, questions: dict) -> dict:
        """One batched decision call: all questions over one state share a read."""
        out = self.router.predict(state, questions)
        return out if isinstance(out, dict) else out.to_dict()


def make_gate() -> "LayaGate":
    """TWILA_GATE=local (default) | http (any /v1/systemone server, e.g. Jev)."""
    import os

    if os.environ.get("TWILA_GATE") == "http":
        return HTTPGate()
    return LayaGate(checkpoint=os.environ.get("TWILA_GATE_CHECKPOINT") or None)


class HTTPGate(LayaGate):
    """Same gate() and plan() logic; the decision comes from any /v1/systemone
    server instead of the local checkpoint.

    TWILA_GATE_BASE_URL  e.g. https://api.laya-ai.com/v1/systemone,
                          your Jev base, or http://localhost:8000/v1/systemone
    TWILA_GATE_API_KEY   bearer token
    TWILA_GATE_MODEL     optional model id (english / typed-decisions / Jev ids)
    """

    def __init__(self):
        import os

        self.base_url = os.environ["TWILA_GATE_BASE_URL"]
        self.key = os.environ["TWILA_GATE_API_KEY"]
        self.model = os.environ.get("TWILA_GATE_MODEL", "")
        self.last_decision = None  # set by gate() as usual; no heavy import

    def _predict(self, state: dict, questions: dict) -> dict:
        import os

        import requests

        body = {"state": " ".join(str(v) for v in state.values()), "questions": questions}
        if self.model:
            body["model"] = self.model
        r = requests.post(
            self.base_url, timeout=30,
            headers={"Authorization": f"Bearer {self.key}"},
            json=body,
        )
        r.raise_for_status()
        answers = (r.json().get("answers") or {})
        return self._normalize(answers, questions)

    @staticmethod
    def _normalize(answers: dict, questions: dict) -> dict:
        """Turn /v1/systemone answers into the internal {value, probabilities} shape
        that gate()/verify() already parse."""
        out = {}
        for qid, spec in questions.items():
            ans = answers.get(qid, {})
            typ = spec.get("type")
            if typ == "choice":
                val = ans.get("choice")
            elif typ == "score":
                val = float(ans.get("score", 1))  # level index, may be fractional
            else:
                val = float(ans.get("noul", 0.0)) > 0.5
            out[qid] = {"value": val,
                        "probabilities": ans.get("probabilities", {}),
                        "confidence": ans.get("confidence"),
                        "answer_confidence": ans.get("answer_confidence")}
        return out

    @staticmethod
    def _value(block, default=0.0) -> float:
        if not isinstance(block, dict):
            return default
        v = block.get("value", block.get("label", default))
        return float(v) if isinstance(v, (int, float)) else default

    # -- pre-check: what did the user ask for, and how much Execution do we owe --
    def gate(self, text: str) -> Decision:
        if not hasattr(self, "router"):
            self._load()
        t0 = time.perf_counter()
        raw = self._predict({"text": text}, PRE_QUESTIONS)

        def block(k):
            v = raw.get(k, {})
            return v if isinstance(v, dict) else {"value": v}

        dom = block("domain")
        self.last_decision = Decision(
            domain=str(dom.get("value", dom.get("label", "general"))),
            difficulty=_score_to_level(block("difficulty").get("value", 1)),
            needs_thinking=self._value(block("needs_thinking")) > 0.5,
            policy_violation=self._value(block("policy_violation")) > 0.85,
            raw=raw,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        # Laya degraded on odd input (no strong signal): fail safe to hard.
        dom_conf = max(block("domain").get("probabilities", {}).values() or [0]) if isinstance(block("domain"), dict) else 0
        if dom_conf < 0.3 and self.last_decision.difficulty < 2:
            self.last_decision.difficulty = 2
        return self.last_decision

    # -- post-check: is the finished answer trustworthy -----------------------
    def verify(self, question: str, answer: str) -> dict:
        if not hasattr(self, "router"):
            self._load()
        if "\\boxed" in answer:
            answer = answer.rsplit("\\boxed", 1)[-1].rstrip("$]}").strip()
        raw = self._predict({"question": question, "answer": answer[:2000]},
                            VERIFY_QUESTIONS)
        return {"answered": self._value(raw.get("answered", {})),
                "confidence": self._value(raw.get("high_confidence", {})),
                "raw": raw}

    # -- plan: decision -> generation settings --------------------------------
    def plan(self, text: str, variant: str = "balanced") -> Plan:
        d = self.gate(text)
        if d.policy_violation:
            return Plan(d, False, 0, "", False, refused=True,
                        reason="laya policy gate (p>0.85)")
        # level: what the decision says, but never fully coast on easy scores
        # below level 1 unless laya is clear that no reasoning is needed.
        if d.needs_thinking or d.difficulty >= 2:
            level = d.difficulty
        else:
            level = min(d.difficulty, 0)
        pol = LEVELS[level]
        return Plan(d, pol["thinking"], pol["budget"],
                    DOMAIN_PROMPTS.get(d.domain, DOMAIN_PROMPTS["general"]),
                    pol["retry"])
