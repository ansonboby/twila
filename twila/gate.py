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
        "instructions": "How much sustained reasoning does a correct answer need?",
        "criteria": ["a one-liner", "a few clear steps", "contest-class, long derivation"],
    },
    "needs_thinking": {
        "type": "noul",
        "instructions": "Would a careful expert need to work through multiple steps of derivation or computation on paper to get this right (contest math proofs, multi-step algorithm debugging)? Or is recall/scomputed quickly enough that one line suffices?",
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

        if self._checkpoint:  # pin one checkpoint as the default route
            self.router = Router(default=_alias_checkpoint(self._checkpoint), preload=True)
        else:
            self.router = Router(preload=True)

    def _predict(self, state: dict, questions: dict) -> dict:
        """One batched decision call: all questions over one state share a read."""
        out = self.router.predict(state, questions)
        out = out if isinstance(out, dict) else out.to_dict()
        return out.get("answers", out)

    @staticmethod
    def _value(block, default=0.0) -> float:
        if not isinstance(block, dict):
            return default
        for k in ("noul", "score", "value", "label"):
            if k in block:
                v = block[k]
                return float(v) if isinstance(v, (int, float)) else _bool_val(v)
        return default

    # -- pre-check: what did the user ask, and how much TwIL do we owe it? ----
    def gate(self, text: str) -> Decision:
        t0 = time.perf_counter()
        raw = self._predict({"text": text}, PRE_QUESTIONS)

        def block(k):
            v = raw.get(k, {})
            return v if isinstance(v, dict) else {"value": v}

        dom = block("domain")
        probs = dom.get("probabilities") or {}
        self.last_decision = Decision(
            domain=str(dom.get("choice", dom.get("value", dom.get("label", "general")))),
            difficulty=_score_to_level(block("difficulty").get("score",
                                       block("difficulty").get("value", 1))),
            needs_thinking=self._value(block("needs_thinking"),
                                       default=_bool_val(block("needs_thinking"))) > 0.5,
            policy_violation=self._value(block("policy_violation"),
                                         default=_bool_val(block("policy_violation"))) > 0.85,
            raw=raw,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )
        # Laya unconfident on this input: fail safe toward more reasoning.
        if probs and max(probs.values()) < 0.3 and self.last_decision.difficulty < 2:
            self.last_decision.difficulty = 2
        return self.last_decision

    # -- post-check: is the finished answer trustworthy -----------------------
    def verify(self, question: str, answer: str) -> dict:
        if "\\boxed" in answer:
            answer = answer.rsplit("\\boxed", 1)[-1].rstrip("$]}").strip()
        raw = self._predict({"question": question, "answer": answer[:2000]},
                            VERIFY_QUESTIONS)
        return {"answered": self._val(raw.get("answered", {})),
                "confidence": self._val(raw.get("high_confidence", {})),
                "raw": raw}

    @staticmethod
    def _val(block) -> float:
        if not isinstance(block, dict):
            return 0.0
        for k in ("noul", "value", "label"):
            if k in block:
                v = block[k]
                return float(v) if isinstance(v, (int, float)) else _bool_val(v)
        return 0.0

    # -- plan: decision -> generation settings --------------------------------
    def plan(self, text: str, variant: str = "balanced") -> Plan:
        d = self.gate(text)
        if d.policy_violation:
            return Plan(d, False, 0, "", False, refused=True,
                        reason="laya policy gate (p>0.85)")
        # needs_thinking proved zero-shot-unreliable on base checkpoints; it is
        # logged, not trusted. math/code at score>=1 always think, like the
        # card's own protocol (thinking on for graded tasks).
        if d.domain in ("math", "code") and d.difficulty >= 1:
            level = max(d.difficulty, 1)
        elif d.difficulty >= 2:
            level = 2
        elif d.difficulty <= 0:
            level = 0
        else:
            level = 1
        pol = LEVELS[level]
        return Plan(d, pol["thinking"], pol["budget"],
                    DOMAIN_PROMPTS.get(d.domain, DOMAIN_PROMPTS["general"]),
                    pol["retry"])


def _alias_checkpoint(name: str) -> str:
    """Map any user-facing checkpoint name onto a Router-recognized alias."""
    if name.startswith("convaiinnovations/laya"):
        return name[len("convaiinnovations/"):]  # e.g. 'laya-typed-decisions'
    return name


def _bool_val(v) -> float:
    """Local router wraps booleans as plain bools; HTTP path sends probabilities."""
    return 1.0 if v in (True, "true", "True") else (0.0 if v in (False, "false", "False") else 0.5)


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
        self.last_decision = None  # no heavy local checkpoint

    def _predict(self, state: dict, questions: dict) -> dict:
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
        return self._normalize(r.json().get("answers") or {}, questions)

    @staticmethod
    def _normalize(answers: dict, questions: dict) -> dict:
        """Turn /v1/systemone answers into the internal {value, probabilities}
        shape the base class already parses."""
        out = {}
        for qid, spec in questions.items():
            ans = answers.get(qid, {})
            typ = spec.get("type")
            if typ == "choice":
                val = ans.get("choice")
            elif typ == "score":
                val = float(ans.get("score", 1))  # level index, may be fractional
            else:
                val = float(ans.get("noul", 0.0))
            out[qid] = {"value": val,
                        "probabilities": ans.get("probabilities", {}),
                        "confidence": ans.get("confidence"),
                        "answer_confidence": ans.get("answer_confidence")}
        return out


def make_gate() -> LayaGate:
    """TWILA_GATE=local (default) | http (any /v1/systemone server, e.g. Jev)."""
    import os

    if os.environ.get("TWILA_GATE") == "http":
        return HTTPGate()
    return LayaGate(checkpoint=os.environ.get("TWILA_GATE_CHECKPOINT") or None)
