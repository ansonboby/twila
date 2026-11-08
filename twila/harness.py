"""The loop: Laya decides, TwIL reasons, Laya grades. History and receipts kept."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .gate import LayaGate, Plan
from .twil import Turn, TwilClient

RECEIPTS_DIR = Path.home() / ".twila" / "receipts"


@dataclass
class Receipt:
    id: str
    ts: float
    question: str
    answer: str
    plan: dict          # what Laya decided and why
    verify: dict | None # post-check scores, when present
    retried: bool = False
    generation: dict = field(default_factory=dict)

    def save(self) -> Path:
        RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RECEIPTS_DIR / f"{self.id}.json"
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path


class Harness:
    """One instance orchestrates gate->generate->verify and remembers history."""

    def __init__(self, base_url: str | None = None, checkpoint: str | None = None,
                 served_variant: str = "balanced", receipts: bool = True):
        from .gate import make_gate

        self.gate = make_gate()
        self.twil = TwilClient(base_url or "http://127.0.0.1:2718/v1")
        self.variant = served_variant
        self.history: list[dict] = []
        self.save_receipts = receipts

    def ask(self, message: str) -> tuple[str, Receipt]:
        """Full pipeline for one message. Returns (delivery text, receipt)."""
        plan = self.gate.plan(message, variant=self.variant)
        if plan.refused:
            answer = "Refused by the local policy gate."
            return answer, self._receipt(message, answer, plan, None, False, {})

        turn = self.twil.ask(
            self.history + [{"role": "user", "content": message}],
            budget=plan.budget, thinking=plan.thinking, system=plan.system_prompt,
        )
        if turn.truncated and plan.can_retry:
            plan = plan.escalated(plan)
            turn2 = self.twil.ask(
                [{"role": "user", "content": message + " (You were cut off before producing a final answer. Work efficiently, verify, answer.)"}],
                budget=18000, thinking=True, system=plan.system_prompt,
            )
            turn = turn2
            plan.can_retry = False
            retried = True
            verify = self.gate.verify(message, turn.content)
            low_conf = verify["confidence"] < 0.33 or verify["answered"] < 0.35
            if low_conf:
                plan = plan.escalated(plan)
                turn3 = self.twil.ask(
                    [{"role": "user", "content": message}],
                    budget=24576, thinking=True, system=plan.system_prompt,
                )
                if turn3.tokens_out > 0 and not turn3.truncated:
                    turn = turn3
                retried = True
                verify = self.gate.verify(message, turn.content)
        else:
            retried = False
            verify = self.gate.verify(message, turn.content) if plan.budget >= 4096 else None

        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": turn.content})
        receipt = self._receipt(message, turn.reasoning, plan, verify, retried,
                                {"tokens_out": turn.tokens_out, "latency": turn.latency,
                                 "variant": self.variant})
        return turn.content, receipt

    def _receipt(self, question, answer, plan, verify, retried, gen):
        d = plan.decision
        r = Receipt(
            id=uuid.uuid4().hex[:12], ts=time.time(),
            question=question, answer=answer,
            plan={"domain": d.domain, "difficulty": d.difficulty,
                  "needs_thinking": d.needs_thinking, "policy_violation": d.policy_violation,
                  "budget": plan.budget, "thinking": plan.thinking, "gate_latency_ms": d.latency_ms},
            verify=verify, retried=retried, generation=gen,
        )
        if self.save_receipts:
            r.save()
        return r


# History grows forever, and with Granite's 131072 context you mostly don't care,
# but a chatty session feeding context into the server penalizes first-token
# latency enough to be worth pruning when we exceed this.
MAX_HISTORY_TURNS = 20
