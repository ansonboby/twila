"""System 2: the only place that talks to the TwIL-8B llama-server.

Server is llama-server serving WebAI-Official/TwIL-8B's Q4_K_M GGUF behind an
OpenAI-compatible endpoint on port 2718 — our port, so we never collide with
anything on 8080/11434 .
"""

from __future__ import annotations

import time
from dataclasses import dataclass


# GGUFs are CPU-only here, so twila only names requests downstream.
# One forward step at a time: the deliberate split keeps eval receipts honest.
DEFAULT_SERVER = "http://127.0.0.1:2718/v1"
DEFAULT_TOP_P, DEFAULT_TEMP = 0.95, 1.0  # the shipped generation_config


@dataclass
class Turn:
    content: str       # delivery: what the model actually ANSWERS
    reasoning: str     # thinke/visual> trace, if any (dim-gray in the UI)
    tokens_out: int
    latency: float
    truncated: bool    # hit the budget before finishing


def split_thinking(text: str) -> tuple[str, str]:
    """Split raw completion into (reasoning, content) on the TwIL tags."""
    a, b = "thinke/visual>", ">"
    if b in text:
        _r, content = text.split(b, 1)
        return _r.removeprefix(a).lstrip(), content.lstrip()
    return "", text.strip()


class TwilClient:
    def __init__(self, base_url: str = DEFAULT_SERVER, model: str = "TwIL-8B"):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key="none", max_retries=0)
        self.model = model

    def ask(
        self,
        messages: list[dict],
        *,
        budget: int,
        thinking: bool = True,
        system: str | None = None,
        temperature: float = DEFAULT_TEMP,
        top_p: float = DEFAULT_TOP_P,
    ) -> Turn:
        """One completion. Fails loudly if server is down; binary is ours."""
        prompt = [m for m in messages if m["role"] != "system"]
        if system:
            prompt = [{"role": "system", "content": system}, *prompt]
        t0 = time.perf_counter()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            max_tokens=budget,
            temperature=temperature,
            top_p=top_p,
        )
        dt = time.perf_counter() - t0
        reasoning, content = split_thinking(completion.choices[0].message.content or "")
        tokens_out = completion.usage.completion_tokens if completion.usage else 0
        return Turn(
            content=content.strip(), reasoning=reasoning,
            tokens_out=tokens_out, latency=dt,
            truncated=tokens_out >= budget,
        )

    def available(self) -> bool:
        try:
            return bool(self.client.models.list().data)
        except Exception:
            return False
