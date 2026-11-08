"""The proof: Laya-routing vs one-size-fits-all on the same problem set.

uv run twila eval --limit 10 --mode both

raw    = control: every problem gets the same generous budget + thinking on.
routed = the harness: laya picks effort, budget, and retries per problem.
Same problems, same scorer — the delta is the router's value.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from .harness import Harness

EVALS = Path(__file__).resolve().parent.parent / "evals" / "math500_mini.jsonl"

# --- scoring ----------------------------------------------------------------

_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract_answer(text: str) -> str:
    text = (text or "")
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1/\2)", text)  # \frac{a}{b} -> (a/b)
    m = _BOXED.findall(text)
    if m:
        return m[-1].strip().rstrip(".")
    text = (text or "").strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return ""
    m = re.search(r"-?\d+(?:\.\d+)?(?:/\d+)?", lines[-1].replace(",", ""))
    return m.group(0).lstrip("$") if m else ""


def same_answer(expected: str, got: str) -> bool:
    e, g = extract_answer(expected), extract_answer(got) or got.strip()
    norm = lambda s: s.strip().rstrip(".").replace(" ", "")
    if norm(e) == norm(g):
        return True
    try:  # numeric equality: 0.5 == 1/2
        return abs(float(eval(norm(e))) - float(eval(norm(g)))) < 1e-9
    except Exception:
        return False


# --- the runs ----------------------------------------------------------------

@dataclass
class Result:
    id: str
    level: int
    expected: str
    got: str
    correct: bool
    tokens: int     # exact from receipt/usage
    seconds: float
    gate_level: int # what laya thought


def run_arm(harness, problems, arm: str):
    rows, wall = [], time.perf_counter()
    for p in problems:
        t0 = time.perf_counter()
        if arm == "raw":
            turn = harness.twil.ask([{"role": "user", "content": p["question"]}],
                                    budget=4096, thinking=True)
            content, diff, toks = turn.content, 2, turn.tokens_out
        else:
            content, r = harness.ask(p["question"])
            d = r.plan
            diff, toks = d["difficulty"], r.generation.get("tokens_out", 0)
        rows.append(Result(p["id"], p["level"], p["answer"], content,
                           same_answer(p["answer"], content), toks or _approx(content),
                           time.perf_counter() - t0, diff))
    return rows, dict(n=len(rows), seconds=round(time.perf_counter() - wall, 1),
                      accuracy=sum(r.correct for r in rows) / len(rows),
                      tokens=sum(r.tokens for r in rows))


def _approx(text):  # only when server omits usage
    from .twil import split_thinking
    return max(1, len(split_thinking(text)[1]) // 4)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="twila eval")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--mode", choices=["routed", "raw", "both"], default="both")
    ap.add_argument("--budget", type=int, default=16384)  # backward-compat, unused by arms
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--checkpoint", default=None)
    ns = ap.parse_args(argv)

    from rich.console import Console
    from rich.table import Table

    console = Console()
    problems = [json.loads(l) for l in EVALS.read_text().splitlines() if l.strip()][: ns.limit]
    harness = Harness(base_url=ns.base_url, checkpoint=ns.checkpoint)
    arms = {"routed", "raw"} if ns.mode == "both" else {ns.mode}

    stats, tables = {}, {}
    for arm in sorted(arms):
        rows, st = run_arm(harness, problems, arm)
        stats[arm] = st
        t = Table(title=f"{arm} — acc {st['accuracy']:.0%} · tok {st['tokens']:,} · {st['seconds']}s")
        t.add_column("id"); t.add_column("lvl"); t.add_column("gate")
        t.add_column("ok"); t.add_column("tok", justify="right"); t.add_column("s", justify="right")
        for r in rows:
            t.add_row(r.id, str(r.level), str(r.gate_level),
                      "[green]✓[/]" if r.correct else "[red]✗[/]",
                      f"{r.tokens:,}", f"{r.seconds:.0f}")
        tables[arm] = t
    for t in tables.values():
        console.print(t)
    if {"routed", "raw"} <= set(stats):
        rr, rw = stats["routed"], stats["raw"]
        d_tok = 1 - rr["tokens"] / rw["tokens"] if rw["tokens"] else 0
        console.print(f"[bold]\\[routing verdict][/bold] tokens {d_tok:+.0%}, "
                      f"accuracy {rr['accuracy'] - rw['accuracy']:+.0%}")
