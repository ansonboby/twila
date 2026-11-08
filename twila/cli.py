"""twila — the CLI. ask, chat, eval. Serve llama-server before use: scripts/serve up."""

from __future__ import annotations

import argparse
import json
import pathlib

from .twil import DEFAULT_SERVER


def _harness(ns):
    from .harness import Harness

    return Harness(base_url=ns.base_url, checkpoint=ns.checkpoint, variant=ns.variant)


def cmd_ask(ns):
    from rich.console import Console

    console = Console()
    answer, r = _harness(ns).ask(ns.message)
    tag = (f"[dim]laya {r.plan['gate_latency_ms']}ms · lvl {r.plan['difficulty']} · "
           f"task {r.plan['domain']} · budget {r.plan['budget']:,} · thinking "
           f"{'on' if r.plan['thinking'] else 'off'} · gen {r.generation.get('latency', 0):.1f}s · "
           f"{r.generation.get('tokens_out', 0)} tok · {'retry' if r.retried else 'ok'}[/dim]")
    console.print(tag)
    console.print(answer)


def cmd_chat(ns):
    from rich.console import Console

    h = _harness(ns)
    console = Console()
    console.print("[bold cyan]twila[/bold cyan] — laya gates, TwIL-8B thinks. "
                  "[dim]/v to see last receipt · ctrl-C to quit[/dim]")
    last = None
    while True:
        try:
            message = console.input("[bold cyan]you ›[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            break
        if not message.strip():
            continue
        if message.strip() == "/v" and last is not None:
            console.print_json(json.dumps({"plan": last.plan, "verify": last.verify,
                                           "generation": last.generation}, default=str))
            continue
        answer, last = h.ask(message)
        if h.gate.last_decision and not h.gate.last_decision.policy_violation:
            d = h.gate.last_decision
            console.print(f"[dim]laya {d.latency_ms}ms · lvl {d.difficulty} · "
                          f"task {d.domain} · budget {last.plan['budget']:,} · gen "
                          f"{last.generation.get('latency', 0):.1f}s · "
                          f"{last.generation.get('tokens_out', 0)} tok · "
                          f"{last.plan.get('budget') and 'ok' or 'refused'}[/dim]")
        console.print(answer)
        console.print()


def cmd_receipts(ns):
    from .harness import RECEIPTS_DIR

    files = sorted(RECEIPTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)[-ns.n:]
    for f in files:
        console.print(json.dumps(json.loads(f.read_text()), indent=2, default=str))


def cmd_eval(ns):
    from .eval import main as eval_main

    eval_main(["--limit", str(ns.limit), "--budget", str(ns.budget), "--mode", ns.mode,
               *(["--base-url", ns.base_url] if ns.base_url else []),
               *(["--checkpoint", ns.checkpoint] if ns.checkpoint else [])])


def main(argv=None):
    ap = argparse.ArgumentParser(prog="twila", description=__doc__)
    ap.add_argument("--base-url", default=DEFAULT_SERVER)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--variant", default="balanced", choices=["balanced", "verbose"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ask"); p.add_argument("message"); p.set_defaults(fn=cmd_ask)
    p = sub.add_parser("chat"); p.set_defaults(fn=cmd_chat)
    p = sub.add_parser("receipts"); p.add_argument("-n", type=int, default=3);
    p.set_defaults(fn=cmd_receipts)
    p = sub.add_parser("eval")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--budget", type=int, default=16384)
    p.add_argument("--mode", choices=["routed", "raw", "both"], default="both")
    p.set_defaults(fn=cmd_eval)

    ns = ap.parse_args(argv)
    ns.fn(ns)
