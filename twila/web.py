"""twila web — the decisions, visible.

uv run twila web   ->  http://127.0.0.1:2790
"""

from __future__ import annotations

import json
from pathlib import Path

from .harness import RECEIPTS_DIR, Harness

_page = (Path(__file__).parent / "_web.html").read_text()
_HARNESS: list = []  # lazy: llama-server owns RAM while idle; laya loads on first ask


def _get_harness():
    if not _HARNESS:
        _HARNESS.append(Harness())
    return _HARNESS[0]

    app = FastAPI(title="twila")

def create_app():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="twila")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _page

    @app.post("/api/ask")
    def ask(body: dict):
        message = (body.get("message") or "").strip()
        if not message:
            return JSONResponse({"error": "empty message"}, status_code=400)
        h = _get_harness()
        answer, r = h.ask(message)
        decision = h.gate.last_decision
        return {
            "answer": answer,
            "gate": {"domain": r.plan["domain"], "difficulty": r.plan["difficulty"],
                     "budget": r.plan["budget"], "thinking": r.plan["thinking"],
                     "latency_ms": r.plan["gate_latency_ms"]},
            "verify": None if r.verify is None else {k: r.verify[k] for k in ("answered", "confidence")},
            "generation": r.generation,
            "retried": r.retried,
            "refused": bool(r.plan.get("thinking") is None) if False else False,
            "receipt_id": r.id,
        }

    @app.get("/api/stats")
    def stats():
        files = sorted(RECEIPTS_DIR.glob("*.json"))
        n, ms, tok = len(files), [], 0
        doms = {}
        file_items = []
        for f in files[-200:]:
            file_items.append(f)
        for f in file_items:
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            ms.append(d["plan"].get("gate_latency_ms", 0))
            tok += d.get("generation", {}).get("tokens_out", 0) or 0
            doms[d["plan"].get("domain", "?")] = doms.get(d["plan"].get("domain", "?"), 0) + 1
        return {"decisions": n, "avg_gate_ms": round(sum(ms) / len(ms)) if ms else None,
                "tokens_out": tok, "domains": doms}

    return app


def main():
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=2790, log_level="warning")
