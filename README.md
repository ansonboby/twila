# twila

> **90% of frontier reasoning. $0 API spend.** A 421M model decides *how* to think; an 8.8B model does the thinking. Everything runs on your laptop.

`twila` pairs two open models into one harness:

| | Role | Model | Size | Speed |
|---|---|---|---|---|
| **System 1** | decides: domain, difficulty, effort, policy, confidence | [Laya](https://laya-ai.com) (typed decisions: choice/score/probability) | 421M | ~50ms, one forward pass, no text |
| **System 2** | reasons and answers | [TwIL-8B](https://huggingface.co/webAI-Official/TwIL-8B) (math/code/instruction) | 8.8B | gated GGUF via llama.cpp |

The idea: **decisions are cheap, generation is expensive.** Laya triages every
request in milliseconds — *how hard is this, should the big model think at all,
how big is the token budget, is this safe to answer* — and TwIL-8B reasons only
as hard as the decision says. A post-gate double-checks the answer and retries
with an escalated budget if Laya doesn't believe it.

## Why this is interesting

- **Dynamic effort routing.** "What's 2+2" gets 2k tokens and no thinking; an
  AIME problem gets 16k and full reasoning. Same model, 8x token spread.
- **Receipts.** Every turn logs the gate decision, verify scores, tokens and
  latency to `~/.twila/receipts/`. The decision pipeline is audible.
- **The eval that proves it.** `twila eval` runs the same problem set through
  raw (one-size-fits-all) and routed (laya) arms and prints the delta:
  accuracy held, tokens cut.
- **Free fine-tune data.** Receipts are (decision → outcome) pairs on *your*
  traffic — periodically fine-tune the gate on them and the router gets
  better at exactly what you run through it.

## Quickstart

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if needed
git clone <this-repo> && cd twila
uv sync

# terminal 1: serve TwIL-8B (Q4_K_M, ~5.3GB RAM, first run downloads ~5GB)
uv run bash scripts/serve.sh

# terminal 2
uv run twila ask "How many positive integers n < 1000 are divisible by 3 but not by 5?"
# ✦ laya 38ms · lvl 2 · math · budget 16,384 · gen 61.2s · 1,942 tok · ok
# There are 267 such integers.

uv run twila chat        # REPL; shows the inline receipt every turn
uv run twila eval        # raw vs routed, the money table
```

## The receipt

```json
{
  "plan":   {"domain": "math", "difficulty": 2, "budget": 16384,
             "thinking": true, "gate_latency_ms": 38},
  "verify": {"answered": 0.91, "confidence": 0.87},
  "generation": {"tokens_out": 1942, "latency": 61.2, "variant": "balanced"}
}
```

## Hardware notes

Built on an i5-1235U / 15GB laptop, no GPU: the gate overhead is noise, the
reasoner is the bottleneck. Budgets in `twila/gate.py` are CPU-sized
(16k max trail). On better hardware raise them; the policy is shape-equal.

## Pluggable System-1: Laya local, or Jev / hosted Laya

Every decision model in the Jev ecosystem speaks the same `POST /v1/systemone`
format, so the gate is a base-URL configuration, not a rewrite:

```bash
# default: local 421M checkpoint — free, ~20-50ms, private, fine-tunable
uv run twila chat

# any hosted gate — Jev (TypeSafe) or hosted Laya (api.laya-ai.com):
TWILA_GATE=http \
TWILA_GATE_BASE_URL=https://api.laya-ai.com/v1/systemone \
TWILA_GATE_API_KEY=... \
uv run twila chat
```

Same-run head-to-head ([source](https://laya-ai.com/laya-vs-jev)), so you can
pick per deployment:

| | Jev (hosted) | Laya (local) |
|---|---|---|
| accuracy, 2-6 labels | 0.927-0.963 | 0.773-0.920 |
| accuracy, 77 labels | **0.813** | 0.363 |
| non-English | **0.872** | 0.544 |
| latency | ~410ms (network) | **18-54ms** |
| cost | $0.042/M input tok | free |
| calibration | good everywhere | good small sets only |
| fine-tunable | no | **yes** (the receipts flywheel) |

Rule of thumb: our chat gate uses 3-4 choices, where Laya is close to Jev.
If your app grows large choice sets or non-English traffic, switch the gate;
nothing else changes. `twila eval` runs gate-vs-gate as arms — publish your
own comparison table, nobody has one for the router role.

## Files

```
twila/gate.py      Laya system-1: PRE_QUESTIONS, LEVELS policy, verify()
twila/twil.py      llama-server client, thinke/visual> parser
twila/harness.py   gate → generate → verify → retry loop, receipts
twila/eval.py      raw-vs-routed runner + scorer
evals/             problem sets (math500-style mini)
tests/             gate/scorer tests (pytest)
scripts/serve.sh   download + serve the GGUF
```

## Honest limits

- Laya's default checkpoints are weak zero-shot on domains far from their
  training distribution — thresholds `0.5`/`0.85` here are first-guesses, not
  calibrated. Calibrate on your own receipts.
- `choice` degrades over ~20 options (upstream docs); chunk-filter large
  candidate sets via `noul` first.
- TwIL-8B ships a webai non-commercial license; Laya is Apache-2.0.
- Q4_K_M quantization costs some score vs the card's bf16 evals.

## License

MIT for this repo's code. Model weights keep their upstream licenses.
