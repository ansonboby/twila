#!/usr/bin/env bash
# Autonomous smoke eval: waits for the llama server, runs routed vs raw on 3
# problems, writes a JSON verdict. Survives session churn (setsid'd).
# Output: /tmp/twila_smoke_result.json + /tmp/twila_smoke.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=/tmp/twila_smoke.log
echo "waiting for llama-server health..." > "$LOG"
for i in $(seq 1 60); do
  curl -s --max-time 3 http://127.0.0.1:2718/health | grep -q ok && break
  sleep 5
done
echo "server healthy (or given up after retries); starting eval" >> "$LOG"

"$PY" -u - << 'EOF' >> "$LOG" 2>&1
import json, time
from pathlib import Path
from twila.eval import run_arm
from twila.harness import Harness

probs = [json.loads(l) for l in Path("evals/math500_mini.jsonl").read_text().splitlines() if l.strip()][:3]
h = Harness()
res = {}
for arm in ("routed", "raw"):
    t0 = time.time()
    rows, st = run_arm(h, probs, arm)
    res[arm] = st
    print(arm, st, [(r.id, r.correct, r.tokens) for r in rows], flush=True)
res["verdict"] = {"tokens_saved": 1 - res["routed"]["tokens"]/max(res["raw"]["tokens"],1),
                  "acc_delta": res["routed"]["accuracy"] - res["raw"]["accuracy"]}
Path("/tmp/twila_smoke_result.json").write_text(json.dumps(res, indent=2))
print("DONE", res["verdict"], flush=True)
EOF
echo "smoke eval finished" >> "$LOG"
