#!/usr/bin/env bash
# Smoke eval, crash-proof edition: one problem at a time, 2 retries each,
# partial results written after every problem. Run as a systemd unit:
#   systemd-run --user --unit=twila-smokeN -p WorkingDirectory=$PWD bash scripts/smoke_eval.sh
# Output: /tmp/twila_smoke_result.json + /tmp/twila_smoke.log
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=/tmp/twila_smoke.log
OUT=/tmp/twila_smoke_result.json
rm -f "$OUT"
echo "waiting for llama-server health..." > "$LOG"
for i in $(seq 1 60); do
  curl -s --max-time 3 http://127.0.0.1:2718/health | grep -q ok && break
  sleep 5
done
curl -s --max-time 3 http://127.0.0.1:2718/health | grep -q ok || { echo "server never became healthy" >> "$LOG"; exit 1; }
echo "server healthy; starting eval" >> "$LOG"

"$PY" -u - << 'EOF' >> "$LOG" 2>&1
import json, time
from pathlib import Path
from twila.eval import run_arm
from twila.harness import Harness

probs = [json.loads(l) for l in Path("evals/math500_mini.jsonl").read_text().splitlines() if l.strip()][:3]
h = Harness()
res = {"routed": {"tokens": 0, "correct": 0, "n": 0, "seconds": 0.0, "rows": []},
       "raw":    {"tokens": 0, "correct": 0, "n": 0, "seconds": 0.0, "rows": []}}

def save():
    Path("/tmp/twila_smoke_result.json").write_text(json.dumps(res, indent=2))

for arm in ("routed", "raw"):
    for p in probs:
        for attempt in (1, 2):
            try:
                rows, st = run_arm(h, [p], arm)
                r = rows[0]
                a = res[arm]
                a["tokens"] += r.tokens; a["correct"] += int(r.correct); a["n"] += 1
                a["seconds"] += r.seconds; a["seconds"] = round(a["seconds"], 1)
                a["rows"].append({"id": r.id, "correct": r.correct, "tokens": r.tokens,
                                  "gate_level": r.gate_level, "seconds": round(r.seconds, 1)})
                print(f"{arm} {r.id}: ok={r.correct} tok={r.tokens} gate={r.gate_level} "
                      f"({r.seconds:.0f}s, attempt {attempt})", flush=True)
                save()
                break
            except Exception as e:
                print(f"{arm} {p['id']}: attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
                time.sleep(15)

rt, rw = res["routed"], res["raw"]
res["verdict"] = {"tokens_saved": round(1 - rt["tokens"] / max(rw["tokens"], 1), 3),
                  "acc_delta": round((rt["correct"] / max(rt["n"], 1)) -
                                     (rw["correct"] / max(rw["n"], 1)), 3)}
print("DONE", res["verdict"], flush=True)
save()
EOF
echo "smoke eval finished" >> "$LOG"
