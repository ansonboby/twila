#!/usr/bin/env bash
# Serve TwIL-8B (Q4_K_M) to twila. First call downloads the 5GB GGUF.
set -euo pipefail
cd "$(dirname "$0")/.."

GGUF=models/TwIL-8B-Q4_K_M-balanced.gguf
PORT=2718

if [ ! -s "$GGUF" ]; then
  echo "downloading $GGUF (~5GB)..."
  curl -L -C - --retry 5 -o "$GGUF" \
    "https://huggingface.co/webAI-Official/TwIL-8B/resolve/main/TwIL-8B-Q4_K_M-balanced.gguf"
fi

# 10 threads / 2 parallel slots (40k ctx split 2 -> 20.5k each, fits prompt+16k
# trail) / 32k max gen: the card's contest-math budgets, CPU-sized. -np 2 keeps
# interactive asks off the eval bench queue.
exec llama serve -m "$GGUF" --port "$PORT" -t 10 -c 40960 -n 32768 -np 2 "$@"
