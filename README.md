# G1 Platform — MVP (Phase 0/1 Vertical Slice)

This is the minimum viable prototype for **Team G1's CUDA-Accelerated LLM
Inference & Agent Orchestration Platform**, scoped directly from the
technical proposal's Phase 0 ("Proof of Concept") and Phase 1
("Foundation & Architecture") exit criteria. It is intentionally small:
one fake tool, one real GPU-backed model call, one gateway endpoint, one
load simulator — enough to prove the *shape* of the full platform before
building out the real tools in Phase 2.

**What this MVP proves:** a caller can hit a single HTTP endpoint, have a
compiled LangGraph agent decide to call a tool, fold that tool's output
into the prompt, get a real answer back from a GPU-served LLM (via vLLM),
and see the whole thing hold up under concurrent load — with numbers to
show for it, not a demo that only works once.

## Architecture (this slice)

```
        POST /v1/agent/invoke
               |
               v
      +-------------------+
      |  FastAPI Gateway   |   gateway/main.py
      +-------------------+
               |
               v
      +-------------------+
      |  LangGraph agent   |   agent/graph.py
      |  tool_node -> model_node
      +-------------------+
        |               |
        v               v
  fake_context_tool   vLLM (OpenAI-compatible /v1/chat/completions)
  agent/tools.py        served by an actual GPU
```

`tool_node` and `model_node` are separate graph nodes (not an if/else
script), matching the Definition of Done in Section 8 of the proposal:
*"The response is produced by a compiled LangGraph agent that calls at
least one tool."*

## What's real vs. placeholder in this MVP

| Piece | Status | Notes |
|---|---|---|
| GPU-served LLM (vLLM) | **Real** | Needs an actual NVIDIA GPU (16GB+ VRAM recommended). This is not mocked. |
| FastAPI gateway | **Real** | `/v1/agent/invoke` and `/health` both fully functional. |
| LangGraph two-node graph | **Real** | Compiled graph, not a linear script; timeout/fallback on the tool node. |
| Context tool | **Placeholder** | `agent/tools.py:fake_context_tool` returns canned data. Phase 2 replaces this with a live `nvidia-smi` read and a FAISS retrieval query. |
| Load simulator | **Real**, tokens/sec is **approximate** | Latency numbers (p50/p95) are real measurements. Tokens/sec is estimated from response word count; wire in vLLM's real `usage.total_tokens` for an exact figure (see `simulator/load_test.py` comment). |
| Redis / session state | **Not included** | Out of scope for this MVP; add when multi-turn conversation state is needed. |
| Docker Compose w/ GPU passthrough | **Included** | `docker-compose.yml`; requires the NVIDIA Container Toolkit on the host. |

## Prerequisites

- A machine with an NVIDIA GPU, recent driver, and (for the Docker path)
  the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed.
- Python 3.10+ if running the gateway outside Docker.
- ~15-20GB free disk for model weights (Qwen2.5-3B by default; see below
  for the larger 7B option).

## Quickstart — Option A: Docker Compose (recommended)

```bash
cp .env.example .env
# edit .env if you want a different model / ports

docker compose up --build
```

This starts two containers:

1. `vllm` — serves `Qwen/Qwen2.5-3B-Instruct` on `:8000` with an
   OpenAI-compatible API. First start will download the model weights
   (several GB) — expect the health check to take a few minutes.
2. `gateway` — the FastAPI service on `:8080`, which waits for vLLM's
   health check to pass before accepting traffic (mitigates the "model
   fails to load at startup" risk from Section 7 of the proposal).

Once both are healthy:

```bash
./scripts/smoke_test.sh
```

## Quickstart — Option B: Run locally without Docker

Terminal 1 — start vLLM directly (requires `pip install vllm` on a
CUDA-capable machine):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

Terminal 2 — start the gateway:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn gateway.main:app --host 0.0.0.0 --port 8080
```

Terminal 3 — smoke test:

```bash
./scripts/smoke_test.sh
```

## Running the load simulator

Once the gateway is up and answering `./scripts/smoke_test.sh`
successfully, run the concurrency sweep called for in Section 8:

```bash
python -m simulator.load_test --levels 1 5 10 20 --requests-per-level 10
```

This prints a p50/p95 latency and tokens/sec table per concurrency level
and writes `simulator/results.csv`. Use this to fill in Section 8's
metrics table with real, measured numbers instead of estimates — that's
the whole point of the exercise per the proposal's stated principle:
*"Measured results over impressive claims."*

## API reference (MVP surface)

### `POST /v1/agent/invoke`

Request body (matches Section 3.3's "Agent task request" contract):

```json
{
  "goal": "Summarize current GPU utilization and recommend an action.",
  "max_tokens": 256,
  "temperature": 0.7
}
```

`task_id`, `context`, and `allowed_tools` are accepted but not yet
enforced in this MVP (there is only one tool, so "allowed_tools" has
nothing to select between yet — wire this up when Phase 2 adds a second
real tool).

Response:

```json
{
  "task_id": "…",
  "answer": "…",
  "tool_context": { "source": "fake_context_tool", "...": "..." },
  "tool_error": null,
  "model_error": null,
  "latency_ms": 812.4,
  "timings_ms": { "tool_node": 51.2, "model_node": 760.9 }
}
```

### `GET /health`

Reports gateway status and whether vLLM is currently reachable —
used both by a human curling it and by Compose's own health check.

## Known limitations of this MVP (be upfront about these in review)

- The context tool is fake. It proves the wiring, not domain knowledge.
- No authentication, rate limiting, or multi-tenancy — explicitly out of
  scope per Section 2.3 of the proposal for this phase.
- No Redis-backed session/conversation state — single-turn only.
- Tokens/sec in the simulator is an estimate, not read from vLLM's actual
  usage stats.
- Only one failure mode has a demonstrated code path in this MVP (tool
  timeout → fallback to inference-only). The proposal's Definition of
  Done asks for three; the other two (GPU OOM handling, startup health
  gating) are partially covered by the Compose health check but need a
  deliberate test to demonstrate under load — see `NOTES.md`.

## Where this goes next (Phase 2, not in this MVP)

- Replace `fake_context_tool` with a real `nvidia-smi` read and a real
  FAISS + sentence-transformers retrieval tool.
- Let the LangGraph supervisor choose between multiple tools per
  request instead of always calling the one fake tool.
- Add Redis for session/conversation state.
- Add Prometheus scraping of the metrics the gateway already times
  (`timings_ms`) plus GPU utilization samples.
- Swap the approximate simulator tokens/sec for vLLM's real
  `usage.total_tokens`.

See the full technical proposal (`G1_Technical_Proposal_Tender.docx`) for
the complete phased plan, team responsibilities, and risk register.
