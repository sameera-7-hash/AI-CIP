# MVP Notes — read before the review / demo

Working notes on what this MVP actually does, what it fakes, and what to
say if someone on the review panel pokes at it. Written to be read
alongside `README.md`, not instead of it.

## What "MVP" means here specifically

This is Phase 0 + a slice of Phase 1 from the proposal — the smallest
thing that proves the full request path end to end:

`HTTP request → gateway validates it → LangGraph agent runs a tool node
→ folds that into a prompt → GPU-served model answers → response comes
back with real latency numbers.`

It is **not** the vertical slice (Phase 2) and should not be presented as
such. The proposal itself draws this line — Phase 0's whole purpose is to
"prove the hardest part... quickly and cheaply, before the team commits
to building out every subsystem in parallel."

## Decisions made while building this MVP (and why)

- **Qwen2.5-3B as the default model, not 7B.** The proposal lists 7B with
  3B as a VRAM fallback. Default to 3B here so the MVP runs on more GPUs
  (fits comfortably in 16GB) and downloads/starts faster during
  iteration. Bump `VLLM_MODEL` in `.env` to the 7B checkpoint once you're
  on hardware that comfortably fits it and want the stronger
  instruction-following the proposal calls out.
- **One fake tool, not zero.** A model-only call would not satisfy the
  Definition of Done ("calls at least one tool"), and a real tool
  (nvidia-smi / FAISS) is Phase 2 scope. The fake tool is written so its
  *call signature* (`goal: str -> dict`) won't need to change when you
  swap in the real one — only the body of `fake_context_tool` changes.
- **Timeout + fallback is implemented, not just mentioned.** Section 7
  lists "tool call hangs" as a must-handle failure mode for this phase.
  `agent/graph.py:tool_node` wraps the tool call in `asyncio.wait_for`
  and the model node checks for `tool_error` and explicitly tells the
  model it had no context, rather than silently guessing. This is one of
  the two failure modes worth demonstrating live (see below).
- **Approximate tokens/sec in the simulator.** vLLM's actual response
  includes real usage stats; this MVP estimates from word count instead
  to avoid coupling the simulator to a response field the gateway
  doesn't currently pass through. Flag this honestly if asked for exact
  throughput numbers — it's a known, called-out gap, not a hidden one.
- **No Redis, no auth, no multi-tenancy.** All explicitly out of scope
  per Section 2.3 of the proposal for this phase. Don't build these into
  the MVP even if it looks easy — scope creep here eats the time
  budgeted for the real Phase 2 tools.

## What to demonstrate live (and how)

1. **Happy path.** Run `./scripts/smoke_test.sh` — shows vLLM answering
   directly, the gateway's health check, and a full agent invoke with
   `tool_context` visibly present in the response (proves the tool
   actually ran and its output reached the model).
2. **Tool timeout fallback.** Temporarily set `TOOL_TIMEOUT_SECONDS=0` in
   `.env` (or pass it as an env var) and re-run the smoke test — you'll
   see `tool_error` populated and the model still answering, just without
   tool context. This is the one failure mode from Section 7 that's
   fully wired up in this MVP; call it out explicitly.
3. **Concurrency numbers.** Run
   `python -m simulator.load_test --levels 1 5 10 20` live, or show a
   pre-run `simulator/results.csv`. This is the "measured, not estimated"
   principle from the proposal's closing section, in practice.

## Gaps to be upfront about (don't get caught flat-footed)

- GPU OOM handling under high concurrency (Section 7, "High — service
  crash") is **not yet demonstrated** in this MVP — vLLM's own
  `--gpu-memory-utilization` cap provides some protection, but no test
  here proves queuing-not-crashing behavior under real pressure. Next
  step: push the simulator to a concurrency level that saturates VRAM
  and confirm requests queue rather than the process dying.
- Startup health gating is implemented at the Compose level (`gateway`
  waits on `vllm`'s health check) but not exercised by an actual "start
  gateway before vLLM is ready" test.
- No streaming responses yet — the proposal's inference contract
  (Section 3.3) includes a `stream` boolean; this MVP always requests
  non-streaming completions for simplicity. Wire up streaming when the
  gateway needs to forward partial tokens to a caller.
- `allowed_tools` is accepted in the request schema but not enforced —
  meaningless right now with only one tool, becomes real work once
  Phase 2 adds a second tool.

## Suggested order for extending this into Phase 2

1. Swap `fake_context_tool` for a real `nvidia-smi` read (fast, no new
   dependencies — Python's `subprocess` is enough).
2. Add the FAISS + sentence-transformers retrieval tool as a second node,
   and let the graph route between the two based on `allowed_tools` /
   goal content — this is where LangGraph's branching actually earns its
   keep over the two-node MVP graph.
3. Wire vLLM's real `usage.total_tokens` through the gateway response so
   the simulator's tokens/sec stops being an estimate.
4. Add the GPU-OOM stress test called out above before claiming that
   failure mode as "handled" in the final report.
5. Only then reach for Redis, Prometheus, and the rest of Section 5.4's
   roadmap items — they're explicitly deferred for a reason.
