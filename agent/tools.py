"""
Phase 0 tools for the G1 orchestration graph.

Per the technical proposal (Section 5.1, Phase 0 — Proof of Concept), the
goal here is to prove the *shape* of the pipeline — a tool call that feeds
context into the model call — before building the real GPU-metrics
(nvidia-smi) and FAISS retrieval tools called for in Phase 2.

This module intentionally ships a single fake tool, `fake_context_tool`,
that returns a canned but plausible-looking payload. Swap its body for a
real `nvidia-smi` read or a FAISS similarity search when you move into
Phase 2 — the call signature (`goal: str -> dict`) is designed to stay
the same so the graph in `agent/graph.py` does not need to change.
"""

from __future__ import annotations

import time


def fake_context_tool(goal: str) -> dict:
    """
    Stand-in for the real tools (GPU metrics / retrieval) described in the
    proposal. Returns deterministic, inspectable "context" so you can see,
    end to end, that the tool's output actually reaches the final prompt.

    Replace this with:
      - a live `nvidia-smi --query-gpu=...` read (Phase 2 GPU-metrics tool), or
      - a FAISS similarity search against an embedded knowledge base
        (Phase 2 retrieval tool).
    """
    time.sleep(0.05)  # simulate a small amount of real tool latency
    return {
        "source": "fake_context_tool",
        "goal_received": goal,
        "note": (
            "This is placeholder context from Phase 0. In Phase 2 this "
            "will be replaced by a live nvidia-smi read and/or a FAISS "
            "retrieval hit against a real knowledge base."
        ),
        "sample_metric": {"gpu_util_pct": 42.0, "mem_used_mb": 8192},
    }
