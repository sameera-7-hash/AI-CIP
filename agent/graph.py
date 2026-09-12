"""
Two-node LangGraph agent: tool node -> model node.

This is the graph called for by Section 5.4 (Definition of Done): "The
response is produced by a compiled LangGraph agent that calls at least one
tool" — not a linear if/else script. Phase 0 uses `fake_context_tool`;
Phase 2 swaps in the real GPU-metrics and FAISS tools without touching the
graph shape.

Failure handling (Section 7 of the proposal):
  - Tool call hangs -> explicit timeout, falls back to inference-only.
  - vLLM unreachable / errors -> caught and surfaced as a clear error
    message rather than a raw stack trace, so the gateway can decide how
    to respond to the caller.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional, TypedDict

import httpx
from langgraph.graph import StateGraph, END

from agent.tools import fake_context_tool

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "3"))


class AgentState(TypedDict, total=False):
    task_id: str
    goal: str
    max_tokens: int
    temperature: float
    tool_context: Optional[dict]
    tool_error: Optional[str]
    answer: Optional[str]
    model_error: Optional[str]
    timings_ms: dict


def _now_ms() -> float:
    return time.perf_counter() * 1000


async def tool_node(state: AgentState) -> AgentState:
    """Calls the (currently fake) context tool with a hard timeout."""
    t0 = _now_ms()
    goal = state["goal"]
    try:
        # fake_context_tool is synchronous/cheap; run it in a thread so a
        # slower real tool (nvidia-smi subprocess, FAISS query) can later
        # drop in here without blocking the event loop.
        result = await asyncio.wait_for(
            asyncio.to_thread(fake_context_tool, goal),
            timeout=TOOL_TIMEOUT_SECONDS,
        )
        state["tool_context"] = result
    except asyncio.TimeoutError:
        state["tool_context"] = None
        state["tool_error"] = f"tool call exceeded {TOOL_TIMEOUT_SECONDS}s timeout"
    except Exception as exc:  # pragma: no cover - defensive
        state["tool_context"] = None
        state["tool_error"] = f"tool call failed: {exc}"
    state.setdefault("timings_ms", {})["tool_node"] = round(_now_ms() - t0, 2)
    return state


async def model_node(state: AgentState) -> AgentState:
    """Calls the vLLM OpenAI-compatible chat completions endpoint."""
    t0 = _now_ms()
    goal = state["goal"]
    tool_context = state.get("tool_context")
    tool_error = state.get("tool_error")

    system_parts = [
        "You are the G1 platform's assistant. Ground your answer in the "
        "tool context provided below when it is relevant."
    ]
    if tool_context:
        system_parts.append(f"Tool context: {tool_context}")
    elif tool_error:
        system_parts.append(
            f"Note: the context tool failed ({tool_error}); answer from "
            "the goal alone and say you had no tool context."
        )

    payload = {
        "model": VLLM_MODEL,
        "messages": [
            {"role": "system", "content": "\n".join(system_parts)},
            {"role": "user", "content": goal},
        ],
        "max_tokens": state.get("max_tokens", 256),
        "temperature": state.get("temperature", 0.7),
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{VLLM_BASE_URL}/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            state["answer"] = data["choices"][0]["message"]["content"]
    except Exception as exc:
        state["model_error"] = str(exc)
        state["answer"] = None

    state.setdefault("timings_ms", {})["model_node"] = round(_now_ms() - t0, 2)
    return state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("tool_node", tool_node)
    graph.add_node("model_node", model_node)
    graph.set_entry_point("tool_node")
    graph.add_edge("tool_node", "model_node")
    graph.add_edge("model_node", END)
    return graph.compile()


# Compiled once at import time and reused across requests.
compiled_graph = build_graph()


async def run_agent(
    task_id: str,
    goal: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> AgentState:
    initial_state: AgentState = {
        "task_id": task_id,
        "goal": goal,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    return await compiled_graph.ainvoke(initial_state)
