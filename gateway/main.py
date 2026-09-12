"""
FastAPI gateway for the G1 CUDA-Accelerated LLM Inference & Agent
Orchestration Platform (MVP / Phase 0-1 slice).

Exposes:
  POST /v1/agent/invoke  - the agent task request contract from
                            Section 3.3 of the technical proposal.
  GET  /health            - readiness probe; used by Compose's health
                            check and by the gateway's own startup gate
                            (Section 7 risk: "model fails to load at
                            startup").

Run directly with:
    uvicorn gateway.main:app --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.graph import VLLM_BASE_URL, run_agent

GATEWAY_HOST = os.getenv("GATEWAY_HOST", "0.0.0.0")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "8080"))

app = FastAPI(
    title="G1 Agent Orchestration Gateway",
    description="MVP gateway: validates requests, runs the LangGraph agent, "
    "returns a grounded response.",
    version="0.1.0",
)


class AgentTaskRequest(BaseModel):
    """Matches Section 3.3 'Agent task request (gateway -> orchestration layer)'."""

    task_id: Optional[str] = Field(default=None, description="Client-supplied trace id; generated if omitted.")
    goal: str = Field(..., min_length=1, description="Natural-language objective the agent must act on.")
    context: Optional[dict[str, Any]] = Field(default=None, description="Conversation id / prior turns, if any.")
    allowed_tools: Optional[list[str]] = Field(default=None, description="Tools the orchestration layer may call.")
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class AgentTaskResponse(BaseModel):
    task_id: str
    answer: Optional[str]
    tool_context: Optional[dict[str, Any]]
    tool_error: Optional[str]
    model_error: Optional[str]
    latency_ms: float
    timings_ms: dict[str, float]


@app.get("/health")
async def health():
    """
    Reports the gateway's own status plus whether it can currently reach
    the vLLM server. Compose's health check and the gateway's startup
    gate (Section 7: "model fails to load at startup") both key off this.
    """
    vllm_ok = False
    detail = None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(f"{VLLM_BASE_URL}/models")
            vllm_ok = resp.status_code == 200
    except Exception as exc:
        detail = str(exc)

    return {
        "gateway": "ok",
        "vllm_reachable": vllm_ok,
        "vllm_base_url": VLLM_BASE_URL,
        "detail": detail,
    }


@app.post("/v1/agent/invoke", response_model=AgentTaskResponse)
async def invoke_agent(request: AgentTaskRequest):
    task_id = request.task_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        result = await run_agent(
            task_id=task_id,
            goal=request.goal,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"agent run failed: {exc}") from exc

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return AgentTaskResponse(
        task_id=task_id,
        answer=result.get("answer"),
        tool_context=result.get("tool_context"),
        tool_error=result.get("tool_error"),
        model_error=result.get("model_error"),
        latency_ms=latency_ms,
        timings_ms=result.get("timings_ms", {}),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("gateway.main:app", host=GATEWAY_HOST, port=GATEWAY_PORT, reload=False)
