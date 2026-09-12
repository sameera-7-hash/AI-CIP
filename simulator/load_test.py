"""
Concurrent request simulator for the G1 platform (Section 8: Validation,
Metrics & Definition of Done).

Runs the gateway at concurrency levels 1, 5, 10, and 20 (configurable),
and reports/saves, per level:
  - p50 and p95 end-to-end latency (ms)
  - aggregate tokens/sec (approximated from response length; see note
    below — wire in real vLLM usage stats for an exact number)
  - success vs failure count

Usage:
    python -m simulator.load_test --url http://localhost:8080/v1/agent/invoke \
        --levels 1 5 10 20 --requests-per-level 10 \
        --goal "Summarize current GPU utilization and recommend an action."

Output:
    Prints a results table to stdout and writes simulator/results.csv.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx

DEFAULT_GOAL = "Summarize current GPU utilization and recommend an action."


async def _one_request(client: httpx.AsyncClient, url: str, goal: str) -> dict:
    t0 = time.perf_counter()
    ok = True
    tokens_est = 0
    try:
        resp = await client.post(url, json={"goal": goal}, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer") or ""
        # Rough token estimate (words * 1.3) purely for a throughput signal
        # in the MVP. Replace with real usage.total_tokens from the vLLM
        # response once the gateway passes that field through.
        tokens_est = max(1, int(len(answer.split()) * 1.3))
    except Exception:
        ok = False
    latency_ms = (time.perf_counter() - t0) * 1000
    return {"ok": ok, "latency_ms": latency_ms, "tokens_est": tokens_est}


async def _run_level(url: str, goal: str, concurrency: int, requests_per_level: int) -> dict:
    total_requests = concurrency * requests_per_level
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def bound():
            async with sem:
                return await _one_request(client, url, goal)

        t0 = time.perf_counter()
        results = await asyncio.gather(*[bound() for _ in range(total_requests)])
        wall_s = time.perf_counter() - t0

    latencies = sorted(r["latency_ms"] for r in results if r["ok"])
    failures = sum(1 for r in results if not r["ok"])
    total_tokens = sum(r["tokens_est"] for r in results if r["ok"])

    def pctile(data, p):
        if not data:
            return float("nan")
        k = (len(data) - 1) * p
        f, c = int(k), min(int(k) + 1, len(data) - 1)
        return data[f] + (data[c] - data[f]) * (k - f)

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "failures": failures,
        "p50_ms": round(pctile(latencies, 0.50), 1),
        "p95_ms": round(pctile(latencies, 0.95), 1),
        "tokens_per_sec": round(total_tokens / wall_s, 1) if wall_s > 0 else 0.0,
        "wall_s": round(wall_s, 2),
    }


async def main_async(args):
    rows = []
    for level in args.levels:
        print(f"Running concurrency={level} ...")
        row = await _run_level(args.url, args.goal, level, args.requests_per_level)
        rows.append(row)
        print(
            f"  p50={row['p50_ms']}ms  p95={row['p95_ms']}ms  "
            f"tok/s~={row['tokens_per_sec']}  failures={row['failures']}/{row['total_requests']}"
        )

    out_path = Path(__file__).parent / "results.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="G1 MVP load simulator")
    parser.add_argument("--url", default="http://localhost:8080/v1/agent/invoke")
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--requests-per-level", type=int, default=10)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
