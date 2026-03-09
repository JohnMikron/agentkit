#!/usr/bin/env python3
"""
Debugging and observability example.

This example demonstrates:
- Event hooks for monitoring agent execution
- Debugging tools
- Logging configuration
"""

import asyncio
import json

from agentkit import Agent
from agentkit.core.types import Event, EventType
from agentkit.utils.logging import get_logger, setup_logging


async def main():
    """Run debugging examples."""

    # Configure logging
    setup_logging(level="DEBUG", format="text")
    get_logger(__name__)

    # ============================================================
    # Example 1: Basic Debugging Hooks
    # ============================================================
    print("=" * 60)
    print("Example 1: Basic Debugging Hooks")
    print("=" * 60)

    agent = Agent("debug_agent", model="gpt-4o-mini")

    # Add a simple tool
    @agent.tool
    def echo(text: str) -> str:
        """Echo back the input text."""
        return f"Echo: {text}"

    # Register debugging hooks
    call_log = []

    @agent.on_start
    def on_start(event: Event):
        print("\n🚀 Agent started")
        call_log.append(("start", None))

    @agent.on_end
    def on_end(event: Event):
        iterations = event.data.get("iterations", 0)
        print(f"\n🏁 Agent finished in {iterations} iterations")
        call_log.append(("end", iterations))

    @agent.on_llm_request
    def on_llm_request(event: Event):
        print("\n📡 Sending request to LLM...")
        call_log.append(("llm_request", None))

    @agent.on_llm_response
    def on_llm_response(event: Event):
        has_tools = event.data.get("has_tool_calls", False)
        print(f"\n📥 LLM response received (has_tool_calls: {has_tools})")
        call_log.append(("llm_response", has_tools))

    @agent.on_tool_call_start
    def on_tool_start(event: Event):
        tool_name = event.data.get("tool_name", "unknown")
        args = event.data.get("arguments", {})
        print(f"\n🔧 Tool call: {tool_name}")
        print(f"   Arguments: {json.dumps(args, indent=2)}")
        call_log.append(("tool_start", tool_name))

    @agent.on_tool_call_end
    def on_tool_end(event: Event):
        tool_name = event.data.get("tool_name", "unknown")
        is_error = event.data.get("is_error", False)
        status = "❌" if is_error else "✅"
        print(f"\n{status} Tool completed: {tool_name}")
        call_log.append(("tool_end", tool_name))

    print("\nHooks registered. Execution log will be tracked.")

    # ============================================================
    # Example 2: Detailed Event Monitoring
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 2: Detailed Event Monitoring")
    print("=" * 60)

    detailed_agent = Agent("detailed_agent", model="gpt-4o-mini")

    # More detailed hooks
    @detailed_agent.on_tool_call_start
    def detailed_tool_monitor(event: Event):
        print("\n" + "-" * 40)
        print("[TOOL CALL START]")
        print(f"  Agent: {event.agent_name}")
        print(f"  Tool: {event.data.get('tool_name')}")
        print(f"  Call ID: {event.data.get('tool_call_id')}")
        print(f"  Arguments: {event.data.get('arguments')}")
        print("-" * 40)

    @detailed_agent.on_tool_call_end
    def detailed_tool_result(event: Event):
        print("\n" + "-" * 40)
        print("[TOOL CALL END]")
        print(f"  Tool: {event.data.get('tool_name')}")
        print(f"  Is Error: {event.data.get('is_error')}")
        print("-" * 40)

    # ============================================================
    # Example 3: Error Handling Hook
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 3: Error Handling")
    print("=" * 60)

    error_agent = Agent("error_agent", model="gpt-4o-mini")

    errors = []

    @error_agent.on_error
    def on_error(event: Event):
        error_msg = event.data.get("error", "Unknown error")
        error_type = event.data.get("error_type", "Unknown")
        print(f"\n❌ Error occurred: {error_type}")
        print(f"   Message: {error_msg}")
        errors.append((error_type, error_msg))

    @error_agent.on_tool_call_end
    def check_tool_errors(event: Event):
        if event.data.get("is_error"):
            print(f"\n⚠️ Tool execution failed: {event.data.get('tool_name')}")

    print("\nError handling hooks registered.")

    # ============================================================
    # Example 4: Performance Monitoring
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 4: Performance Monitoring")
    print("=" * 60)

    perf_agent = Agent("perf_agent", model="gpt-4o-mini")

    import time

    timings = {}

    @perf_agent.on_llm_request
    def start_timer(event: Event):
        timings["llm_start"] = time.perf_counter()

    @perf_agent.on_llm_response
    def end_timer(event: Event):
        if "llm_start" in timings:
            elapsed = (time.perf_counter() - timings["llm_start"]) * 1000
            print(f"\n⏱️ LLM latency: {elapsed:.2f}ms")

    @perf_agent.on_tool_call_start
    def tool_start_timer(event: Event):
        tool_name = event.data.get("tool_name", "unknown")
        timings[f"tool_{tool_name}"] = time.perf_counter()

    @perf_agent.on_tool_call_end
    def tool_end_timer(event: Event):
        tool_name = event.data.get("tool_name", "unknown")
        key = f"tool_{tool_name}"
        if key in timings:
            elapsed = (time.perf_counter() - timings[key]) * 1000
            print(f"\n⏱️ Tool '{tool_name}' execution: {elapsed:.2f}ms")

    print("\nPerformance monitoring hooks registered.")

    # ============================================================
    # Example 5: State Change Monitoring
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 5: State Change Monitoring")
    print("=" * 60)

    state_agent = Agent("state_agent", model="gpt-4o-mini")

    @state_agent.on_state_change
    def on_state_change(event: Event):
        old = event.data.get("old_state", "unknown")
        new = event.data.get("new_state", "unknown")
        print(f"\n🔄 State change: {old} → {new}")

    print("\nState monitoring hook registered.")
    print(f"Current state: {state_agent.state.value}")

    # ============================================================
    # Example 6: Custom Metrics Collection
    # ============================================================
    print("\n" + "=" * 60)
    print("Example 6: Custom Metrics Collection")
    print("=" * 60)

    metrics_agent = Agent("metrics_agent", model="gpt-4o-mini")

    @metrics_agent.tool
    def test_tool(n: int) -> str:
        """Test tool that returns n items."""
        return f"Generated {n} items"

    metrics = {
        "tool_calls": 0,
        "llm_requests": 0,
        "errors": 0,
    }

    @metrics_agent.on_tool_call_start
    def count_tool(event: Event):
        metrics["tool_calls"] += 1

    @metrics_agent.on_llm_request
    def count_llm(event: Event):
        metrics["llm_requests"] += 1

    @metrics_agent.on_error
    def count_error(event: Event):
        metrics["errors"] += 1

    print("\nMetrics collection configured:")
    print(f"  Initial metrics: {metrics}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print("\nAvailable event hooks:")
    hook_names = [
        "on_start - Agent starts execution",
        "on_end - Agent finishes execution",
        "on_error - Error occurs",
        "on_llm_request - Before LLM call",
        "on_llm_response - After LLM response",
        "on_tool_call_start - Before tool execution",
        "on_tool_call_end - After tool execution",
        "on_state_change - Agent state changes",
    ]
    for hook in hook_names:
        print(f"  • {hook}")

    print("\nEvent types:")
    for et in EventType:
        print(f"  • {et.value}")


if __name__ == "__main__":
    asyncio.run(main())
