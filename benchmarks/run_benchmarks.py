"""
Benchmarks for AgentKit.

Measures instantiation latency, basic execution overhead, and orchestrator scale.
Run this against competitor frameworks like LangGraph and CrewAI to see the difference.
"""

import asyncio
import time
from agentkit import Agent
from agentkit.orchestration.stategraph import StateGraph

def benchmark_instantiation():
    """Measure how fast Agent objects are created."""
    start = time.perf_counter()
    agents = [Agent(f"Agent{i}") for i in range(1000)]
    end = time.perf_counter()
    total_ms = (end - start) * 1000
    print(f"Agent Instantiation (1000 objects): {total_ms:.2f} ms")


def benchmark_stategraph_overhead():
    """Measure raw cyclic routing overhead with mock nodes."""
    graph = StateGraph(dict)
    
    # Setup mock nodes that do 0 work, just return state
    def node_a(state): return {"count": state.get("count", 0) + 1}
    def node_b(state): return {"count": state["count"] + 1}
    
    graph.add_node("A", node_a)
    graph.add_node("B", node_b)
    graph.set_entry_point("A")
    graph.add_edge("A", "B")
    
    # Loop B -> A until 100 iterations
    def condition(state):
        if state["count"] >= 1000:
            return StateGraph.END
        return "A"
        
    graph.add_conditional_edge("B", condition)
    
    start = time.perf_counter()
    res = graph.invoke({"count": 0}, recursion_limit=5000)
    end = time.perf_counter()
    total_ms = (end - start) * 1000
    print(f"StateGraph Routing Overhead (1000 cyclic transitions): {total_ms:.2f} ms")


if __name__ == "__main__":
    print("=== AgentKit Performance Benchmarks ===")
    benchmark_instantiation()
    benchmark_stategraph_overhead()
