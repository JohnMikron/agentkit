import asyncio

import pytest

from agentkit.orchestration.stategraph import StateGraph


def test_stategraph_basic():
    # A simple sequential pipeline A -> B -> END
    graph = StateGraph(dict)

    def node_a(state):
        return {"a_done": True, "count": state.get("count", 0) + 1}

    def node_b(state):
        return {"b_done": True, "count": state["count"] + 1}

    graph.add_node("A", node_a)
    graph.add_node("B", node_b)

    graph.set_entry_point("A")
    graph.add_edge("A", "B")
    graph.set_finish_point("B")

    result = graph.invoke({"initial": True})

    assert result["initial"] is True
    assert result["a_done"] is True
    assert result["b_done"] is True
    assert result["count"] == 2

def test_stategraph_conditional_loop():
    # A loop A -> B -> A until count == 3
    graph = StateGraph(dict)

    def node_process(state):
        return {"count": state.get("count", 0) + 1}

    def check_done(state) -> str:
        if state["count"] >= 3:
            return StateGraph.END
        return "Process"

    graph.add_node("Process", node_process)
    graph.set_entry_point("Process")

    # Conditional edge pointing either back to self or END
    graph.add_conditional_edge("Process", check_done)

    result = graph.invoke({"count": 0})

    assert result["count"] == 3

def test_stategraph_recursion_limit():
    # Infinite loop
    graph = StateGraph(dict)

    def infinite(state):
        return {"count": state.get("count", 0) + 1}

    graph.add_node("Loop", infinite)
    graph.set_entry_point("Loop")
    graph.add_edge("Loop", "Loop")

    with pytest.raises(RecursionError):
        graph.invoke({"count": 0}, recursion_limit=10)

@pytest.mark.asyncio
async def test_stategraph_async():
    graph = StateGraph(dict)

    async def async_worker(state):
        await asyncio.sleep(0.01)
        return {"async_done": True}

    graph.add_node("AsyncWorker", async_worker)
    graph.set_entry_point("AsyncWorker")
    graph.set_finish_point("AsyncWorker")

    result = await graph.ainvoke({})
    assert result["async_done"] is True
