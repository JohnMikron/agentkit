"""
StateGraph workflow engine for cyclic graphs.

Provides a LangGraph-style state machine for defining workflows
that can loop, retry, and dynamically route based on state.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

logger = logging.getLogger("agentkit.stategraph")

# State type can be a dict or a BaseModel
StateType = TypeVar("StateType", bound=dict[str, Any] | BaseModel)


class NodeResult(BaseModel):
    """Result of a node execution."""

    state_update: dict[str, Any]
    output: Any = None


# A node is a callable that takes the current state and returns an update
NodeFunc = Callable[[Any], dict[str, Any] | NodeResult]
AsyncNodeFunc = Callable[[Any], Any]  # Awaitable


class StateGraph(Generic[StateType]):
    """
    A graph-based workflow engine supporting cyclic execution.

    Similar to LangGraph, it routes state between nodes.
    Features:
    - Infinite loops and retries
    - Conditional edges
    - Type-safe state (Pydantic or Dict)
    """

    # Special node names
    START = "__start__"
    END = "__end__"

    def __init__(self, state_schema: type[StateType]) -> None:
        """
        Initialize the StateGraph.

        Args:
            state_schema: The schema for the graph's state (BaseModel class or dict).
        """
        self.state_schema = state_schema
        self._nodes: dict[str, NodeFunc | AsyncNodeFunc] = {}
        self._edges: dict[str, list[tuple[Callable[[StateType], str], str]]] = {}
        self._compiled = False

    def add_node(self, name: str, node: NodeFunc | AsyncNodeFunc) -> None:
        """Add a processing node to the graph."""
        if name in [self.START, self.END]:
            raise ValueError(f"Cannot use reserved node name: {name}")
        self._nodes[name] = node

    def add_edge(self, source: str, target: str) -> None:
        """Add a guaranteed routing edge from source to target."""
        self.add_conditional_edge(source, lambda _: target)

    def add_conditional_edge(self, source: str, condition: Callable[[StateType], str]) -> None:
        """
        Add a conditional edge from source.

        The condition function receives the state and returns the name
        of the next node to execute.
        """
        if source not in self._edges:
            self._edges[source] = []

        # We store edges as a list of conditions, but typically
        # only the first one that matches is used.
        self._edges[source].append((condition, ""))

    def set_entry_point(self, node: str) -> None:
        """Set the starting node of the graph."""
        self.add_edge(self.START, node)

    def set_finish_point(self, node: str) -> None:
        """Set a node to route directly to END."""
        self.add_edge(node, self.END)

    def compile(self) -> None:
        """Compile and validate the graph."""
        if self.START not in self._edges:
            raise ValueError("No entry point set. Call set_entry_point().")

        # Verify all targets exist
        for source, _edges in self._edges.items():
            if source != self.START and source not in self._nodes:
                raise ValueError(f"Edge from unknown node: {source}")

        self._compiled = True

    def _update_state(self, current_state: StateType, update: dict[str, Any]) -> StateType:
        """Merge updates into the state."""
        if isinstance(current_state, dict):
            new_state = current_state.copy()
            new_state.update(update)
            return new_state  # type: ignore
        elif isinstance(current_state, BaseModel):
            # For Pydantic models, create a new instance with the updates
            data = current_state.model_dump()
            data.update(update)
            return self.state_schema(**data)  # type: ignore
        return current_state

    async def ainvoke(self, initial_state: StateType, recursion_limit: int = 100) -> StateType:
        """
        Execute the graph asynchronously.

        Args:
            initial_state: The starting state.
            recursion_limit: Maximum number of node transitions to prevent infinite loops.
        """
        if not self._compiled:
            self.compile()

        state = initial_state
        current_node = self.START
        iterations = 0

        while iterations < recursion_limit:
            if current_node == self.END:
                break

            iterations += 1
            logger.debug(f"StateGraph: Entering node '{current_node}'")

            # 1. Execute Node
            if current_node != self.START:
                node_func = self._nodes[current_node]

                if asyncio.iscoroutinefunction(node_func):
                    result = await node_func(state)
                else:
                    result = node_func(state)

                # Process result
                update = {}
                if isinstance(result, NodeResult):
                    update = result.state_update
                elif isinstance(result, dict):
                    update = result

                # Update state
                state = self._update_state(state, update)

            # 2. Routing
            edges = self._edges.get(current_node, [])
            if not edges:
                raise ValueError(f"No edges defined from node '{current_node}'")

            # We take the first condition and evaluate it
            # The edge definition is (condition_func, target_str)
            # For add_edge, condition_func just returns target_str
            condition_func = edges[0][0]
            next_node = condition_func(state)

            if next_node not in self._nodes and next_node != self.END:
                raise ValueError(f"Node '{current_node}' routed to unknown node '{next_node}'")

            current_node = next_node

        if iterations >= recursion_limit:
            raise RecursionError(f"StateGraph reached recursion limit of {recursion_limit}")

        return state

    def invoke(self, initial_state: StateType, recursion_limit: int = 100) -> StateType:
        """Execute the graph synchronously."""
        return asyncio.run(self.ainvoke(initial_state, recursion_limit))
