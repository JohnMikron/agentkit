"""
Router for directing requests to specialized agents.

Provides intelligent routing based on:
- Keyword matching
- LLM-based classification
- Custom routing functions
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.types import AgentResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentkit.core.agent import Agent


class RouteStrategy(str, Enum):
    """Strategy for routing decisions."""

    KEYWORD = "keyword"  # Match keywords
    REGEX = "regex"  # Match regex patterns
    LLM = "llm"  # Use LLM to classify
    CUSTOM = "custom"  # Custom routing function
    ALL = "all"  # Route to all agents


@dataclass
class Route:
    """
    A route definition for the router.

    Attributes:
        name: Route name
        agent: Agent to route to
        keywords: Keywords to match (for KEYWORD strategy)
        patterns: Regex patterns (for REGEX strategy)
        priority: Route priority (higher = more important)
        condition: Custom condition function
    """

    name: str
    agent: Agent
    keywords: list[str] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    priority: int = 0
    condition: Callable[[str], bool] | None = None

    # Compiled patterns
    _compiled_patterns: list[re.Pattern[Any]] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        """Compile regex patterns."""
        for pattern in self.patterns:
            with contextlib.suppress(re.error):
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))

    def matches(self, input: str) -> bool:
        """Check if input matches this route."""
        # Check keywords
        input_lower = input.lower()
        for keyword in self.keywords:
            if keyword.lower() in input_lower:
                return True

        # Check patterns
        for compiled in self._compiled_patterns:
            if compiled.search(input):
                return True

        # Check custom condition
        if self.condition:
            try:
                return self.condition(input)
            except Exception:
                pass

        return False


class RouterResult(BaseModel):
    """Result of routing execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    routed_to: list[str] = Field(default_factory=list)
    results: dict[str, AgentResult] = Field(default_factory=dict)
    final_output: str = ""
    latency_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class Router:
    """
    Router for directing requests to specialized agents.

    Intelligently routes requests to the most appropriate agent
    based on the routing strategy.

    Example:
        router = Router()

        router.add_route("code", coder, keywords=["code", "programming", "function"])
        router.add_route("math", mathematician, keywords=["calculate", "math", "equation"])
        router.add_route("general", assistant)  # Default route

        result = await router.arun("Write a Python function to sort a list")
        # Routes to coder agent
    """

    def __init__(
        self,
        name: str = "router",
        strategy: RouteStrategy = RouteStrategy.KEYWORD,
        default_agent: Agent | None = None,
        aggregate_results: bool = False,
    ) -> None:
        self.name = name
        self.strategy = strategy
        self.default_agent = default_agent
        self.aggregate_results = aggregate_results

        self._routes: dict[str, Route] = {}
        self._classifier: Agent | None = None

    def add_route(
        self,
        name: str,
        agent: Agent,
        keywords: list[str] | None = None,
        patterns: list[str] | None = None,
        priority: int = 0,
        condition: Callable[[str], bool] | None = None,
    ) -> Router:
        """
        Add a route to the router.

        Returns self for chaining.
        """
        self._routes[name] = Route(
            name=name,
            agent=agent,
            keywords=keywords or [],
            patterns=patterns or [],
            priority=priority,
            condition=condition,
        )
        return self

    def set_classifier(self, agent: Agent) -> Router:
        """
        Set an LLM agent for classification.

        Used when strategy is LLM.
        """
        self._classifier = agent
        return self

    def _route_keyword(self, input: str) -> list[Route]:
        """Route based on keyword matching."""
        matches = []

        for route in self._routes.values():
            if route.matches(input):
                matches.append(route)

        # Sort by priority
        matches.sort(key=lambda r: r.priority, reverse=True)

        return matches

    async def _route_llm(self, input: str) -> list[Route]:
        """Route using LLM classification."""
        if not self._classifier:
            return self._route_keyword(input)

        route_names = list(self._routes.keys())

        prompt = f"""Classify this request into one of these categories: {", ".join(route_names)}

Request: {input}

Respond with ONLY the category name, nothing else."""

        try:
            result = await self._classifier.arun(prompt)
            category = result.content.strip().lower()

            for route in self._routes.values():
                if route.name.lower() == category:
                    return [route]

        except Exception:
            pass

        return []

    def _determine_routes(self, input: str) -> list[Route]:
        """Determine which routes to use based on strategy."""
        if self.strategy == RouteStrategy.ALL:
            return list(self._routes.values())

        if self.strategy == RouteStrategy.KEYWORD or self.strategy == RouteStrategy.REGEX:
            return self._route_keyword(input)

        if self.strategy == RouteStrategy.CUSTOM:
            # Find first matching custom route
            for route in self._routes.values():
                if route.condition and route.condition(input):
                    return [route]

        return []

    async def arun(self, input: str, **kwargs: Any) -> RouterResult:
        """
        Route input and execute agents.

        Args:
            input: The input to route
            **kwargs: Additional parameters for agents

        Returns:
            RouterResult with execution results
        """
        start_time = time.perf_counter()

        results: dict[str, AgentResult] = {}
        routed_to: list[str] = []
        errors: list[str] = []

        # Determine routes
        if self.strategy == RouteStrategy.LLM:
            routes = await self._route_llm(input)
        else:
            routes = self._determine_routes(input)

        # Fall back to default if no routes match
        if not routes and self.default_agent:
            routes = [Route(name="default", agent=self.default_agent)]

        if not routes:
            return RouterResult(
                success=False,
                errors=["No matching route found"],
                latency_ms=(time.perf_counter() - start_time) * 1000,
            )

        # Execute matched routes
        async def execute_route(route: Route) -> tuple[str, AgentResult]:
            try:
                result = await route.agent.arun(input, **kwargs)
                return route.name, result
            except Exception as e:
                return route.name, AgentResult(success=False, error=str(e))

        # Run routes in parallel
        tasks = [execute_route(route) for route in routes]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, BaseException):
                errors.append(str(item))
            else:
                name, result = item
                results[name] = result
                routed_to.append(name)

        # Aggregate outputs if configured
        final_output = ""
        if self.aggregate_results and results:
            outputs = [r.content for r in results.values() if r.success]
            final_output = "\n\n".join(outputs)
        elif results:
            # Return first successful result
            for result in results.values():
                if result.success:
                    final_output = result.content
                    break

        return RouterResult(
            success=len(errors) == 0,
            routed_to=routed_to,
            results=results,
            final_output=final_output,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            errors=errors,
        )

    def run(self, input: str, **kwargs: Any) -> RouterResult:
        """Route and execute synchronously."""
        return asyncio.run(self.arun(input, **kwargs))

    def __repr__(self) -> str:
        return f"Router(name={self.name!r}, routes={len(self._routes)}, strategy={self.strategy.value})"
