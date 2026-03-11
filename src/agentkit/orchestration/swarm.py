"""
Swarm orchestration for AgentKit.

Provides an advanced orchestration pattern where agents can dynamically
transfer context and control to other specialized agents.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.agent import Agent
from agentkit.core.tools import Tool

if TYPE_CHECKING:
    from agentkit.core.types import Message


class SwarmResult(BaseModel):
    """Result of a swarm execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    final_output: str = ""
    agent_history: list[str] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    latency_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


class TransferTarget(BaseModel):
    """Signal to transfer control to another agent."""

    target_agent: str
    context: str = ""


class Swarm:
    """
    A Swarm orchestrator for highly dynamic multi-agent systems.

    Unlike a static Team or Workflow, a Swarm allows agents to
    proactively hand off execution to other agents when they
    encounter tasks outside their specialty.

    Example:
        swarm = Swarm(name="customer_support")
        swarm.add_agent(triage_agent)
        swarm.add_agent(billing_agent)
        swarm.add_agent(tech_support_agent)

        # triage_agent can now automatically transfer the user
        # to billing or tech support based on the conversation.
        result = await swarm.arun(triage_agent, "I need a refund")
    """

    def __init__(
        self,
        name: str = "swarm",
        max_hops: int = 15,
        timeout: float = 600.0,
    ) -> None:
        self.name = name
        self.max_hops = max_hops
        self.timeout = timeout
        self._agents: dict[str, Agent] = {}
        import threading
        self._lock = threading.Lock()

    def add_agent(self, agent: Agent) -> Swarm:
        """
        Add an agent to the swarm.
        This automatically gives the agent the ability to transfer
        to any other currently registered agent in the swarm.
        """
        with self._lock:
            if agent.name in self._agents:
                return self

            # 1. Existing agents get a tool to transfer to the new agent
            for existing_name, existing_agent in self._agents.items():
                tool_name = f"transfer_to_{agent.name}"
                existing_agent.tools = [t for t in existing_agent.tools if t.name != tool_name]
                existing_agent.add_tool(self._make_transfer_tool(agent.name))

            # 2. The new agent gets tools to transfer to all existing agents
            for existing_name in self._agents.keys():
                tool_name = f"transfer_to_{existing_name}"
                agent.tools = [t for t in agent.tools if t.name != tool_name]
                agent.add_tool(self._make_transfer_tool(existing_name))

            self._agents[agent.name] = agent
        return self

    def _make_transfer_tool(self, target: str) -> Tool:
        def transfer_routine(context: str) -> TransferTarget:
            """Transfer execution to another agent."""
            return TransferTarget(target_agent=target, context=context)

        transfer_routine.__doc__ = f"Transfer the conversation to the '{target}' agent.\n\nArgs:\n    context: Information to pass to the next agent."

        return Tool(
            name=f"transfer_to_{target}",
            description=f"Transfer control to the {target} agent for specialized handling.",
            func=transfer_routine,
        )

    async def arun(
        self,
        starting_agent: str | Agent,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> SwarmResult:
        """
        Execute the swarm asynchronously.

        Args:
            starting_agent: The first agent to receive the task.
            task: The user input or task description.
            context: Optional shared context dictionary.
        """
        start_time = time.perf_counter()

        if isinstance(starting_agent, Agent):
            current_agent = starting_agent
            if current_agent.name not in self._agents:
                self.add_agent(current_agent)
        else:
            if starting_agent not in self._agents:
                return SwarmResult(
                    success=False, errors=[f"Starting agent {starting_agent} not found."]
                )
            current_agent = self._agents[starting_agent]

        messages: list[Message] = []
        agent_history: list[str] = [current_agent.name]
        errors: list[str] = []

        current_input = task
        hops = 0

        while hops < self.max_hops:
            hops += 1

            # The context dictionary can be formatted into the prompt if desired.
            prompt = current_input
            if context and hops == 1:
                prompt = f"Context: {context}\n\nTask: {current_input}"

            try:
                # Run the current agent
                result = await current_agent.arun(prompt)

                # Check messages for a TransferTarget return type from a tool
                transfer_target = None

                # We look at the actual tool results
                for tr in result.tool_results:
                    if tr.name.startswith("transfer_to_"):
                        # In our tool implementation, if the func returns a BaseModel,
                        # the stringified version or internal context might not be perfectly extracted.
                        # For a robust implementation, ToolResult should store the raw object.
                        # Here we rely on parsing the name since we injected it.
                        target_name = tr.name[12:]
                        if target_name in self._agents:
                            transfer_target = target_name
                            
                            # Extract context from raw_result if it's a TransferTarget
                            ctx = tr.content
                            if hasattr(tr, "raw_result") and isinstance(tr.raw_result, TransferTarget):
                                ctx = tr.raw_result.context

                            current_input = (
                                f"Transferred from {current_agent.name}. Context: {ctx}"
                            )
                            break

                messages.extend(result.messages)

                if transfer_target:
                    # Hop to next agent
                    current_agent = self._agents[transfer_target]
                    agent_history.append(current_agent.name)
                    continue
                else:
                    # Agent completed its work without transferring
                    return SwarmResult(
                        success=True,
                        final_output=result.content,
                        agent_history=agent_history,
                        messages=messages,
                        latency_ms=(time.perf_counter() - start_time) * 1000,
                        errors=errors,
                    )

            except Exception as e:
                errors.append(f"Agent '{current_agent.name}' failed: {e!s}")
                return SwarmResult(
                    success=False,
                    final_output=f"Failed at {current_agent.name}",
                    agent_history=agent_history,
                    messages=messages,
                    latency_ms=(time.perf_counter() - start_time) * 1000,
                    errors=errors,
                )

        errors.append(f"Max hops ({self.max_hops}) exceeded.")
        return SwarmResult(
            success=False,
            final_output="Max hops exceeded.",
            agent_history=agent_history,
            messages=messages,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            errors=errors,
        )

    def run(
        self, starting_agent: str | Agent, task: str, context: dict[str, Any] | None = None
    ) -> SwarmResult:
        """Execute the swarm synchronously."""
        return asyncio.run(self.arun(starting_agent, task, context))
