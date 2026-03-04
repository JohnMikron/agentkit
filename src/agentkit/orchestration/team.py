"""
Team orchestration for multi-agent systems.

This module provides the Team class for coordinating multiple agents
working together on complex tasks.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.agent import Agent
from agentkit.core.types import AgentResult, Message


class TeamRole(str, Enum):
    """Role of an agent in a team."""

    LEADER = "leader"
    WORKER = "worker"
    SPECIALIST = "specialist"
    REVIEWER = "reviewer"


class TeamStrategy(str, Enum):
    """Strategy for team execution."""

    SEQUENTIAL = "sequential"  # Run agents one after another
    PARALLEL = "parallel"  # Run agents simultaneously
    HIERARCHICAL = "hierarchical"  # Leader distributes work
    ROUND_ROBIN = "round_robin"  # Each agent handles part of task


@dataclass
class AgentConfig:
    """Configuration for a team member agent."""

    agent: Agent
    role: TeamRole = TeamRole.WORKER
    weight: float = 1.0  # For voting/weighted decisions
    max_tasks: Optional[int] = None  # Max concurrent tasks


class TeamConfig(BaseModel):
    """Configuration for a Team."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="team", min_length=1)
    strategy: TeamStrategy = Field(default=TeamStrategy.SEQUENTIAL)
    max_parallel: int = Field(default=5, ge=1)
    timeout: float = Field(default=300.0, ge=1.0)
    retry_failed: bool = Field(default=True)
    aggregate_results: bool = Field(default=True)


class TeamResult(BaseModel):
    """Result of team execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    results: Dict[str, AgentResult] = Field(default_factory=dict)
    final_output: str = ""
    total_iterations: int = 0
    latency_ms: float = 0.0
    errors: List[str] = Field(default_factory=list)


class Team:
    """
    A team of agents working together.

    Coordinates multiple agents to solve complex tasks using
    various strategies.

    Example:
        team = Team("research_team")

        team.add_agent(researcher, role=TeamRole.WORKER)
        team.add_agent(writer, role=TeamRole.WORKER)
        team.add_agent(reviewer, role=TeamRole.REVIEWER)

        result = await team.arun("Research and write about AI")
    """

    def __init__(
        self,
        name: str = "team",
        strategy: TeamStrategy = TeamStrategy.SEQUENTIAL,
        config: Optional[TeamConfig] = None,
    ) -> None:
        self.config = config or TeamConfig(name=name, strategy=strategy)
        self._agents: Dict[str, AgentConfig] = {}
        self._leader: Optional[Agent] = None

    def add_agent(
        self,
        agent: Agent,
        role: TeamRole = TeamRole.WORKER,
        weight: float = 1.0,
        max_tasks: Optional[int] = None,
    ) -> "Team":
        """
        Add an agent to the team.

        Returns self for chaining.
        """
        self._agents[agent.name] = AgentConfig(
            agent=agent,
            role=role,
            weight=weight,
            max_tasks=max_tasks,
        )

        if role == TeamRole.LEADER:
            self._leader = agent

        return self

    def remove_agent(self, name: str) -> Optional[Agent]:
        """Remove an agent from the team."""
        if name in self._agents:
            config = self._agents.pop(name)
            if config.role == TeamRole.LEADER:
                self._leader = None
            return config.agent
        return None

    def get_agents(self, role: Optional[TeamRole] = None) -> List[Agent]:
        """Get agents, optionally filtered by role."""
        agents = [cfg.agent for cfg in self._agents.values()]
        if role:
            agents = [a for cfg in self._agents.values() if cfg.role == role for a in [cfg.agent]]
        return agents

    async def arun(self, task: str, **kwargs: Any) -> TeamResult:
        """
        Run the team on a task.

        Args:
            task: The task to execute
            **kwargs: Additional parameters

        Returns:
            TeamResult with all agent results
        """
        start_time = time.perf_counter()

        if not self._agents:
            return TeamResult(
                success=False,
                errors=["No agents in team"],
            )

        strategy = self.config.strategy

        if strategy == TeamStrategy.SEQUENTIAL:
            result = await self._run_sequential(task, **kwargs)
        elif strategy == TeamStrategy.PARALLEL:
            result = await self._run_parallel(task, **kwargs)
        elif strategy == TeamStrategy.HIERARCHICAL:
            result = await self._run_hierarchical(task, **kwargs)
        elif strategy == TeamStrategy.ROUND_ROBIN:
            result = await self._run_round_robin(task, **kwargs)
        else:
            result = await self._run_sequential(task, **kwargs)

        result.latency_ms = (time.perf_counter() - start_time) * 1000
        return result

    def run(self, task: str, **kwargs: Any) -> TeamResult:
        """Run the team synchronously."""
        return asyncio.run(self.arun(task, **kwargs))

    async def _run_sequential(self, task: str, **kwargs: Any) -> TeamResult:
        """Run agents sequentially, passing output to next agent."""
        results: Dict[str, AgentResult] = {}
        current_input = task
        total_iterations = 0
        errors: List[str] = []

        workers = [cfg for cfg in self._agents.values() if cfg.role != TeamRole.REVIEWER]

        for i, cfg in enumerate(workers):
            agent = cfg.agent

            # Prepare input
            if i > 0 and self.config.aggregate_results:
                input_text = f"Previous work: {current_input}\n\nContinue the task."
            else:
                input_text = current_input

            try:
                result = await agent.arun(input_text, **kwargs)
                results[agent.name] = result
                total_iterations += result.iterations

                if result.success:
                    current_input = result.content
                elif not self.config.retry_failed:
                    errors.append(f"Agent {agent.name} failed: {result.error}")

            except Exception as e:
                errors.append(f"Agent {agent.name} error: {str(e)}")
                if not self.config.retry_failed:
                    break

        # Run reviewers if any
        reviewers = self.get_agents(TeamRole.REVIEWER)
        if reviewers and current_input:
            for reviewer in reviewers:
                try:
                    review_task = f"Review and improve:\n\n{current_input}"
                    result = await reviewer.arun(review_task, **kwargs)
                    results[reviewer.name] = result
                    if result.success:
                        current_input = result.content
                except Exception as e:
                    errors.append(f"Reviewer {reviewer.name} error: {str(e)}")

        return TeamResult(
            success=len(errors) == 0,
            results=results,
            final_output=current_input,
            total_iterations=total_iterations,
            errors=errors,
        )

    async def _run_parallel(self, task: str, **kwargs: Any) -> TeamResult:
        """Run agents in parallel and aggregate results."""
        results: Dict[str, AgentResult] = {}
        total_iterations = 0
        errors: List[str] = []

        # Create tasks for all agents
        async def run_agent(cfg: AgentConfig) -> tuple[str, AgentResult]:
            try:
                result = await cfg.agent.arun(task, **kwargs)
                return cfg.agent.name, result
            except Exception as e:
                return cfg.agent.name, AgentResult(
                    success=False,
                    error=str(e),
                )

        # Run in parallel with semaphore
        semaphore = asyncio.Semaphore(self.config.max_parallel)

        async def bounded_run(cfg: AgentConfig):
            async with semaphore:
                return await run_agent(cfg)

        tasks = [bounded_run(cfg) for cfg in self._agents.values()]
        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for item in completed:
            if isinstance(item, Exception):
                errors.append(str(item))
            else:
                name, result = item
                results[name] = result
                total_iterations += result.iterations

        # Aggregate outputs
        outputs = [r.content for r in results.values() if r.success]
        final_output = "\n\n---\n\n".join(outputs) if outputs else ""

        return TeamResult(
            success=len(errors) == 0,
            results=results,
            final_output=final_output,
            total_iterations=total_iterations,
            errors=errors,
        )

    async def _run_hierarchical(self, task: str, **kwargs: Any) -> TeamResult:
        """Run with leader distributing work to workers."""
        if not self._leader:
            # Fall back to sequential if no leader
            return await self._run_sequential(task, **kwargs)

        results: Dict[str, AgentResult] = {}
        total_iterations = 0
        errors: List[str] = []

        # Leader analyzes task and creates subtasks
        leader_prompt = f"""Analyze this task and break it into subtasks for your team.

Task: {task}

Available workers: {', '.join(a.name for a in self.get_agents(TeamRole.WORKER))}

Create a JSON list of subtasks with assigned workers:
{{"subtasks": [{{"worker": "name", "task": "description"}}]}}"""

        leader_result = await self._leader.arun(leader_prompt, **kwargs)
        results[self._leader.name] = leader_result
        total_iterations += leader_result.iterations

        if not leader_result.success:
            return TeamResult(
                success=False,
                results=results,
                errors=["Leader failed to analyze task"],
            )

        # Parse subtasks
        import json

        try:
            # Extract JSON from response
            content = leader_result.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                subtasks_data = json.loads(content[start:end])
                subtasks = subtasks_data.get("subtasks", [])
            else:
                subtasks = []
        except (json.JSONDecodeError, KeyError):
            subtasks = []

        # Distribute subtasks to workers
        workers = {a.name: a for a in self.get_agents(TeamRole.WORKER)}

        for subtask in subtasks:
            worker_name = subtask.get("worker", "")
            subtask_desc = subtask.get("task", "")

            if worker_name in workers:
                try:
                    result = await workers[worker_name].arun(subtask_desc, **kwargs)
                    results[worker_name] = result
                    total_iterations += result.iterations
                except Exception as e:
                    errors.append(f"Worker {worker_name} error: {str(e)}")

        # Leader aggregates results
        worker_outputs = {
            name: r.content for name, r in results.items()
            if r.success and name != self._leader.name
        }

        if worker_outputs:
            aggregate_prompt = f"""Combine these worker outputs into a final result:

{json.dumps(worker_outputs, indent=2)}

Original task: {task}"""

            final_result = await self._leader.arun(aggregate_prompt, **kwargs)
            results[f"{self._leader.name}_final"] = final_result

            return TeamResult(
                success=True,
                results=results,
                final_output=final_result.content,
                total_iterations=total_iterations,
                errors=errors,
            )

        return TeamResult(
            success=len(errors) == 0,
            results=results,
            final_output=leader_result.content,
            total_iterations=total_iterations,
            errors=errors,
        )

    async def _run_round_robin(self, task: str, **kwargs: Any) -> TeamResult:
        """Run agents in round-robin fashion."""
        results: Dict[str, AgentResult] = {}
        current_input = task
        total_iterations = 0
        errors: List[str] = []

        workers = [cfg.agent for cfg in self._agents.values() if cfg.role == TeamRole.WORKER]

        if not workers:
            workers = [cfg.agent for cfg in self._agents.values()]

        # Each agent contributes to the task
        for i, agent in enumerate(workers):
            prompt = f"""Continue this task. You are agent {i + 1} of {len(workers)}.

Current state:
{current_input}

Make progress and pass to the next agent."""

            try:
                result = await agent.arun(prompt, **kwargs)
                results[agent.name] = result
                total_iterations += result.iterations

                if result.success:
                    current_input = result.content
            except Exception as e:
                errors.append(f"Agent {agent.name} error: {str(e)}")

        return TeamResult(
            success=len(errors) == 0,
            results=results,
            final_output=current_input,
            total_iterations=total_iterations,
            errors=errors,
        )

    def __repr__(self) -> str:
        return f"Team(name={self.config.name!r}, agents={len(self._agents)}, strategy={self.config.strategy.value})"
