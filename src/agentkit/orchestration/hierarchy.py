"""
Hierarchical Orchestration pattern for Agent Teams.

Implements a Manager/Worker paradigm where a Supervisor agent intelligently delegates tasks to a designated team of specialized Worker agents.
"""

import json
from enum import Enum
from typing import Dict, List, Optional, Type, Any

from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.agent import Agent
from agentkit.core.tools import Tool, ToolResult
from agentkit.core.types import Message


class TaskStatus(str, Enum):
    """Status of a managed task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DelegatedTask(BaseModel):
    """A task delegated to a worker."""
    worker_name: str
    instructions: str


class SupervisorResponse(BaseModel):
    """The structured response the supervisor returns deciding what to do next."""
    thoughts: str = Field(description="Reasoning about the current state and what to do next.")
    delegations: List[DelegatedTask] = Field(description="List of tasks to delegate to workers. Should be empty if finishing.")
    is_finished: bool = Field(description="Set to true if all work is complete.")
    final_answer: str = Field(description="The final aggregated response if is_finished is true.")


class HierarchicalTeam:
    """
    A Manager-Worker orchestration pattern.
    
    The supervisor analyzes the objective and delegates work to the correct workers. 
    It runs in a dynamic loop until the supervisor determines the overarching goal is met.
    """

    def __init__(
        self,
        supervisor: Agent,
        workers: List[Agent],
        max_iterations: int = 10,
    ) -> None:
        """
        Initialize the hierarchical team.
        
        Args:
            supervisor: The Manager agent responsible for task delegation.
            workers: The pool of specialized Worker agents.
            max_iterations: Maximum number of delegation loops.
        """
        self.supervisor = supervisor
        # We need structured output support for the supervisor
        if not hasattr(self.supervisor, "arun_structured"):
            raise ValueError("Supervisor must be a v1.3.0+ Agent supporting `arun_structured`.")
            
        self.workers = {w.name: w for w in workers}
        self.max_iterations = max_iterations
        self.history: List[Message] = []

    def _build_supervisor_prompt(self, objective: str) -> str:
        worker_profiles = "\n".join(
            [f"- **{name}**: {worker.config.system_prompt[:100]}..." for name, worker in self.workers.items()]
        )
        return f"""You are the Supervisor of a specialized team of AI Agents.

Your Objective:
{objective}

Your Available Workers:
{worker_profiles}

Analyze the history of tasks completed so far (provided in the conversation context), and determine what needs to be done next.
Delegate clear, specific tasks to the appropriate workers.
If the overarching objective has been fully achieved, synthesize the final answer and mark `is_finished` as true.
"""

    async def arun(self, objective: str) -> str:
        """
        Execute the hierarchical orchestration async.
        
        Args:
            objective: The overarching objective for the team.
            
        Returns:
            The final aggregated string response from the supervisor.
        """
        self.history.append(Message.user(f"Initial Objective: {objective}"))
        
        for iteration in range(self.max_iterations):
            prompt = self._build_supervisor_prompt(objective)
            
            # The supervisor context needs to contain the history of work
            history_text = "\n".join([f"{msg.role}: {msg.content}" for msg in self.history])
            full_prompt = f"{prompt}\n\nWork History:\n{history_text}"
            
            # Supervisor decides what to do
            plan: SupervisorResponse = await self.supervisor.arun_structured(
                prompt=full_prompt, 
                response_model=SupervisorResponse
            ) # type: ignore
            
            if plan.is_finished:
                return plan.final_answer
                
            if not plan.delegations:
                return "Supervisor unexpectedly stopped delegating without finishing."

            self.history.append(Message.assistant(f"Supervisor decided: {plan.thoughts}"))
            
            # Execute delegated tasks concurrently
            # In a real system, you might use asyncio.gather, but doing sequentially for safety here
            for task in plan.delegations:
                if task.worker_name not in self.workers:
                    worker_res = f"Cannot delegate to {task.worker_name}: Worker not found."
                else:
                    worker = self.workers[task.worker_name]
                    worker_res = await worker.arun(task.instructions)
                    
                self.history.append(Message.user(f"Worker {task.worker_name} reported:\n{worker_res}"))
                
        raise Exception(f"HierarchicalTeam reached max iterations ({self.max_iterations}) without finishing.")
