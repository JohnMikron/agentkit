"""
Multi-agent orchestration for AgentKit.

Provides:
- Team: Group of agents working together
- Workflow: State machine for complex agent pipelines
- Router: Route requests to specialized agents
"""

from agentkit.orchestration.swarm import Swarm, SwarmResult
from agentkit.orchestration.team import Team, TeamConfig, TeamResult, TeamRole, TeamStrategy
from agentkit.orchestration.workflow import (
    StepStatus,
    TransitionType,
    Workflow,
    WorkflowResult,
    WorkflowState,
)

__all__ = [
    "StepStatus",
    # Swarm
    "Swarm",
    "SwarmResult",
    # Team
    "Team",
    "TeamConfig",
    "TeamResult",
    "TeamRole",
    "TeamStrategy",
    "TransitionType",
    # Workflow
    "Workflow",
    "WorkflowResult",
    "WorkflowState",
]
