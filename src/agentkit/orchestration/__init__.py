"""
Multi-agent orchestration for AgentKit.

Provides:
- Team: Group of agents working together
- Workflow: State machine for complex agent pipelines
- Router: Route requests to specialized agents
"""

from agentkit.orchestration.team import Team, TeamConfig, TeamRole, TeamStrategy, TeamResult
from agentkit.orchestration.workflow import Workflow, WorkflowState, WorkflowResult, StepStatus, TransitionType
from agentkit.orchestration.swarm import Swarm, SwarmResult

__all__ = [
    # Team
    "Team",
    "TeamConfig",
    "TeamRole",
    "TeamStrategy",
    "TeamResult",
    
    # Workflow
    "Workflow",
    "WorkflowState",
    "WorkflowResult",
    "StepStatus",
    "TransitionType",

    # Swarm
    "Swarm",
    "SwarmResult",
]
