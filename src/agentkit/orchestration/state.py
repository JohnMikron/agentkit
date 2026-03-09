from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.types import AgentResult, Message


class SharedState(BaseModel):
    """
    Unified strongly-typed state context shared across all orchestrators
    (Workflow, Team, Swarm, Hierarchy).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    input: Any = None
    history: list[Message] = Field(default_factory=list)

    def get_result(self, key: str) -> AgentResult | None:
        """Fetch a specific agent/step resulting AgentResult directly."""
        return getattr(self, f"{key}_result_obj", None)
