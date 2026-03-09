"""
ReflectionAgent for self-correcting outputs.

Enables an agent to critique its own work and refine it iteratively.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic import BaseModel

    from agentkit.core.agent import Agent
    from agentkit.core.types import AgentResult

logger = logging.getLogger("agentkit.reflection")


class ReflectionAgent:
    """
    An agent that performs self-reflection and refinement.

    The ReflectionAgent uses a loop of:
    1. Generation: Create an initial response.
    2. Critique: Identify weaknesses or errors in the response.
    3. Refinement: Improve the response based on the critique.
    """

    def __init__(
        self,
        agent: Agent,
        critique_agent: Agent | None = None,
        max_iterations: int = 3,
    ) -> None:
        """
        Initialize the ReflectionAgent.

        Args:
            agent: The primary agent for generation and refinement.
            critique_agent: Optional separate agent for critiquing (defaults to `agent`).
            max_iterations: Maximum number of refinement iterations.
        """
        self.agent = agent
        self.critique_agent = critique_agent or agent
        self.max_iterations = max_iterations

    async def arun(self, prompt: str, **kwargs: Any) -> AgentResult:
        """
        Run the reflection loop asynchronously.

        Returns the final refined AgentResult.
        """
        # 1. Initial Generation
        current_result = await self.agent.arun(prompt, **kwargs)

        for i in range(self.max_iterations):
            logger.info(f"Reflection iteration {i+1}/{self.max_iterations}")

            # 2. Critique
            critique_prompt = f"Critique the following response to the prompt: '{prompt}'\n\nResponse:\n{current_result.content}\n\nIdentify any errors, omissions, or areas for improvement."
            critique_result = await self.critique_agent.arun(critique_prompt, **kwargs)

            # 3. Refinement
            refine_prompt = f"Refine the following response based on the provided critique.\n\nOriginal Prompt: '{prompt}'\n\nOriginal Response:\n{current_result.content}\n\nCritique:\n{critique_result.content}\n\nProvide the final improved version."
            current_result = await self.agent.arun(refine_prompt, **kwargs)

        return current_result

    def run(self, prompt: str, **kwargs: Any) -> AgentResult:
        """Run the reflection loop synchronously."""
        import asyncio
        return asyncio.run(self.arun(prompt, **kwargs))

    async def arun_structured(
        self,
        prompt: str,
        response_model: type[BaseModel],
        **kwargs: Any
    ) -> AgentResult:
        """
        Run the reflection loop and return a structured output.

        Refinement is performed on the text, and the final result is
        extracted into the provided Pydantic model.
        """
        # Normal reflection loop
        final_result = await self.arun(prompt, **kwargs)

        # Final extraction
        return await self.agent.arun_structured(
            f"Extract the information from this text into the required format:\n\n{final_result.content}",
            response_model,
            **kwargs
        )
