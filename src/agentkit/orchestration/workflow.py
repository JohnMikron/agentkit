"""
Workflow engine for complex agent pipelines.

Provides a state machine for defining multi-step agent workflows
with conditional transitions.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.agent import Agent
from agentkit.core.types import AgentResult


class StepStatus(str, Enum):
    """Status of a workflow step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TransitionType(str, Enum):
    """Type of transition between steps."""

    ALWAYS = "always"  # Always transition
    ON_SUCCESS = "on_success"  # Only on success
    ON_FAILURE = "on_failure"  # Only on failure
    CONDITIONAL = "conditional"  # Based on condition


@dataclass
class Step:
    """
    A step in a workflow.

    Attributes:
        name: Unique step name
        agent: Agent to execute
        prompt_template: Jinja2 template for prompt
        timeout: Step timeout in seconds
        retry_count: Number of retries on failure
        on_enter: Callback before step starts
        on_exit: Callback after step completes
    """

    name: str
    agent: Agent
    prompt_template: str = "{{ input }}"
    timeout: Optional[float] = None
    retry_count: int = 0
    on_enter: Optional[Callable[[Dict[str, Any]], None]] = None
    on_exit: Optional[Callable[[AgentResult], None]] = None

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: Optional[AgentResult] = None
    attempts: int = 0


@dataclass
class Transition:
    """
    A transition between workflow steps.

    Attributes:
        from_step: Source step name
        to_step: Target step name
        type: Transition type
        condition: Optional condition function
    """

    from_step: str
    to_step: str
    type: TransitionType = TransitionType.ALWAYS
    condition: Optional[Callable[[AgentResult], bool]] = None


class WorkflowState(BaseModel):
    """State of a workflow execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_step: Optional[str] = None
    completed_steps: List[str] = Field(default_factory=list)
    step_results: Dict[str, AgentResult] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    iterations: int = 0


class WorkflowResult(BaseModel):
    """Result of workflow execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    final_output: str = ""
    state: WorkflowState = Field(default_factory=WorkflowState)
    latency_ms: float = 0.0
    errors: List[str] = Field(default_factory=list)


class Workflow:
    """
    A workflow engine for complex agent pipelines.

    Define steps and transitions to create sophisticated
    multi-agent workflows.

    Example:
        workflow = Workflow("research_workflow")

        workflow.add_step("research", researcher, "Research: {{ topic }}")
        workflow.add_step("write", writer, "Write about: {{ research_result }}")
        workflow.add_step("review", reviewer, "Review: {{ article }}")

        workflow.add_transition("research", "write", TransitionType.ON_SUCCESS)
        workflow.add_transition("write", "review", TransitionType.ON_SUCCESS)

        result = await workflow.arun({"topic": "AI agents"})
    """

    def __init__(
        self,
        name: str = "workflow",
        max_iterations: int = 100,
        timeout: float = 600.0,
    ) -> None:
        self.name = name
        self.max_iterations = max_iterations
        self.timeout = timeout

        self._steps: Dict[str, Step] = {}
        self._transitions: Dict[str, List[Transition]] = {}
        self._entry_step: Optional[str] = None

    def add_step(
        self,
        name: str,
        agent: Agent,
        prompt_template: str = "{{ input }}",
        timeout: Optional[float] = None,
        retry_count: int = 0,
        on_enter: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
    ) -> "Workflow":
        """
        Add a step to the workflow.

        Returns self for chaining.
        """
        self._steps[name] = Step(
            name=name,
            agent=agent,
            prompt_template=prompt_template,
            timeout=timeout,
            retry_count=retry_count,
            on_enter=on_enter,
            on_exit=on_exit,
        )

        if self._entry_step is None:
            self._entry_step = name

        return self

    def add_transition(
        self,
        from_step: str,
        to_step: str,
        type: TransitionType = TransitionType.ON_SUCCESS,
        condition: Optional[Callable[[AgentResult], bool]] = None,
    ) -> "Workflow":
        """
        Add a transition between steps.

        Returns self for chaining.
        """
        if from_step not in self._transitions:
            self._transitions[from_step] = []

        self._transitions[from_step].append(Transition(
            from_step=from_step,
            to_step=to_step,
            type=type,
            condition=condition,
        ))

        return self

    def set_entry(self, step_name: str) -> "Workflow":
        """Set the entry point for the workflow."""
        if step_name not in self._steps:
            raise ValueError(f"Step '{step_name}' not found")
        self._entry_step = step_name
        return self

    def _render_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Render a Jinja2 template with context."""
        from jinja2 import Template

        tmpl = Template(template)
        return tmpl.render(**context)

    def _get_next_step(self, current: str, result: AgentResult) -> Optional[str]:
        """Determine the next step based on transitions."""
        transitions = self._transitions.get(current, [])

        for transition in transitions:
            # Check transition type
            if transition.type == TransitionType.ALWAYS:
                pass
            elif transition.type == TransitionType.ON_SUCCESS and not result.success:
                continue
            elif transition.type == TransitionType.ON_FAILURE and result.success:
                continue
            elif transition.type == TransitionType.CONDITIONAL:
                if transition.condition and not transition.condition(result):
                    continue

            return transition.to_step

        return None  # No valid transition - end workflow

    async def arun(
        self,
        input: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """
        Execute the workflow asynchronously.

        Args:
            input: Input data for the workflow
            context: Additional context variables

        Returns:
            WorkflowResult with final output and state
        """
        start_time = time.perf_counter()

        if not self._entry_step:
            return WorkflowResult(
                success=False,
                errors=["No entry step defined"],
            )

        # Initialize state
        state = WorkflowState(
            current_step=self._entry_step,
            context=context or {},
        )

        if isinstance(input, str):
            state.context["input"] = input
        else:
            state.context.update(input)

        errors: List[str] = []

        try:
            while state.current_step and state.iterations < self.max_iterations:
                state.iterations += 1
                step_name = state.current_step

                if step_name not in self._steps:
                    errors.append(f"Step '{step_name}' not found")
                    break

                step = self._steps[step_name]
                step.status = StepStatus.RUNNING

                # Call on_enter callback
                if step.on_enter:
                    try:
                        step.on_enter(state.context)
                    except Exception as e:
                        errors.append(f"on_enter callback failed: {e}")

                # Render prompt
                try:
                    prompt = self._render_prompt(step.prompt_template, state.context)
                except Exception as e:
                    errors.append(f"Prompt rendering failed: {e}")
                    step.status = StepStatus.FAILED
                    break

                # Execute agent with retries
                result = None
                for attempt in range(step.retry_count + 1):
                    step.attempts = attempt + 1

                    try:
                        result = await step.agent.arun(prompt)
                        if result.success:
                            break
                    except Exception as e:
                        errors.append(f"Step '{step_name}' attempt {attempt + 1} failed: {e}")
                        if attempt < step.retry_count:
                            await asyncio.sleep(1.0)  # Brief delay before retry

                if result is None:
                    step.status = StepStatus.FAILED
                    step.result = AgentResult(success=False, error="All attempts failed")
                    break

                step.result = result
                step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

                # Store result in context
                state.step_results[step_name] = result
                state.context[f"{step_name}_result"] = result.content

                # Call on_exit callback
                if step.on_exit:
                    try:
                        step.on_exit(result)
                    except Exception as e:
                        errors.append(f"on_exit callback failed: {e}")

                state.completed_steps.append(step_name)

                # Find next step
                next_step = self._get_next_step(step_name, result)
                state.current_step = next_step

                if not next_step:
                    # Workflow complete
                    break

        except Exception as e:
            errors.append(f"Workflow error: {str(e)}")

        # Get final output from last completed step
        final_output = ""
        if state.completed_steps:
            last_step = state.completed_steps[-1]
            if last_step in state.step_results:
                final_output = state.step_results[last_step].content

        return WorkflowResult(
            success=len(errors) == 0,
            final_output=final_output,
            state=state,
            latency_ms=(time.perf_counter() - start_time) * 1000,
            errors=errors,
        )

    def run(
        self,
        input: Union[str, Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
    ) -> WorkflowResult:
        """Execute the workflow synchronously."""
        return asyncio.run(self.arun(input, context))

    def visualize(self) -> str:
        """Generate a Mermaid diagram of the workflow."""
        lines = ["graph TD"]

        # Add steps
        for name, step in self._steps.items():
            lines.append(f"    {name}[{name}]")

        # Add transitions
        for from_step, transitions in self._transitions.items():
            for t in transitions:
                label = t.type.value
                lines.append(f"    {from_step} -->|{label}| {t.to_step}")

        # Mark entry point
        if self._entry_step:
            lines.insert(1, f"    START((start)) --> {self._entry_step}")

        return "\n".join(lines)
