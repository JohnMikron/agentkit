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
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.types import AgentResult, AgentState

if TYPE_CHECKING:
    from collections.abc import Callable

    from agentkit.core.agent import Agent


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
    timeout: float | None = None
    retry_count: int = 0
    on_enter: Callable[[dict[str, Any]], None] | None = None
    on_exit: Callable[[AgentResult], None] | None = None

    # Runtime state
    status: StepStatus = StepStatus.PENDING
    result: AgentResult | None = None
    attempts: int = 0


@dataclass
class ParallelStep(Step):
    """
    A workflow step that executes multiple agents concurrently.
    """
    agents: dict[str, Agent] = field(default_factory=dict)
    prompt_templates: dict[str, str] = field(default_factory=dict)
    results: dict[str, AgentResult] = field(default_factory=dict)

    # Dummy initializers since Step fields must be populated
    agent: Agent | None = None
    prompt_template: str = ""


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
    condition: Callable[[AgentResult], bool] | None = None


class WorkflowContext(BaseModel):
    """Strongly typed context for workflow execution."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    input: Any = None

    def get_step_result(self, step_name: str) -> AgentResult | None:
        """Fetch a specific step's resulting AgentResult directly."""
        return getattr(self, f"{step_name}_result_obj", None)


class WorkflowState(BaseModel):
    """State of a workflow execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_step: str | None = None
    completed_steps: list[str] = Field(default_factory=list)
    step_results: dict[str, AgentResult] = Field(default_factory=dict)
    context: WorkflowContext = Field(default_factory=WorkflowContext)
    iterations: int = 0


class WorkflowResult(BaseModel):
    """Result of workflow execution."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    success: bool = True
    final_output: str = ""
    state: WorkflowState = Field(default_factory=WorkflowState)
    latency_ms: float = 0.0
    errors: list[str] = Field(default_factory=list)


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

        self._steps: dict[str, Step] = {}
        self._transitions: dict[str, list[Transition]] = {}
        self._entry_step: str | None = None

    def add_step(
        self,
        name: str,
        agent: Agent,
        prompt_template: str = "{{ input }}",
        timeout: float | None = None,
        retry_count: int = 0,
        on_enter: Callable[[WorkflowContext], None] | None = None,
        on_exit: Callable[[AgentResult], None] | None = None,
    ) -> Workflow:
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

    def add_parallel_step(
        self,
        name: str,
        agents: dict[str, Agent],
        prompt_templates: dict[str, str],
        timeout: float | None = None,
        retry_count: int = 0,
        on_enter: Callable | None = None,
        on_exit: Callable | None = None,
    ) -> Workflow:
        """
        Add a parallel execution step to the workflow.

        Args:
            name: Name of the step
            agents: Mapping of sub-task names to Agents
            prompt_templates: Mapping of sub-task names to prompt templates
        """
        self._steps[name] = ParallelStep(
            name=name,
            agents=agents,
            prompt_templates=prompt_templates,
            timeout=timeout,
            retry_count=retry_count,
            on_enter=on_enter,
            on_exit=on_exit,
            # Dummy fields required by inheritance
            agent=None,
            prompt_template="",
        )

        if self._entry_step is None:
            self._entry_step = name

        return self

    def add_transition(
        self,
        from_step: str,
        to_step: str,
        type: TransitionType = TransitionType.ON_SUCCESS,
        condition: Callable[[AgentResult], bool] | None = None,
    ) -> Workflow:
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

    def set_entry(self, step_name: str) -> Workflow:
        """Set the entry point for the workflow."""
        if step_name not in self._steps:
            raise ValueError(f"Step '{step_name}' not found")
        self._entry_step = step_name
        return self

    def _render_prompt(self, template: str, context: WorkflowContext) -> str:
        """Render a Jinja2 template with context."""
        from jinja2 import Template

        tmpl = Template(template)
        return tmpl.render(**context.model_dump())

    def _get_next_step(self, current: str, result: AgentResult) -> str | None:
        """Determine the next step based on transitions."""
        transitions = self._transitions.get(current, [])

        for transition in transitions:
            # Check transition type
            if transition.type == TransitionType.ALWAYS:
                pass
            elif (
                (transition.type == TransitionType.ON_SUCCESS and not result.success)
                or (transition.type == TransitionType.ON_FAILURE and result.success)
                or (transition.type == TransitionType.CONDITIONAL and transition.condition and not transition.condition(result))
            ):
                continue

            return transition.to_step

        return None  # No valid transition - end workflow

    async def arun(
        self,
        input: str | dict[str, Any],
        context: dict[str, Any] | None = None,
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
        initial_context = WorkflowContext()
        if context:
            for k, v in context.items():
                setattr(initial_context, k, v)

        state = WorkflowState(
            current_step=self._entry_step,
            context=initial_context,
        )

        if isinstance(input, str):
            state.context.input = input
        else:
            for k, v in input.items():
                setattr(state.context, k, v)

        errors: list[str] = []

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

                # Execute agent with retries using tenacity
                result = None
                import tenacity

                try:
                    async for attempt in tenacity.AsyncRetrying(
                        stop=tenacity.stop_after_attempt(step.retry_count + 1),
                        wait=tenacity.wait_exponential(multiplier=1, min=1, max=10),
                        reraise=True,
                    ):
                        with attempt:
                            step.attempts = attempt.retry_state.attempt_number
                            try:
                                if isinstance(step, ParallelStep):
                                    # Execute parallel step
                                    tasks = []
                                    keys = list(step.agents.keys())
                                    for key in keys:
                                        agent_obj = step.agents[key]
                                        template = step.prompt_templates.get(key, "{{ input }}")
                                        agent_prompt = self._render_prompt(template, state.context)
                                        tasks.append(agent_obj.arun(agent_prompt))

                                    # Run conceptually in parallel
                                    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

                                    has_failure = False
                                    combined_content = {}
                                    for key, r in zip(keys, raw_results, strict=False):
                                        if isinstance(r, Exception):
                                            has_failure = True
                                            combined_content[key] = f"ERROR: {r!s}"
                                        elif not r.success:
                                            has_failure = True
                                            combined_content[key] = f"FAILED: {r.error}"
                                        else:
                                            combined_content[key] = r.content
                                            step.results[key] = r

                                    import json
                                    result = AgentResult(
                                        success=not has_failure,
                                        content=json.dumps(combined_content),
                                        state=AgentState.COMPLETED if not has_failure else AgentState.FAILED,
                                        error="One or more parallel agents failed" if has_failure else None
                                    )

                                    if has_failure:
                                        raise Exception("Parallel step reported failures")

                                else:
                                    # Execute single step
                                    result = await step.agent.arun(prompt)
                                    if not result.success:
                                        raise Exception(str(result.error) if hasattr(result, "error") else "Agent execution reported failure")
                            except Exception as e:
                                errors.append(f"Step '{step_name}' attempt {step.attempts} failed: {e}")
                                raise
                except Exception:
                    step.status = StepStatus.FAILED
                    step.result = AgentResult(
                        success=False,
                        content="",
                        state=AgentState.FAILED,
                        error="All attempts failed"
                    )
                    break

                step.result = result
                step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED

                # Store result in context
                state.step_results[step_name] = result
                setattr(state.context, f"{step_name}_result", result.content)
                setattr(state.context, f"{step_name}_result_obj", result)

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
            errors.append(f"Workflow error: {e!s}")

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
        input: str | dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute the workflow synchronously."""
        return asyncio.run(self.arun(input, context))

    def visualize(self) -> str:
        """Generate a Mermaid diagram of the workflow."""
        lines = ["graph TD"]

        # Add steps
        for name, _step in self._steps.items():
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
