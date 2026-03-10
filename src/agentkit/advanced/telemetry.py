"""
OpenTelemetry integration for AgentKit.

Provides an AgentHook that automatically traces agent execution, LLM requests,
and tool calls, sending them to an OpenTelemetry collector (e.g. Jaeger, Zipkin, Datadog).
"""

from typing import TYPE_CHECKING

from agentkit.core.agent import Agent
from agentkit.core.types import Event

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer  # type: ignore[import-not-found]

try:
    from opentelemetry import trace  # type: ignore[import-not-found]
    from opentelemetry.trace.status import Status, StatusCode  # type: ignore[import-not-found]

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


class OpenTelemetryHook:
    """
    Hook that exports agent events as OpenTelemetry spans.

    Usage:
        agent = Agent("researcher")
        otel_hook = OpenTelemetryHook("agentkit-tracer")
        otel_hook.attach(agent)
    """

    def __init__(self, tracer_name: str = "agentkit") -> None:
        """Initialize the OpenTelemetry hook."""
        if not HAS_OTEL:
            raise ImportError(
                "OpenTelemetry hook requires opentelemetry packages. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
        self.tracer: Tracer = trace.get_tracer(tracer_name)

        # State tracking for spans
        self._active_spans: dict[str, Span] = {}

    def attach(self, agent: Agent) -> None:
        """Attach this hook to an agent's lifecycle."""
        agent.on_start(self.on_agent_start)
        agent.on_end(self.on_agent_end)
        agent.on_error(self.on_agent_error)
        agent.on_llm_request(self.on_llm_request)
        agent.on_llm_response(self.on_llm_response)
        agent.on_tool_call_start(self.on_tool_call_start)
        agent.on_tool_call_end(self.on_tool_call_end)

    def on_agent_start(self, event: Event) -> None:
        span = self.tracer.start_span(f"AgentRunner.{event.agent_name}")
        span.set_attribute("agent.name", event.agent_name)
        if "prompt" in event.data:
            span.set_attribute("agent.prompt", event.data["prompt"])
        self._active_spans["agent"] = span

    def on_agent_end(self, event: Event) -> None:
        span = self._active_spans.get("agent")
        if span:
            span.set_attribute("agent.iterations", event.data.get("iterations", 0))
            span.set_status(Status(StatusCode.OK))
            span.end()
            self._active_spans.pop("agent", None)

    def on_agent_error(self, event: Event) -> None:
        span = self._active_spans.get("agent")
        if span:
            span.record_exception(Exception(event.data.get("error", "Unknown Error")))
            span.set_status(Status(StatusCode.ERROR))
            span.end()
            self._active_spans.pop("agent", None)

    def on_llm_request(self, event: Event) -> None:
        span = self.tracer.start_span("LLM.Complete")
        span.set_attribute("llm.iteration", event.data.get("iteration", 0))
        self._active_spans["llm"] = span

    def on_llm_response(self, event: Event) -> None:
        span = self._active_spans.get("llm")
        if span:
            span.set_attribute("llm.has_tools", event.data.get("has_tool_calls", False))
            span.set_status(Status(StatusCode.OK))
            span.end()
            self._active_spans.pop("llm", None)

    def on_tool_call_start(self, event: Event) -> None:
        tool_name = event.data.get("tool_name", "unknown")
        call_id = event.data.get("tool_call_id", "unknown")

        span = self.tracer.start_span(f"Tool.{tool_name}")
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.call_id", call_id)

        # Serialize arguments safely
        args = event.data.get("arguments", {})
        span.set_attribute("tool.arguments", str(args))

        self._active_spans[f"tool_{call_id}"] = span

    def on_tool_call_end(self, event: Event) -> None:
        call_id = event.data.get("tool_call_id", "unknown")
        is_error = event.data.get("is_error", False)

        span = self._active_spans.get(f"tool_{call_id}")
        if span:
            if is_error:
                span.set_status(Status(StatusCode.ERROR))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()
            self._active_spans.pop(f"tool_{call_id}", None)
