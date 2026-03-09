"""
Custom exceptions for AgentKit.

This module defines a comprehensive exception hierarchy for proper
error handling throughout the framework.
"""

from __future__ import annotations

from typing import Any


class AgentKitError(Exception):
    """
    Base exception for all AgentKit errors.

    All custom exceptions in AgentKit inherit from this class,
    making it easy to catch any AgentKit-specific error.

    Attributes:
        message: Human-readable error message
        code: Error code for programmatic handling
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "UNKNOWN_ERROR"
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary format."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }

    def __str__(self) -> str:
        if self.details:
            return f"[{self.code}] {self.message} - {self.details}"
        return f"[{self.code}] {self.message}"


# =============================================================================
# Configuration Errors
# =============================================================================


class ConfigurationError(AgentKitError):
    """Raised when there's a configuration error."""

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if config_key:
            details["config_key"] = config_key
        super().__init__(message, code="CONFIG_ERROR", details=details)


class MissingAPIKeyError(ConfigurationError):
    """Raised when a required API key is missing."""

    def __init__(self, provider: str) -> None:
        super().__init__(
            message=f"API key required for provider '{provider}'",
            config_key=f"{provider.upper()}_API_KEY",
        )
        self.code = "MISSING_API_KEY"
        self.provider = provider


class InvalidModelError(ConfigurationError):
    """Raised when an invalid model is specified."""

    def __init__(self, model: str, provider: str, available_models: list[str] | None = None) -> None:
        details = {"model": model, "provider": provider}
        if available_models:
            details["available_models"] = available_models
        super().__init__(
            message=f"Invalid model '{model}' for provider '{provider}'",
            config_key="model",
            details=details,
        )
        self.code = "INVALID_MODEL"


# =============================================================================
# Provider Errors
# =============================================================================


class ProviderError(AgentKitError):
    """Base exception for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        details["provider"] = provider
        super().__init__(message, code=code or "PROVIDER_ERROR", details=details)
        self.provider = provider


class ProviderConnectionError(ProviderError):
    """Raised when connection to provider fails."""

    def __init__(self, provider: str, original_error: Exception | None = None) -> None:
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            message=f"Failed to connect to {provider}",
            provider=provider,
            code="PROVIDER_CONNECTION_ERROR",
            details=details,
        )


class ProviderRateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        provider: str,
        retry_after: int | None = None,
        limit_type: str | None = None,
    ) -> None:
        details = {}
        if retry_after:
            details["retry_after_seconds"] = retry_after
        if limit_type:
            details["limit_type"] = limit_type
        super().__init__(
            message=f"Rate limit exceeded for {provider}",
            provider=provider,
            code="RATE_LIMIT_EXCEEDED",
            details=details,
        )
        self.retry_after = retry_after


class ProviderAuthenticationError(ProviderError):
    """Raised when authentication fails."""

    def __init__(self, provider: str, message: str | None = None) -> None:
        super().__init__(
            message=message or f"Authentication failed for {provider}",
            provider=provider,
            code="AUTHENTICATION_ERROR",
        )


class ProviderModelNotSupportedError(ProviderError):
    """Raised when the requested model is not supported."""

    def __init__(self, provider: str, model: str, available_models: list[str] | None = None) -> None:
        details = {"requested_model": model}
        if available_models:
            details["available_models"] = available_models
        super().__init__(
            message=f"Model '{model}' not supported by {provider}",
            provider=provider,
            code="MODEL_NOT_SUPPORTED",
            details=details,
        )


class ProviderResponseError(ProviderError):
    """Raised when provider returns an invalid response."""

    def __init__(
        self,
        provider: str,
        message: str,
        response: Any | None = None,
    ) -> None:
        details = {}
        if response:
            details["response"] = str(response)[:500]
        super().__init__(
            message=message,
            provider=provider,
            code="INVALID_RESPONSE",
            details=details,
        )


# =============================================================================
# Tool Errors
# =============================================================================


class ToolError(AgentKitError):
    """Base exception for tool-related errors."""

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, code=code or "TOOL_ERROR", details=details)
        self.tool_name = tool_name


class ToolNotFoundError(ToolError):
    """Raised when a tool is not found."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            message=f"Tool '{tool_name}' not found",
            tool_name=tool_name,
            code="TOOL_NOT_FOUND",
        )


class ToolExecutionError(ToolError):
    """Raised when tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        original_error: Exception | None = None,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        details = {}
        if original_error:
            details["original_error"] = str(original_error)
        if arguments:
            details["arguments"] = arguments
        super().__init__(
            message=f"Tool '{tool_name}' execution failed: {original_error}",
            tool_name=tool_name,
            code="TOOL_EXECUTION_ERROR",
            details=details,
        )
        self.original_error = original_error


class ToolValidationError(ToolError):
    """Raised when tool argument validation fails."""

    def __init__(
        self,
        tool_name: str,
        validation_errors: list[dict[str, Any]],
    ) -> None:
        super().__init__(
            message=f"Tool '{tool_name}' argument validation failed",
            tool_name=tool_name,
            code="TOOL_VALIDATION_ERROR",
            details={"validation_errors": validation_errors},
        )
        self.validation_errors = validation_errors


class RequireApproval(ToolError):
    """Raised when Human-in-the-Loop approval is required but not granted."""

    def __init__(self, tool_name: str, arguments: dict[str, Any] | None = None) -> None:
        super().__init__(
            message=f"Execution of '{tool_name}' requires human approval",
            tool_name=tool_name,
            code="REQUIRE_APPROVAL",
            details={"arguments": arguments},
        )
        self.arguments = arguments


class ToolTimeoutError(ToolError):
    """Raised when tool execution times out."""

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"Tool '{tool_name}' execution timed out after {timeout_seconds}s",
            tool_name=tool_name,
            code="TOOL_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
        )


# =============================================================================
# Agent Errors
# =============================================================================


class AgentError(AgentKitError):
    """Base exception for agent-related errors."""

    def __init__(
        self,
        message: str,
        agent_name: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        details = details or {}
        if agent_name:
            details["agent_name"] = agent_name
        super().__init__(message, code=code or "AGENT_ERROR", details=details)
        self.agent_name = agent_name


class AgentMaxIterationsError(AgentError):
    """Raised when agent exceeds maximum iterations."""

    def __init__(self, agent_name: str, max_iterations: int) -> None:
        super().__init__(
            message=f"Agent '{agent_name}' exceeded maximum iterations ({max_iterations})",
            agent_name=agent_name,
            code="MAX_ITERATIONS_EXCEEDED",
            details={"max_iterations": max_iterations},
        )


class AgentCancelledError(AgentError):
    """Raised when agent execution is cancelled."""

    def __init__(self, agent_name: str) -> None:
        super().__init__(
            message=f"Agent '{agent_name}' execution was cancelled",
            agent_name=agent_name,
            code="AGENT_CANCELLED",
        )


class AgentTimeoutError(AgentError):
    """Raised when agent execution times out."""

    def __init__(self, agent_name: str, timeout_seconds: float) -> None:
        super().__init__(
            message=f"Agent '{agent_name}' execution timed out after {timeout_seconds}s",
            agent_name=agent_name,
            code="AGENT_TIMEOUT",
            details={"timeout_seconds": timeout_seconds},
        )


# =============================================================================
# Memory Errors
# =============================================================================


class MemoryError(AgentKitError):
    """Base exception for memory-related errors."""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code or "MEMORY_ERROR", details=details)


class MemoryStorageError(MemoryError):
    """Raised when memory storage operation fails."""

    def __init__(self, operation: str, original_error: Exception | None = None) -> None:
        details = {"operation": operation}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            message=f"Memory storage operation '{operation}' failed",
            code="MEMORY_STORAGE_ERROR",
            details=details,
        )


# =============================================================================
# Orchestration Errors
# =============================================================================


class OrchestrationError(AgentKitError):
    """Base exception for orchestration errors."""

    def __init__(
        self,
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code or "ORCHESTRATION_ERROR", details=details)


class WorkflowError(OrchestrationError):
    """Raised when workflow execution fails."""

    def __init__(
        self,
        workflow_name: str,
        step: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        details = {"workflow_name": workflow_name}
        if step:
            details["step"] = step
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            message=f"Workflow '{workflow_name}' failed at step '{step}'",
            code="WORKFLOW_ERROR",
            details=details,
        )


class AgentCommunicationError(OrchestrationError):
    """Raised when inter-agent communication fails."""

    def __init__(
        self,
        source_agent: str,
        target_agent: str,
        message: str,
    ) -> None:
        super().__init__(
            message=f"Communication failed from '{source_agent}' to '{target_agent}': {message}",
            code="AGENT_COMMUNICATION_ERROR",
            details={"source_agent": source_agent, "target_agent": target_agent},
        )
