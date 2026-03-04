"""
Configuration management for AgentKit.

This module provides a comprehensive configuration system with:
- Environment variable support
- Type-safe settings
- Validation
- Multiple provider configurations
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Set, Union

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentkit.core.types import ModelId


class LLMSettings(BaseSettings):
    """
    Settings for LLM providers.

    Supports configuration via environment variables or direct initialization.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    google_api_key: Optional[str] = Field(default=None, description="Google AI API key")
    mistral_api_key: Optional[str] = Field(default=None, description="Mistral API key")

    # Default model configuration
    default_model: str = Field(
        default=ModelId.GPT_5_3.value,
        description="Default model to use",
    )
    default_provider: str = Field(
        default="auto",
        description="Default provider (auto, openai, anthropic, google, mistral, ollama)",
    )

    # Model parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, ge=1, description="Maximum tokens in response")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stop_sequences: Optional[List[str]] = Field(default=None, description="Stop sequences")

    # Timeout settings
    request_timeout: float = Field(default=60.0, ge=1.0, description="Request timeout in seconds")
    connect_timeout: float = Field(default=10.0, ge=1.0, description="Connection timeout")

    # Retry settings
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, ge=0.0, description="Base retry delay in seconds")
    retry_multiplier: float = Field(default=2.0, ge=1.0, description="Exponential backoff multiplier")

    # Rate limiting
    requests_per_minute: Optional[int] = Field(default=None, description="Rate limit for requests")
    tokens_per_minute: Optional[int] = Field(default=None, description="Rate limit for tokens")

    @field_validator("default_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model identifier."""
        # Allow any string - validation happens at provider level
        return v

    @model_validator(mode="after")
    def check_api_keys(self) -> "LLMSettings":
        """Check that at least one API key is set if using cloud providers."""
        # This is just a warning, not an error - local models don't need keys
        return self


class AgentSettings(BaseSettings):
    """
    Settings for agent behavior.

    Controls how agents execute, handle tools, and manage memory.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_AGENT_",
        env_file=".env",
        extra="ignore",
    )

    # Execution settings
    max_iterations: int = Field(default=10, ge=1, description="Maximum agent iterations")
    max_tool_calls: int = Field(default=50, ge=1, description="Maximum tool calls per run")
    max_concurrent_tools: int = Field(default=5, ge=1, description="Maximum concurrent tool calls")

    # Tool settings
    tool_timeout: float = Field(default=30.0, ge=1.0, description="Tool execution timeout")
    validate_tool_args: bool = Field(default=True, description="Validate tool arguments")
    strict_tool_validation: bool = Field(default=False, description="Use strict JSON Schema validation")

    # Memory settings
    memory_enabled: bool = Field(default=False, description="Enable conversation memory")
    memory_max_messages: int = Field(default=100, ge=1, description="Maximum messages in memory")
    memory_file: Optional[str] = Field(default=None, description="File path for persistent memory")

    # Streaming
    streaming_enabled: bool = Field(default=True, description="Enable streaming responses")

    # Hooks
    hooks_enabled: bool = Field(default=True, description="Enable event hooks")


class CacheSettings(BaseSettings):
    """
    Settings for caching.

    Controls response caching and semantic caching behavior.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_CACHE_",
        env_file=".env",
        extra="ignore",
    )

    # Cache settings
    enabled: bool = Field(default=True, description="Enable caching")
    ttl_seconds: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    max_size: int = Field(default=1000, ge=1, description="Maximum cache entries")

    # Redis settings (optional)
    redis_url: Optional[str] = Field(default=None, description="Redis URL for distributed cache")
    redis_prefix: str = Field(default="agentkit:", description="Redis key prefix")

    # Semantic cache
    semantic_cache_enabled: bool = Field(default=False, description="Enable semantic caching")
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Similarity threshold")


class ObservabilitySettings(BaseSettings):
    """
    Settings for observability.

    Controls logging, tracing, and metrics.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_OBS_",
        env_file=".env",
        extra="ignore",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")

    # Tracing
    tracing_enabled: bool = Field(default=False, description="Enable distributed tracing")
    tracing_exporter: str = Field(default="otlp", description="Trace exporter (otlp, jaeger, zipkin)")
    tracing_endpoint: Optional[str] = Field(default=None, description="Trace endpoint URL")

    # Metrics
    metrics_enabled: bool = Field(default=False, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics server port")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v_upper


class MCPSettings(BaseSettings):
    """
    Settings for Model Context Protocol (MCP) support.

    MCP is an open standard for connecting AI agents to external systems.
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_MCP_",
        env_file=".env",
        extra="ignore",
    )

    # MCP settings
    enabled: bool = Field(default=False, description="Enable MCP support")

    # Server settings
    server_name: str = Field(default="agentkit", description="MCP server name")
    server_version: str = Field(default="1.0.0", description="MCP server version")

    # Transport settings
    transport: str = Field(default="stdio", description="Transport type (stdio, http)")
    http_port: int = Field(default=8080, description="HTTP transport port")

    # Tool settings
    expose_agent_tools: bool = Field(default=True, description="Expose agent tools via MCP")
    expose_memory: bool = Field(default=True, description="Expose memory operations via MCP")


class Settings(BaseSettings):
    """
    Main settings class combining all configuration sections.

    This is the primary configuration class for AgentKit.
    Load it once and access all settings.

    Example:
        from agentkit.core.config import Settings

        settings = Settings()
        print(settings.llm.default_model)
    """

    model_config = SettingsConfigDict(
        env_prefix="AGENTKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Nested settings
    llm: LLMSettings = Field(default_factory=LLMSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)

    # Debug mode
    debug: bool = Field(default=False, description="Enable debug mode")

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls()

    @classmethod
    def from_file(cls, path: str) -> "Settings":
        """Load settings from a file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Settings are loaded once and cached for the lifetime of the application.
    Clear the cache with get_settings.cache_clear() if needed.
    """
    return Settings.from_env()


def clear_settings_cache() -> None:
    """Clear the settings cache."""
    get_settings.cache_clear()
