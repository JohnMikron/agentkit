"""
Tool system for AgentKit.

This module provides a comprehensive tool system with:
- Simple decorator-based tool creation
- Automatic JSON Schema generation
- Validation
- Async support
- Built-in tools
"""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Callable
from typing import (
    Any,
    Literal,
    TypeVar,
    Union,
    get_type_hints,
)

import jsonschema  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from agentkit.core.exceptions import ToolError, ToolValidationError
from agentkit.core.types import ToolDefinition, ToolResult

# Type variables
F = TypeVar("F", bound=Callable[..., Any])
T = TypeVar("T")


# JSON Schema type mapping
_PYTHON_TO_JSON_TYPE: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
    type(None): "null",
}


def _get_json_type(python_type: type) -> str:
    """Convert Python type to JSON Schema type."""
    # Handle typing types
    origin = getattr(python_type, "__origin__", None)

    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    if origin is Union:
        # Handle Optional[X] = Union[X, None]
        args = getattr(python_type, "__args__", ())
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return _get_json_type(non_none_args[0])
        return "any"

    return _PYTHON_TO_JSON_TYPE.get(python_type, "string")


def _generate_schema_from_function(func: Callable[..., Any]) -> dict[str, Any]:
    """
    Generate JSON Schema from a function signature.

    Analyzes type hints and docstring to create a complete schema.
    """
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    # Extract parameter descriptions from docstring
    param_docs = _parse_param_docs(func.__doc__ or "")

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        param_type = hints.get(param_name, str)
        json_type = _get_json_type(param_type)

        prop: dict[str, Any] = {"type": json_type}

        # Add description from docstring
        if param_name in param_docs:
            prop["description"] = param_docs[param_name]

        # Add enum if it's a Literal type
        origin = getattr(param_type, "__origin__", None)
        if origin is Literal:
            prop["enum"] = list(param_type.__args__)

        # Handle array items
        if json_type == "array" and hasattr(param_type, "__args__"):
            item_type = param_type.__args__[0] if param_type.__args__ else str
            prop["items"] = {"type": _get_json_type(item_type)}

        properties[param_name] = prop

        # Check if required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }

    if required:
        schema["required"] = required

    return schema


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """
    Parse parameter descriptions from docstring.

    Supports Google, NumPy, and Sphinx style docstrings.
    """
    params: dict[str, str] = {}

    lines = docstring.strip().split("\n")
    in_args_section = False
    current_param: str | None = None

    for line in lines:
        stripped = line.strip()

        # Check for Args/Arguments section
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args_section = True
            continue

        if in_args_section:
            # Check for next section
            if stripped and stripped.endswith(":") and not stripped.startswith("-"):
                break

            # Parse parameter
            # Format: param_name (type): description
            # or: param_name: description
            if ":" in stripped and not stripped.startswith("-"):
                parts = stripped.split(":", 1)
                param_part = parts[0].strip()

                # Remove type annotation if present
                if "(" in param_part:
                    param_part = param_part.split("(")[0].strip()

                current_param = param_part
                if len(parts) > 1:
                    params[current_param] = parts[1].strip()
            elif current_param and stripped:
                # Continuation of previous parameter
                params[current_param] += " " + stripped

    return params


class Tool(BaseModel):
    """
    A tool that an agent can use.

    Tools are functions that the LLM can call to interact with external
    systems, APIs, or perform computations.

    Attributes:
        name: The name of the tool (used by the LLM to call it)
        description: What the tool does (shown to the LLM)
        func: The actual function to execute
        parameters: JSON Schema of the parameters
        strict: Whether to use strict validation
        timeout: Execution timeout in seconds
        metadata: Additional metadata
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
    description: str = Field(..., min_length=1)
    func: Callable[..., Any]
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    strict: bool = False
    timeout: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **data: Any) -> None:
        # Auto-generate schema if not provided
        if "parameters" not in data and "func" in data:
            data["parameters"] = _generate_schema_from_function(data["func"])

        # Extract description from docstring if not provided
        if "description" not in data and "func" in data:
            doc = data["func"].__doc__
            if doc:
                # First line is the description
                data["description"] = doc.strip().split("\n")[0]
            else:
                data["description"] = f"Execute {data.get('name', 'tool')}"

        super().__init__(**data)

    def to_definition(self) -> ToolDefinition:
        """Convert to ToolDefinition for LLM provider."""
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            strict=self.strict,
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """
        Validate arguments against the tool's JSON Schema.

        Raises:
            ToolValidationError: If validation fails
        """
        if not self.parameters.get("properties"):
            return

        try:
            jsonschema.validate(instance=arguments, schema=self.parameters)
        except jsonschema.ValidationError as e:
            raise ToolValidationError(
                tool_name=self.name,
                validation_errors=[{"path": list(e.path), "message": e.message}],
            ) from e

    def execute(self, arguments: dict[str, Any], validate: bool = True) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            arguments: Dictionary of arguments to pass to the tool
            validate: Whether to validate arguments before execution

        Returns:
            ToolResult containing the execution result
        """
        if validate:
            self.validate_arguments(arguments)

        start_time = time.perf_counter()

        try:
            # Handle async functions safely by checking the function before calling it
            if inspect.iscoroutinefunction(self.func):
                try:
                    asyncio.get_running_loop()
                    # We're in an async context, can't use run_until_complete in the main thread
                    import concurrent.futures

                    def run_in_new_loop():
                        return asyncio.run(self.func(**arguments))

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(run_in_new_loop)
                        result = future.result()
                except RuntimeError:
                    # No running loop, create one manually
                    result = asyncio.run(self.func(**arguments))
            else:
                result = self.func(**arguments)

            execution_time = (time.perf_counter() - start_time) * 1000

            return ToolResult(
                tool_call_id="",
                name=self.name,
                content=self._serialize_result(result),
                raw_result=result,
                execution_time_ms=execution_time,
            )

        except ToolValidationError:
            raise
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_call_id="",
                name=self.name,
                content=f"Error: {type(e).__name__}: {e!s}",
                is_error=True,
                execution_time_ms=execution_time,
            )

    async def aexecute(self, arguments: dict[str, Any], validate: bool = True) -> ToolResult:
        """
        Execute the tool asynchronously.

        Args:
            arguments: Dictionary of arguments to pass to the tool
            validate: Whether to validate arguments before execution

        Returns:
            ToolResult containing the execution result
        """
        if validate:
            self.validate_arguments(arguments)

        start_time = time.perf_counter()

        try:
            result = self.func(**arguments)

            if asyncio.iscoroutine(result):
                result = await result

            execution_time = (time.perf_counter() - start_time) * 1000

            return ToolResult(
                tool_call_id="",
                name=self.name,
                content=self._serialize_result(result),
                raw_result=result,
                execution_time_ms=execution_time,
            )

        except ToolValidationError:
            raise
        except Exception as e:
            execution_time = (time.perf_counter() - start_time) * 1000
            return ToolResult(
                tool_call_id="",
                name=self.name,
                content=f"Error: {type(e).__name__}: {e!s}",
                is_error=True,
                execution_time_ms=execution_time,
            )

    def _serialize_result(self, result: Any) -> str:
        """Serialize result to string."""
        if isinstance(result, str):
            return result
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(result)


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
    timeout: float | None = None,
) -> Tool | Callable[[F], Tool]:
    """
    Decorator to create a Tool from a function.

    Can be used with or without arguments:

        @tool
        def my_func(x: str) -> str:
            return x

    Args:
        func: The function to wrap
        name: Optional custom name
        description: Optional custom description
        strict: Whether to use strict validation
        timeout: Execution timeout in seconds

    Returns:
        Tool instance or decorator function
    """

    def decorator(f: F) -> Tool:
        return Tool(
            name=name or f.__name__,
            description=description
            or (f.__doc__.strip().split("\n")[0] if f.__doc__ else f"Execute {f.__name__}"),
            func=f,
            strict=strict,
            timeout=timeout,
        )

    if func is not None:
        return decorator(func)
    return decorator



@tool
def duckduckgo_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.

    Args:
        query: The search query
    """
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"DuckDuckGo search for '{query}' returned no results. Falling back to internal knowledge."

        formatted_results = []
        for r in results:
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "")
            if title and href:
                formatted_results.append(f"{title}\n{href}\n{body}")

        return "\n\n".join(formatted_results)
    except ImportError:
        return "DuckDuckGo search error: The 'duckduckgo-search' library is not installed. Please install it using `pip install duckduckgo-search`."
    except Exception as e:
        return f"DuckDuckGo search error: {e!s}. Please check your internet connection."


class ToolRegistry:
    """
    Registry for managing tools.

    Maintains a collection of tools and provides methods to add,
    retrieve, and execute tools by name.
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool, overwrite: bool = False) -> ToolRegistry:
        """
        Add a tool to the registry.

        Returns self for chaining.
        """
        if tool.name in self._tools:
            if not overwrite:
                raise ToolError(f"Tool '{tool.name}' already registered", tool_name=tool.name)
        self._tools[tool.name] = tool
        return self

    def remove(self, name: str) -> Tool | None:
        """Remove a tool from the registry."""
        return self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Check if a tool exists."""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> list[ToolDefinition]:
        """Get tool definitions for all tools."""
        return [t.to_definition() for t in self._tools.values()]

    def execute(
        self, name: str, arguments: str | dict[str, Any], validate: bool = True
    ) -> ToolResult:
        """
        Execute a tool by name.

        Args:
            name: Name of the tool to execute
            arguments: JSON string or dict of arguments
            validate: Whether to validate arguments

        Returns:
            ToolResult with execution result
        """
        tool = self._tools.get(name)
        if not tool:
            raise ToolError(f"Tool '{name}' not found", tool_name=name, code="TOOL_NOT_FOUND")

        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise ToolValidationError(
                    tool_name=name,
                    validation_errors=[{"message": f"Invalid JSON: {e}"}],
                ) from e
            if not isinstance(arguments, dict):
                arguments = {}

        return tool.execute(arguments, validate=validate)

    async def aexecute(
        self, name: str, arguments: str | dict[str, Any], validate: bool = True
    ) -> ToolResult:
        """
        Execute a tool asynchronously.

        Args:
            name: Name of the tool to execute
            arguments: JSON string or dict of arguments
            validate: Whether to validate arguments

        Returns:
            ToolResult with execution result
        """
        tool = self._tools.get(name)
        if not tool:
            raise ToolError(f"Tool '{name}' not found", tool_name=name, code="TOOL_NOT_FOUND")

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise ToolValidationError(
                    tool_name=name,
                    validation_errors=[{"message": f"Invalid JSON: {e}"}],
                ) from e
            if not isinstance(arguments, dict):
                arguments = {}

        return await tool.aexecute(arguments, validate=validate)

    def __contains__(self, name: str) -> bool:
        """Check if a tool exists (supports 'in' operator)."""
        return self.has(name)

    def __getitem__(self, name: str) -> Tool:
        """Get a tool by name (supports [] access)."""
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found")
        return tool

    def __len__(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def __iter__(self) -> Any:
        """Iterate over tool names."""
        return iter(self._tools)


# =============================================================================
# Built-in Tools
# =============================================================================


@tool
def calculator(expression: str) -> float:
    """
    Evaluate a mathematical expression safely using AST-based validation.

    Supports basic arithmetic (+, -, *, /), power (^ or **), and common
    math functions (sqrt, sin, cos, tan, log, exp, pi, e).

    Args:
        expression: Mathematical expression to evaluate
    """
    import ast
    import math
    import operator

    # Supported operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # Supported constants and functions
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "floor": math.floor,
        "ceil": math.ceil,
    }

    def _eval(node: Any) -> Any:
        if isinstance(node, (ast.Num, ast.Constant)):
            return getattr(node, "n", getattr(node, "value", None))
        elif isinstance(node, ast.BinOp):
            return operators[type(node.op)](_eval(node.left), _eval(node.right))  # type: ignore
        elif isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](_eval(node.operand))  # type: ignore
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in allowed_names:
                args = [_eval(arg) for arg in node.args]
                import typing

                func = typing.cast("typing.Callable[..., typing.Any]", allowed_names[node.func.id])
                return func(*args)
            raise ValueError(
                f"Function {node.func.id if isinstance(node.func, ast.Name) else 'unknown'} not allowed"
            )
        elif isinstance(node, ast.Name):
            if node.id in allowed_names and not callable(allowed_names[node.id]):
                return allowed_names[node.id]
            raise ValueError(f"Variable {node.id} not allowed")
        else:
            raise TypeError(f"Unsupported syntax: {type(node).__name__}")

    # Preliminary cleanup
    expression = expression.replace("^", "**")

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree.body)
        return float(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate expression: {e}") from e


@tool
def current_datetime(timezone: str | None = None) -> str:
    """
    Get the current date and time.

    Args:
        timezone: Optional timezone name (e.g., 'UTC', 'America/New_York')
    """
    from datetime import datetime

    if timezone:
        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo(timezone)
            return datetime.now(tz).isoformat()
        except ImportError:
            pass

    return datetime.utcnow().isoformat() + "Z"


@tool
def json_parse(json_string: str) -> Any:
    """
    Parse a JSON string into a Python object.

    Args:
        json_string: Valid JSON string to parse
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e


@tool
def json_stringify(obj: Any, indent: int = 2) -> str:
    """
    Convert a Python object to a JSON string.

    Args:
        obj: Python object to convert
        indent: Number of spaces for indentation
    """
    try:
        return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot serialize to JSON: {e}") from e


# Collection of all built-in tools
BUILTIN_TOOLS: list[Any] = [
    calculator,
    current_datetime,
    json_parse,
    json_stringify,
    duckduckgo_search,
]


def get_builtin_tools(
    include: list[str] | None = None, exclude: list[str] | None = None
) -> list[Tool]:
    """
    Get built-in tools, optionally filtered.

    Args:
        include: List of tool names to include
        exclude: List of tool names to exclude

    Returns:
        List of Tool instances
    """
    tools = BUILTIN_TOOLS

    if include:
        tools = [t for t in tools if t.name in include]
    if exclude:
        tools = [t for t in tools if t.name not in exclude]

    return tools
