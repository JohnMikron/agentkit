"""
Command-line interface for AgentKit.

Provides tools for:
- Running agents from command line
- Testing providers
- Managing memory
- Benchmarking
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="agentkit",
    help="Enterprise-grade AI Agent Framework",
    add_completion=False,
)

console = Console()


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The prompt to send to the agent"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model to use"),
    memory: bool = typer.Option(False, "--memory", help="Enable memory"),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt"),
    tools: str | None = typer.Option(None, "--tools", "-t", help="Python file with tools"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug mode"),
) -> None:
    """Run an agent with a prompt."""
    from agentkit import Agent
    from agentkit.utils.logging import setup_logging

    # Setup logging
    log_level = "DEBUG" if debug else "INFO"
    setup_logging(level=log_level, format="text")

    # Create agent
    agent = Agent(
        name="cli_agent",
        model=model,
        memory=memory,
        system_prompt=system,
    )

    # Load tools if specified
    if tools:
        _load_tools(agent, tools)

    # Run agent
    if stream:
        _run_streaming(agent, prompt)
    else:
        _run_sync(agent, prompt)


def _load_tools(agent, tools_file: str) -> None:
    """Load tools from a Python file."""
    path = Path(tools_file)
    if not path.exists():
        console.print(f"[red]Error: Tools file not found: {tools_file}[/red]")
        raise typer.Exit(1)

    # Read and exec the file to get tools
    import importlib.util

    spec = importlib.util.spec_from_file_location("tools_module", path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find functions decorated with @tool or functions starting with tool_
        for name in dir(module):
            obj = getattr(module, name)
            if callable(obj) and (name.startswith("tool_") or getattr(obj, "_is_tool", False)):
                agent.tool(obj)


def _run_streaming(agent, prompt: str) -> None:
    """Run agent with streaming output."""
    console.print("\n[bold green]Assistant:[/bold green]")

    try:
        for chunk in agent.stream(prompt):
            console.print(chunk, end="")
        console.print()  # Newline at end
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


def _run_sync(agent, prompt: str) -> None:
    """Run agent without streaming."""
    with console.status("[bold green]Thinking...[/bold green]"):
        try:
            result = agent.run(prompt)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from e

    console.print(f"\n[bold green]Assistant:[/bold green] {result}")


@app.command()
def providers() -> None:
    """List available providers and their status."""
    table = Table(title="Available Providers")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Models")

    providers_info = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-4o, gpt-4o-mini, o1, o1-mini"),
        ("Anthropic", "ANTHROPIC_API_KEY", "claude-3-5-sonnet, claude-3-opus"),
        ("Google", "GOOGLE_API_KEY", "gemini-2.0-flash, gemini-2.0-pro"),
        ("Mistral", "MISTRAL_API_KEY", "mistral-large, mistral-small, codestral"),
        ("Ollama", "OLLAMA_HOST (optional)", "llama3.2, mistral, phi3, qwen2.5"),
    ]

    import os

    for provider, env_var, models in providers_info:
        # Check if API key is set
        key_name = env_var.split(" ")[0]
        is_configured = bool(os.environ.get(key_name))

        status = "[green]✓ Configured[/green]" if is_configured else "[yellow]○ Not configured[/yellow]"
        table.add_row(provider, status, models)

    console.print(table)


@app.command()
def models(
    provider: str = typer.Option("openai", "--provider", "-p", help="Provider to list models for"),
) -> None:
    """List available models for a provider."""
    if provider == "ollama":
        try:
            from agentkit.providers.ollama import OllamaProvider

            ollama = OllamaProvider(model="dummy")
            models_list = ollama.list_models()

            if models_list:
                console.print("\n[bold]Available Ollama Models:[/bold]\n")
                for model in models_list:
                    console.print(f"  • {model}")
            else:
                console.print("[yellow]No models found. Pull a model with: ollama pull llama3.2[/yellow]")
        except Exception as e:
            console.print(f"[red]Error connecting to Ollama: {e}[/red]")

    else:
        # Show recommended models for each provider
        models_map = {
            "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini"],
            "anthropic": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest", "claude-3-opus-latest"],
            "google": ["gemini-2.0-flash", "gemini-2.0-pro"],
            "mistral": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"],
        }

        models_list = models_map.get(provider, [])
        if models_list:
            console.print(f"\n[bold]Recommended {provider.title()} Models:[/bold]\n")
            for model in models_list:
                console.print(f"  • {model}")
        else:
            console.print(f"[yellow]Unknown provider: {provider}[/yellow]")


@app.command()
def test(
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model to test"),
    prompt: str = typer.Option("Hello, who are you?", "--prompt", "-p", help="Test prompt"),
) -> None:
    """Test a provider connection."""
    from agentkit import Agent

    console.print(f"\n[bold]Testing {model}...[/bold]\n")

    try:
        agent = Agent(name="test", model=model)
        result = agent.run(prompt)

        console.print(Panel(result, title="Response", border_style="green"))
        console.print("\n[green]✓ Test successful![/green]")

    except Exception as e:
        console.print(f"\n[red]✗ Test failed: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def info() -> None:
    """Show AgentKit information."""
    console.print(
        Panel.fit(
            "[bold cyan]AgentKit[/bold cyan] - Enterprise-grade AI Agent Framework\n\n"
            "Build AI agents in 5 lines of code, not 500.\n\n"
            "[bold]Features:[/bold]\n"
            "• Multiple LLM providers (OpenAI, Anthropic, Google, Mistral, Ollama)\n"
            "• Built-in memory management\n"
            "• Tool system with validation\n"
            "• Streaming support\n"
            "• Debugging hooks\n"
            "• Both sync and async APIs\n\n"
            "[bold]Quick Start:[/bold]\n"
            "  from agentkit import Agent\n\n"
            "  agent = Agent('assistant')\n\n"
            "  @agent.tool\n"
            "  def search(query: str) -> str:\n"
            "      return 'results...'\n\n"
            "  result = agent.run('Hello!')\n",
            title="🎯 AgentKit v1.0.0",
            border_style="cyan",
        )
    )


@app.command()
def shell(
    model: str = typer.Option("gpt-4o-mini", "--model", "-m", help="Model to use"),
    memory: bool = typer.Option(True, "--memory/--no-memory", help="Enable memory"),
    system: str | None = typer.Option(None, "--system", "-s", help="System prompt"),
) -> None:
    """Start an interactive shell session with an agent."""
    from agentkit import Agent

    console.print("\n[bold cyan]AgentKit Shell[/bold cyan]")
    console.print(f"Model: {model} | Memory: {memory}")
    console.print("Type 'exit' or 'quit' to exit, 'clear' to clear memory\n")

    agent = Agent(
        name="shell_agent",
        model=model,
        memory=memory,
        system_prompt=system or "You are a helpful assistant. Respond concisely.",
    )

    while True:
        try:
            prompt = console.input("[bold green]You:[/bold green] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n\n[yellow]Goodbye![/yellow]")
            break

        if prompt.lower() in ("exit", "quit"):
            console.print("\n[yellow]Goodbye![/yellow]")
            break

        if prompt.lower() == "clear":
            agent.clear_memory()
            console.print("[yellow]Memory cleared.[/yellow]\n")
            continue

        if not prompt.strip():
            continue

        try:
            console.print()
            for chunk in agent.stream(prompt):
                console.print(chunk, end="")
            console.print("\n")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")


if __name__ == "__main__":
    app()
