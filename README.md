<p align="center">
  <a href="https://github.com/JohnMikron/agentkit">
    <img src="https://img.shields.io/badge/version-1.1.1-blue.svg" alt="Version">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10+-green.svg" alt="Python">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
  </a>
  <a href="https://pypi.org/project/agentkit/">
    <img src="https://img.shields.io/pypi/dm/agentkit.svg" alt="Downloads">
  </a>
  <a href="https://github.com/JohnMikron/agentkit/actions">
    <img src="https://img.shields.io/github/actions/workflow/status/JohnMikron/agentkit/ci.yml?branch=main" alt="CI">
  </a>
</p>

<h1 align="center">🎯 AgentKit</h1>

<p align="center">
  <strong>Enterprise-Grade AI Agent Framework</strong>
</p>

<p align="center">
  Build AI agents in <strong>5 lines of code, not 500</strong>.
</p>

<p align="center">
  The most powerful, simple, and production-ready Python agent framework for 2026.
</p>

---

## 🚀 Why AgentKit?

| Feature | LangGraph | CrewAI | AutoGen | Pydantic AI | **AgentKit** |
|---------|-----------|--------|---------|-------------|--------------|
| **Lines for basic agent** | 50+ | 30+ | 40+ | 20+ | **5** |
| **Learning curve** | High | Medium | High | Medium | **Low** |
| **Dependencies** | Heavy | Heavy | Heavy | Medium | **Minimal** |
| **Multi-LLM support** | Limited | Limited | Microsoft-focused | Good | **Excellent** |
| **Local models** | Complex | Complex | No | Medium | **One-line** |
| **MCP support** | No | No | No | No | **Yes** |
| **Memory options** | Basic | Basic | Basic | Basic | **Advanced** |
| **Debugging tools** | Complex | Limited | Limited | Good | **Built-in** |
| **Streaming** | Yes | No | No | Yes | **Yes** |
| **Async support** | Yes | No | Yes | Yes | **Yes** |
| **Production ready** | Medium | Medium | Medium | Good | **Enterprise** |

---

## 📦 Installation

```bash
pip install agentkit
```

With specific providers:

```bash
# OpenAI support
pip install agentkit[openai]

# Anthropic support
pip install agentkit[anthropic]

# Google support
pip install agentkit[google]

# Mistral support
pip install agentkit[mistral]

# Local models (Ollama)
pip install agentkit[ollama]

# Redis caching
pip install agentkit[redis]

# Vector memory (semantic search)
pip install agentkit[vector]

# Everything
pip install agentkit[all]
```

---

## ⚡ Quick Start

### Basic Agent (5 lines!)

```python
from agentkit import Agent

agent = Agent("assistant")

@agent.tool
def search(query: str) -> str:
    """Search the web for information"""
    return f"Results for: {query}"

result = agent.run("Search for Python news")
```

### With Memory

```python
agent = Agent("assistant", memory=True)

agent.run("My name is Alice")
agent.run("What's my name?")  # Knows it's Alice!
```

### With Specific LLM

```python
# OpenAI
agent = Agent("assistant", model="gpt-5.3-chat-latest")

# Anthropic
agent = Agent("assistant", model="claude-4-6-sonnet-latest")

# Google
agent = Agent("assistant", model="gemini-3.1-flash-lite-preview")

# Mistral
agent = Agent("assistant", model="mistral-large-2601")

# Local (Ollama) - ONE LINE!
agent = Agent("assistant", model="local:llama3.3")
```

### With Debugging Hooks

```python
agent = Agent("assistant")

@agent.on_tool_call_start
def log_tool(event):
    print(f"🔧 Calling: {event.data['tool_name']}")

@agent.on_tool_call_end
def log_result(event):
    print(f"✅ Completed")

@agent.on_llm_response
def log_response(event):
    print(f"💭 Response received")
```

---

## 🔧 Features

### 🛠️ Simple Tool System

Add tools with a decorator. Automatic JSON Schema generation.

```python
@agent.tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get weather for a location.
    
    Args:
        location: City name or coordinates
        unit: Temperature unit (celsius or fahrenheit)
    """
    return f"Sunny, 22°{unit[0].upper()} in {location}"

@agent.tool
def send_email(to: str, subject: str, body: str) -> bool:
    """Send an email."""
    return True
```

### 🧠 Advanced Memory

Multiple memory backends:

```python
# In-memory (ephemeral)
agent = Agent("assistant", memory=True)

# Persistent file storage
agent = Agent("assistant", memory=True, memory_file="chat.json")

# Redis (distributed)
from agentkit.core.memory import RedisStorage, Memory

storage = RedisStorage(redis_url="redis://localhost:6379")
memory = Memory(storage=storage)
agent = Agent("assistant", memory=memory)
```

### 🔌 Multiple LLM Providers

Works with any LLM out of the box:

```python
# Auto-detect from model name
agent = Agent("assistant", model="gpt-5.3-chat-latest")        # → OpenAI
agent = Agent("assistant", model="claude-4-6-sonnet")  # → Anthropic
agent = Agent("assistant", model="gemini-3.1-flash-lite-preview")   # → Google

# Explicit provider
agent = Agent("assistant", model="openai:gpt-5.3-chat-latest")
agent = Agent("assistant", model="anthropic:claude-4-6-opus")
agent = Agent("assistant", model="google:gemini-3.1-pro")
agent = Agent("assistant", model="mistral:mistral-large-2601")
agent = Agent("assistant", model="local:llama3.3")
```

### 🔍 Built-in Debugging

Monitor every step with event hooks:

```python
agent = Agent("assistant")

@agent.on_start
def on_start(event):
    print(f"🚀 Agent {event.agent_name} started")

@agent.on_llm_request
def on_request(event):
    print(f"📡 Sending request to LLM")

@agent.on_llm_response
def on_response(event):
    print(f"📥 Response: {event.data.get('has_tool_calls')}")

@agent.on_tool_call_start
def on_tool_start(event):
    print(f"🔧 Tool: {event.data['tool_name']}")
    print(f"   Args: {event.data['arguments']}")

@agent.on_tool_call_end
def on_tool_end(event):
    print(f"✅ Tool completed (error: {event.data['is_error']})")

@agent.on_end
def on_end(event):
    print(f"🏁 Agent finished in {event.data['iterations']} iterations")
```

### ⚡ Async & Streaming

Both sync and async APIs:

```python
# Sync
result = agent.run("Hello")

# Async
result = await agent.arun("Hello")

# Streaming
for chunk in agent.stream("Hello"):
    print(chunk, end="")

# Async streaming
async for chunk in agent.astream("Hello"):
    print(chunk, end="")
```

### 💾 Caching

Response caching with multiple backends:

```python
from agentkit.utils.cache import InMemoryCache, RedisCache, cached

# In-memory cache
cache = InMemoryCache(max_size=1000, default_ttl=3600)

# Redis cache
cache = RedisCache(redis_url="redis://localhost:6379")

# Decorator
@cached(cache)
def expensive_operation(query: str) -> str:
    return "result"
```

---

## 📚 CLI Tools

```bash
# Run an agent
agentkit run "What is Python?" --model gpt-4o

# Interactive shell
agentkit shell --model claude-3-5-sonnet --memory

# List providers
agentkit providers

# List models
agentkit models --provider ollama

# Test connection
agentkit test --model gpt-4o

# Show info
agentkit info
```

---

## 🏗️ Architecture

```
agentkit/
├── core/
│   ├── agent.py       # Main Agent class
│   ├── types.py       # Core types and data structures
│   ├── tools.py       # Tool decorator and registry
│   ├── memory.py      # Memory storage backends
│   ├── config.py      # Configuration management
│   └── exceptions.py  # Custom exceptions
├── providers/
│   ├── base.py        # Abstract provider interface
│   ├── openai.py      # OpenAI provider
│   ├── anthropic.py   # Anthropic provider
│   ├── google.py      # Google AI provider
│   ├── mistral.py     # Mistral provider
│   └── ollama.py      # Ollama (local) provider
├── mcp/               # Model Context Protocol support
├── orchestration/     # Multi-agent orchestration
├── utils/
│   ├── cache.py       # Caching utilities
│   └── logging.py     # Structured logging
└── cli.py             # Command-line interface
```

### Minimal Dependencies

```toml
[dependencies]
pydantic = "^2.5"        # Type safety
pydantic-settings = "^2" # Configuration
httpx = "^0.27"          # HTTP client
tenacity = "^8.2"        # Retries
structlog = "^24"        # Logging
cachetools = "^5.3"      # Caching
jsonschema = "^4.21"     # Validation
typer = "^0.9"           # CLI
rich = "^13.7"           # Pretty output
```

No heavy dependencies, no conflicts.

---

## 🔄 Comparison with LangGraph

### LangGraph (50+ lines)

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]

@tool
def search(query: str) -> str:
    return "results"

tools = [search]
tool_executor = ToolExecutor(tools)
model = ChatOpenAI(model="gpt-4").bind_tools(tools)

def should_continue(state):
    if not state["messages"][-1].tool_calls:
        return END
    return "tools"

def call_model(state):
    return {"messages": [model.invoke(state["messages"])]}

def call_tools(state):
    tool_calls = state["messages"][-1].tool_calls
    results = [tool_executor.invoke(tc) for tc in tool_calls]
    return {"messages": results}

graph = StateGraph(State)
graph.add_node("agent", call_model)
graph.add_node("tools", call_tools)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")

app = graph.compile()
result = app.invoke({"messages": [("user", "Search")]})
```

### AgentKit (5 lines)

```python
from agentkit import Agent

agent = Agent("assistant")

@agent.tool
def search(query: str) -> str:
    return "results"

result = agent.run("Search")
```

**Same result. 10x less code.**

---

## 📖 API Reference

### Agent

```python
Agent(
    name: str = "agent",           # Agent name
    model: str = "gpt-4o-mini",    # Model to use
    memory: bool = False,          # Enable memory
    system_prompt: str = None,     # System prompt
    max_iterations: int = 10,      # Max tool iterations
    timeout: float = 60.0,         # Request timeout
)
```

### Methods

| Method | Description |
|--------|-------------|
| `run(prompt)` | Run synchronously |
| `arun(prompt)` | Run asynchronously |
| `stream(prompt)` | Stream response |
| `astream(prompt)` | Stream asynchronously |
| `tool(func)` | Add tool decorator |
| `add_tool(tool)` | Add Tool instance |
| `cancel()` | Cancel execution |
| `clear_memory()` | Clear memory |

### Event Hooks

| Hook | When Called |
|------|-------------|
| `@agent.on_start` | Agent starts |
| `@agent.on_end` | Agent finishes |
| `@agent.on_error` | Error occurs |
| `@agent.on_llm_request` | LLM request |
| `@agent.on_llm_response` | LLM response |
| `@agent.on_tool_call_start` | Tool starts |
| `@agent.on_tool_call_end` | Tool ends |

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT License - see [LICENSE](LICENSE).

---

<p align="center">
  <strong>Stop writing 500 lines. Start building agents.</strong>
</p>

<p align="center">
  <a href="https://github.com/JohnMikron/agentkit">GitHub</a>
</p>
