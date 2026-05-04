# Agent Development Kit (ADK)
> **GCP Professional ML Engineer Study Guide**

---

## 1. What is ADK?

Agent Development Kit (ADK) is a flexible and modular framework for developing and deploying AI agents. While optimized for Gemini and the Google ecosystem, ADK is model-agnostic, deployment-agnostic, and is built for compatibility with other frameworks. ADK was designed to make agent development feel more like software development, to make it easier for developers to create, deploy, and orchestrate agentic architectures that range from simple tasks to complex workflows.

ADK is the same framework powering agents within Google products like Agentspace and the Google Customer Engagement Suite (CES).

**Install:**

```bash
pip install google-adk
# For Vertex AI Agent Engine deployment:
pip install google-cloud-aiplatform[agent_engines,adk]
```

---

## 2. Core ADK Primitives

ADK is built around a few key primitives and concepts that make it powerful and flexible:

- **Agent:** The fundamental worker unit designed for specific tasks. Agents can use language models (LlmAgent) for complex reasoning, or act as deterministic controllers of the execution, which are called "workflow agents" (SequentialAgent, ParallelAgent, LoopAgent).
- **Tool:** Gives agents abilities beyond conversation, letting them interact with external APIs, search information, run code, or call other services.
- **Callbacks:** Custom code snippets you provide to run at specific points in the agent's process, allowing for checks, logging, or behavior modifications.

---

## 3. Agent Types

### LlmAgent (the primary reasoning agent)

LLM Agents (LlmAgent, Agent): These agents utilize Large Language Models (LLMs) as their core engine to understand natural language, reason, plan, generate responses, and dynamically decide how to proceed or which tools to use, making them ideal for flexible, language-centric tasks.

```python
from google.adk.agents import Agent

root_agent = Agent(
    model="gemini-2.0-flash",          # model to use
    name="customer_support_agent",
    description="Handles customer support queries",
    instruction="""You are a helpful customer support agent.
    Use available tools to look up orders and answer questions.""",
    tools=[get_order_status, lookup_product, cancel_order],
)
```

### Workflow Agents

Define workflows using workflow agents (Sequential, Parallel, Loop) for predictable pipelines, or leverage LLM-driven dynamic routing (LlmAgent transfer) for adaptive behavior.

#### SequentialAgent — ordered pipeline

```python
from google.adk.agents import SequentialAgent

pipeline = SequentialAgent(
    name="data_pipeline",
    description="Processes data in order",
    sub_agents=[
        fetch_agent,       # runs first
        transform_agent,   # runs second
        store_agent,       # runs third
    ],
)
```

#### ParallelAgent — concurrent execution

```python
from google.adk.agents import ParallelAgent

parallel_research = ParallelAgent(
    name="parallel_researcher",
    description="Researches multiple topics concurrently",
    sub_agents=[
        flight_agent,     # runs concurrently
        hotel_agent,      # runs concurrently
        weather_agent,    # runs concurrently
    ],
)
```

#### LoopAgent — iterative until condition met

```python
from google.adk.agents import LoopAgent

refinement_loop = LoopAgent(
    name="refinement_loop",
    description="Iteratively refines output",
    sub_agents=[draft_agent, review_agent],
    max_iterations=5,
)
```

### Agent Type Comparison

| Agent Type | Behaviour | Use When |
|-----------|-----------|---------|
| `LlmAgent` | LLM decides dynamically what to do | Flexible reasoning, natural language tasks |
| `SequentialAgent` | Sub-agents run in fixed order | Deterministic pipelines, ETL workflows |
| `ParallelAgent` | Sub-agents run concurrently | Independent tasks, speed optimisation |
| `LoopAgent` | Repeats sub-agents until condition | Iterative refinement, retry logic |

---

## 4. Tools

Tools extend agent capabilities beyond conversation.

### Custom Function Tools

```python
def get_order_status(order_id: str) -> dict:
    """
    Retrieves the current status of a customer order.

    Args:
        order_id: The unique identifier for the order.

    Returns:
        A dictionary containing the order status and details.
    """
    # ADK uses the docstring to understand when and how to use this tool
    response = requests.get(f"https://api.example.com/orders/{order_id}")
    return response.json()

agent = Agent(
    model="gemini-2.0-flash",
    name="order_agent",
    tools=[get_order_status],   # pass function directly
)
```

> **Key point:** ADK uses the function's **docstring** and **type hints** to automatically generate the tool schema — no manual schema definition needed.

### Built-in Tools

```python
from google.adk.tools import google_search, code_execution

agent = Agent(
    model="gemini-2.0-flash",
    name="research_agent",
    tools=[
        google_search,      # web search capability
        code_execution,     # Python code execution sandbox
    ],
)
```

### AgentTool — Use an Agent as a Tool

```python
from google.adk.tools import AgentTool

# Wrap a specialised agent as a tool for the root agent
flight_tool = AgentTool(agent=flight_agent)
hotel_tool  = AgentTool(agent=hotel_agent)

root_agent = Agent(
    model="gemini-2.0-flash",
    name="travel_agent",
    tools=[flight_tool, hotel_tool],
)
```

### MCP Tools (Model Context Protocol)

```python
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

mcp_tools = MCPToolset(
    connection_params=StdioServerParameters(
        command="uvx",
        args=["mcp-server-filesystem", "/path/to/data"],
    )
)

agent = Agent(
    model="gemini-2.0-flash",
    name="filesystem_agent",
    tools=[mcp_tools],
)
```

### Memory Tools

```python
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.load_memory_tool import LoadMemoryTool

agent = Agent(
    model="gemini-2.0-flash",
    name="memory_agent",
    tools=[
        PreloadMemoryTool(),   # always loads memory at start of each turn
        LoadMemoryTool(),      # agent decides when to load memory
    ],
)
```

---

## 5. Multi-Agent Systems

Multi-Agent System Design: Easily build applications composed of multiple, specialized agents arranged hierarchically. Agents can coordinate complex tasks, delegate sub-tasks using LLM-driven transfer or explicit AgentTool invocation, enabling modular and scalable solutions.

### Hierarchical Agent Example

```python
from google.adk.agents import Agent
from google.adk.tools import AgentTool

# Specialist agents
flight_agent = Agent(
    name="flight_specialist",
    model="gemini-2.0-flash",
    instruction="You only handle flight bookings.",
    tools=[search_flights, book_flight],
)

hotel_agent = Agent(
    name="hotel_specialist",
    model="gemini-2.0-flash",
    instruction="You only handle hotel bookings.",
    tools=[search_hotels, book_hotel],
)

# Root orchestrator delegates to specialists
root_agent = Agent(
    name="travel_orchestrator",
    model="gemini-2.0-flash",
    instruction="You coordinate travel bookings by delegating to specialists.",
    tools=[
        AgentTool(agent=flight_agent),
        AgentTool(agent=hotel_agent),
    ],
)
```

### Agent Transfer (LLM-driven routing)

```python
# Sub-agents listed directly — LLM decides which to hand off to
root_agent = Agent(
    name="triage_agent",
    model="gemini-2.0-flash",
    instruction="Route queries to the appropriate specialist.",
    sub_agents=[billing_agent, tech_agent, returns_agent],
)
```

---

## 6. Session & State Management

### Sessions

Every user interaction with an ADK agent gets a session, and that session is managed by the ADK SessionService. Each session contains important fields, like the session ID, user ID, event history (the conversation thread), and the state.

```python
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from google.adk.sessions import VertexAiSessionService

# Local development — in-memory (lost on restart)
session_service = InMemorySessionService()

# Production with SQL database
session_service = DatabaseSessionService(db_url="postgresql://user:pass@host/db")

# Production with Vertex AI Agent Engine (fully managed)
session_service = VertexAiSessionService(project="my-project", location="us-central1")
```

### Session State

Think of it like the agent's scratchpad during that "phone call" with the user. Each session's state contains a list of key-value pairs, whose values are updated by the agent throughout the session.

```python
from google.adk.tools import ToolContext

def update_preference(preference: str, tool_context: ToolContext) -> str:
    """Updates user preference in session state."""
    tool_context.state["user_preference"] = preference
    return f"Preference updated to: {preference}"
```

Use state values in agent instructions with curly braces:

```python
agent = Agent(
    instruction="The user's current preference is: {user_preference}. "
                "Always respect this setting.",
    # ADK auto-injects state values into the prompt on each turn
)
```

### Session Services Comparison

| Service | Persistence | Best For |
|---------|-------------|---------|
| `InMemorySessionService` | None — lost on restart | Local dev, testing |
| `DatabaseSessionService` | SQL DB (SQLite, PostgreSQL) | Self-managed persistence |
| `VertexAiSessionService` | Fully managed — Vertex AI | Production on GCP |

---

## 7. Memory

### Short-term Memory (within a session)

State within a session — automatically managed by `SessionService`. Lost when session ends.

### Long-term Memory (across sessions)

ADK provides a way to store long-term memories persistently outside the ADK runtime, with a VertexAIMemoryBankService. This memory service uses Vertex AI Memory Bank to intelligently store and retrieve memories from past user interactions. Memory Bank uses the Gemini model to extract key information from session data, to store just the key memories for future use.

```python
from google.adk.memory import VertexAiMemoryBankService

memory_service = VertexAiMemoryBankService(
    project="my-project",
    location="us-central1",
    agent_engine_id="AGENT_ENGINE_ID",
)

runner = Runner(
    agent=root_agent,
    app_name="my-agent-app",
    session_service=session_service,
    memory_service=memory_service,
)
```

### Memory Callback — Save session to Memory Bank

```python
from google.adk.agents.callback_context import CallbackContext

async def save_session_to_memory(callback_context: CallbackContext):
    """Saves session data to long-term memory after each agent turn."""
    await callback_context.add_session_to_memory()
    return None

agent = Agent(
    model="gemini-2.0-flash",
    name="memory_agent",
    after_agent_callback=save_session_to_memory,
    tools=[PreloadMemoryTool()],
)
```

### Memory Service Comparison

| Service | Scope | Persistence | Best For |
|---------|-------|-------------|---------|
| Session State | Within one session | Session lifetime | Scratchpad, turn-to-turn context |
| `InMemoryMemoryService` | Across sessions (in-process) | Process lifetime | Testing only |
| `VertexAiMemoryBankService` | Across sessions (cloud) | Permanent | Production long-term recall |

---

## 8. Callbacks

Callbacks are Python functions that intercept agent execution at defined lifecycle points.

### Callback Hooks

| Callback | Fires When |
|----------|-----------|
| `before_agent_callback` | Before the agent starts processing a turn |
| `after_agent_callback` | After the agent finishes a turn |
| `before_tool_callback` | Before a specific tool is called |
| `after_tool_callback` | After a specific tool returns |
| `before_model_callback` | Before the LLM is called |
| `after_model_callback` | After the LLM returns a response |

### Example — Logging and Guardrail Callback

```python
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse

def safety_check_callback(callback_context: CallbackContext) -> LlmResponse | None:
    """Block requests containing sensitive keywords."""
    user_message = callback_context.user_content.parts[0].text
    if "confidential" in user_message.lower():
        # Returning a value short-circuits — agent never runs
        return LlmResponse(text="I cannot process requests about confidential data.")
    return None   # return None to allow normal execution

agent = Agent(
    model="gemini-2.0-flash",
    name="safe_agent",
    before_model_callback=safety_check_callback,
)
```

---

## 9. The Runner

The `Runner` orchestrates the agent execution loop — it connects the agent to the session service, memory service, and event handling.

```python
from google.adk import Runner
from google.adk.sessions import VertexAiSessionService
from google.adk.memory import VertexAiMemoryBankService

runner = Runner(
    agent=root_agent,
    app_name="my-agent-app",
    session_service=VertexAiSessionService(project="my-project", location="us-central1"),
    memory_service=VertexAiMemoryBankService(
        project="my-project",
        location="us-central1",
        agent_engine_id="ENGINE_ID"
    ),
)
```

---

## 10. Local Development & Testing

### ADK CLI

```bash
# Start interactive web UI (browser-based dev interface)
adk web

# Start API server (FastAPI)
adk api_server

# Run agent from command line
adk run my_agent_module
```

### Local Testing with AdkApp

```python
from google.adk.agents import Agent
from vertexai.agent_engines import AdkApp

agent = Agent(
    model="gemini-2.0-flash",
    name="my_agent",
    tools=[my_tool],
)

app = AdkApp(agent=agent)

# Test locally (uses in-memory sessions)
import asyncio
async def test():
    async for event in app.async_stream_query(
        user_id="test-user-123",
        message="Hello, can you help me?",
    ):
        print(event)

asyncio.run(test())
```

---

## 11. Deploying to Vertex AI Agent Engine

Google Cloud Vertex AI Agent Engine is a set of modular services that help developers scale and govern agents in production. The Agent Engine runtime enables you to deploy agents in production with end-to-end managed infrastructure so you can focus on creating intelligent and impactful agents.

### Deploy via Python SDK

```python
import vertexai
from vertexai.agent_engines import AdkApp

vertexai.init(project="my-project", location="us-central1")
client = vertexai.Client(project="my-project", location="us-central1")

app = AdkApp(agent=root_agent)

# Deploy to Vertex AI Agent Engine
remote_agent = client.agent_engines.create(
    agent=app,
    config={
        "requirements": ["google-cloud-aiplatform[agent_engines,adk]"],
        "staging_bucket": "gs://my-staging-bucket",
    },
)

print(f"Deployed: {remote_agent.resource_name}")
```

### Query the Deployed Agent

```python
# Async streaming query
async for event in remote_agent.async_stream_query(
    user_id="user-123",
    message="What is the weather in London today?",
):
    print(event)
```

### Deploy via ADK CLI

```bash
# Deploy to Vertex AI Agent Engine
adk deploy agent_engine \
  --project=my-project \
  --region=us-central1 \
  --staging-bucket=gs://my-staging-bucket \
  ./my_agent_directory/

# Deploy to Cloud Run
adk deploy cloud_run \
  --project=my-project \
  --region=us-central1 \
  --session_service_uri="agentengine://AGENT_ENGINE_ID" \
  --memory_service_uri="agentengine://AGENT_ENGINE_ID" \
  ./my_agent_directory/
```

### Deployment Targets

| Target | Description | Best For |
|--------|-------------|---------|
| **Vertex AI Agent Engine** | Fully managed, scalable, integrated with Memory Bank and Sessions | Production GCP agents |
| **Cloud Run** | Containerised, serverless, flexible | Custom infra, cost control |
| **Local (adk web)** | Dev UI, FastAPI server | Development and testing |

---

## 12. Agent2Agent (A2A) Protocol

Connect any agent, anywhere with the open Agent2Agent (A2A) protocol. This universal communication standard enables agents across different ecosystems to communicate with each other, irrespective of the framework (ADK, LangGraph, Crew.ai, or others) or vendor they are built on. Using A2A, agents can publish their capabilities and negotiate how they will interact with users — all while working securely together.

---

## 13. Artifacts

Artifact Management: Enable agents to handle files and binary data. The framework provides mechanisms (ArtifactService, context methods) for agents to save, load, and manage versioned artifacts like images, documents, or generated reports during their execution.

```python
# Save an artifact from within a tool
async def generate_report(tool_context) -> str:
    report_bytes = create_pdf_report()
    await tool_context.save_artifact(
        filename="report.pdf",
        artifact=types.Part.from_bytes(data=report_bytes, mime_type="application/pdf"),
    )
    return "Report saved as artifact."
```

---

## 14. ADK vs Other Frameworks

| Feature | ADK | LangChain | LangGraph | CrewAI |
|---------|-----|-----------|-----------|--------|
| Primary language | Python, Java | Python | Python | Python |
| GCP / Gemini optimised | Yes | Partial | Partial | No |
| Workflow agents built-in | Yes | No | Yes | Yes |
| MCP support | Yes | Partial | No | No |
| A2A protocol | Yes | No | No | No |
| Vertex AI deployment | Native | Custom | Custom | Custom |
| Memory Bank integration | Native | Manual | Manual | Manual |

> ADK supports using LangChain, LangGraph, and CrewAI agents as tools or sub-agents — you are not locked in.

---

## 15. Exam-Relevant Tips

- ADK is **open-source** and **model-agnostic** — not restricted to Gemini, but optimised for it.
- The three core primitives are: **Agent**, **Tool**, **Callback**.
- **`LlmAgent`** (also importable as `Agent`) uses an LLM for reasoning; **workflow agents** are deterministic.
- `SequentialAgent` → ordered; `ParallelAgent` → concurrent; `LoopAgent` → iterative.
- ADK uses function **docstrings and type hints** to auto-generate tool schemas — always document your tools.
- **`AgentTool`** wraps a sub-agent as a tool, enabling hierarchical delegation.
- **Session State** is within a single session; **Memory Bank** persists across sessions.
- `VertexAiSessionService` and `VertexAiMemoryBankService` are the production-grade managed options.
- `PreloadMemoryTool` always loads memory at turn start; `LoadMemoryTool` lets the LLM decide.
- **Callbacks** returning a non-None value **short-circuit** execution — useful for guardrails.
- The recommended production deployment target is **Vertex AI Agent Engine**.
- `AdkApp` is the wrapper used to package an ADK agent for deployment to Vertex AI Agent Engine.
- `adk web` launches a **browser-based dev UI** for local interactive testing.
- The **A2A protocol** enables cross-framework agent communication.
- `client.agent_engines.create()` deploys the agent; `remote_agent.async_stream_query()` queries it.

---

## 16. Quick Reference Cheat Sheet

```
CORE PRIMITIVES
  Agent / LlmAgent         →  LLM-driven reasoning agent
  SequentialAgent          →  deterministic ordered pipeline
  ParallelAgent            →  concurrent independent tasks
  LoopAgent                →  iterative until max_iterations
  Tool (function)          →  extend agent with external capabilities
  AgentTool                →  wrap a sub-agent as a tool
  Callback                 →  intercept lifecycle events

SESSION & MEMORY
  InMemorySessionService   →  local dev only
  DatabaseSessionService   →  SQL persistence
  VertexAiSessionService   →  managed production sessions
  VertexAiMemoryBankService→  long-term cross-session memory
  PreloadMemoryTool        →  always load memory at turn start
  session state {}         →  inject state into agent instructions

DEPLOYMENT
  AdkApp(agent=...)        →  wrap agent for Vertex AI deployment
  client.agent_engines.create() → deploy to Vertex AI Agent Engine
  adk web                  →  local dev browser UI
  adk deploy agent_engine  →  CLI deployment to Agent Engine
  adk deploy cloud_run     →  CLI deployment to Cloud Run
  async_stream_query()     →  query deployed agent (streaming)
```

---

*Study tip: Focus on the three agent categories (LlmAgent vs workflow agents), the difference between session state and long-term memory, callback lifecycle hooks, and the deployment path through AdkApp → agent_engines.create() — these are the most exam-relevant areas for ADK on the GCP ML Engineer exam.*
