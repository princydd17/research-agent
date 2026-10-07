# Research Assistant Agent

An AI-powered research assistant agent built with LangGraph, LangChain, and Groq. The agent can search the web, read URLs, and synthesize information to answer research questions — with human-in-the-loop approval for URL access.

## Architecture

- **Framework**: LangGraph StateGraph with conditional edges and memory checkpointing
- **LLM**: Groq API (`qwen/qwen3.8-27b`)
- **Observability**: Langfuse tracing
- **Deployment**: FastAPI + Docker

## Project Structure
research-agent/
├── agent/
│ ├── state.py # Task 1 — AgentState TypedDict
│ ├── tools.py # Task 3 — Tool schemas (web_search, read_url, final_answer)
│ ├── nodes.py # Tasks 2,3,5,6,7 — Graph nodes (reasoning, tool use, approval, summarize)
│ ├── hooks.py # Task 5 — Pre/post tool execution hooks
│ ├── graph.py # Tasks 6-7 — LangGraph StateGraph with conditional routing
│ ├── mcp_client.py # Task 4 — MCP client integration
│ └── telemetry.py # Task 8 — Langfuse observability
├── eval/
│ └── judge.py # Task 9 — LLM-as-a-judge evaluation
├── server.py # FastAPI server (POST /query, POST /webhook, GET /health)
├── Dockerfile # Docker containerization
├── requirements.txt # Python dependencies
└── .gitignore


## Tasks Implemented

| Task | Description |
|------|-------------|
| 1 | Agent state definition with TypedDict |
| 2 | Reasoning node with Groq LLM |
| 3 | Tool integration (web search, URL reader, final answer) |
| 4 | MCP client for tool discovery |
| 5 | Pre/post tool execution hooks |
| 6 | Conditional edge routing (tool use, approval, summarization) |
| 7 | Memory and checkpointing with MemorySaver |
| 8 | Langfuse observability and tracing |
| 9 | LLM-as-a-judge evaluation |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install langchain
```

Create a `.env` file:

GROQ_API_KEY=your_groq_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_HOST=https://cloud.langfuse.com


## Running

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /query` — Send a research question (`{"query": "What is quantum computing?"}`)
- `POST /webhook` — Webhook endpoint for external integrations
- `GET /health` — Health check

## Example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is quantum computing?"}'
```

## Docker

```bash
docker build -t research-agent .
docker run -p 8000:8000 --env-file .env research-agent
```
