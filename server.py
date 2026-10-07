"""Task 9 (Part 2) — FastAPI deployment with /query and /webhook endpoints."""

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from agent.graph import build_graph
from agent.telemetry import get_langfuse_handler

load_dotenv()

app = FastAPI(title="Research Agent API", version="1.0.0")

# Build graph once at startup
agent = build_graph()


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"


class QueryResponse(BaseModel):
    answer: str
    steps: int


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "research-agent"}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Run the research agent on a user query."""
    handler = get_langfuse_handler(req.session_id)

    result = agent.invoke(
        {
            "messages": [HumanMessage(content=req.query)],
            "query": req.query,
            "tool_results": [],
            "human_approved": True,  # auto-approve in API mode
            "final_answer": "",
            "step_count": 0,
        },
        config={
            "configurable": {"thread_id": req.session_id},
            "callbacks": [handler],
        },
    )

    # Extract the last message as the answer
    answer = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content"):
            answer = msg.content
            break

    return QueryResponse(
        answer=answer,
        steps=result.get("step_count", 0),
    )


@app.post("/webhook")
async def webhook(request: Request):
    """Generic webhook endpoint for integrations."""
    body = await request.json()
    print(f"[WEBHOOK] Received: {body}")
    return {"received": True}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
