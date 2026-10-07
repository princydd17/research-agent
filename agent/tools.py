"""Task 3 — Tool schemas, implementations, and routing."""

import json
import httpx
from duckduckgo_search import DDGS

# ── JSON schemas exposed to the LLM ──────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "web_search",
        "description": "Search the web with DuckDuckGo and return top results.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (default 5)."
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_url",
        "description": "Fetch the text content of a URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to read."
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "final_answer",
        "description": "Return the final, grounded answer to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer text."
                }
            },
            "required": ["answer"]
        }
    }
]


# ── Tool implementations ─────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo and return a list of {title, href, body}."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def read_url(url: str) -> str:
    """Fetch a URL and return up to 4 000 chars of its text."""
    resp = httpx.get(url, follow_redirects=True, timeout=15)
    resp.raise_for_status()
    return resp.text[:4000]


def final_answer(answer: str) -> str:
    """Pass-through that marks the answer as final."""
    return answer


# ── Registry & router ────────────────────────────────────────────

TOOL_REGISTRY = {
    "web_search": web_search,
    "read_url": read_url,
    "final_answer": final_answer,
}


def execute_tool(name: str, args: dict) -> str:
    """Look up a tool by name and call it with the given args."""
    func = TOOL_REGISTRY.get(name)
    if func is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = func(**args)
        return json.dumps(result) if not isinstance(result, str) else result
    except Exception as exc:
        return json.dumps({"error": str(exc)})
