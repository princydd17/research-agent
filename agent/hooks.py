"""Task 5 — Pre-tool and post-tool hooks (guardrails + logging)."""

import time
import json
from datetime import datetime, timezone


def pre_tool_hook(tool_name: str, args: dict) -> dict:
    """
    Validate / gate a tool call BEFORE execution.
    Returns {"ok": True} to proceed or {"ok": False, "reason": "..."} to block.
    """
    # Block empty search queries
    if tool_name == "web_search":
        query = args.get("query", "").strip()
        if not query:
            return {"ok": False, "reason": "Empty search query."}

    # Block localhost / private URLs
    if tool_name == "read_url":
        url = args.get("url", "")
        if any(blocked in url for blocked in ["localhost", "127.0.0.1", "0.0.0.0"]):
            return {"ok": False, "reason": "Blocked: localhost URLs are not allowed."}
        if not url.startswith(("http://", "https://")):
            return {"ok": False, "reason": "URL must start with http:// or https://."}

    return {"ok": True}


def post_tool_hook(tool_name: str, args: dict, result: str, duration_ms: float):
    """Log tool execution details after the call completes."""
    preview = result[:200] if isinstance(result, str) else str(result)[:200]
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "args": args,
        "result_preview": preview,
        "duration_ms": round(duration_ms, 2),
    }
    print(f"[TOOL LOG] {json.dumps(log_entry, indent=2)}")
