"""Tasks 2, 3, 5, 6, 7 — Graph nodes and conditional-edge functions."""

import os
import json
import time
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from agent.state import AgentState
from agent.tools import TOOL_SCHEMAS, execute_tool
from agent.hooks import pre_tool_hook, post_tool_hook

load_dotenv()

# ── LLM setup ────────────────────────────────────────────────────

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
    max_tokens=500,
)

SYSTEM_PROMPT = f"""You are a Research Assistant Agent.

Your job:
1. Understand the user's question.
2. Use tools to search the web and read pages.
3. Synthesize a grounded, cited answer.

Available tools (call ONE per turn as JSON):
{json.dumps(TOOL_SCHEMAS, indent=2)}

Respond with EXACTLY one JSON block:
{{"tool": "<name>", "args": {{...}}}}

When you have enough information, call the "final_answer" tool."""


# ── Nodes ─────────────────────────────────────────────────────────

def reasoning_node(state: AgentState) -> dict:
    """Ask the LLM to decide the next action or produce a final answer."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }


def tool_node(state: AgentState) -> dict:
    """Parse the LLM's JSON tool call, run hooks, execute, return result."""
    last_msg = state["messages"][-1]
    content = last_msg.content.strip()

    # Parse JSON tool call from the LLM response
    try:
        # Handle markdown-wrapped JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        call = json.loads(content)
        tool_name = call["tool"]
        tool_args = call.get("args", {})
    except (json.JSONDecodeError, KeyError):
        error_msg = f"Could not parse tool call from: {content[:200]}"
        return {"messages": [HumanMessage(content=f"[SYSTEM] {error_msg}")]}

    # Pre-tool hook (guardrail)
    check = pre_tool_hook(tool_name, tool_args)
    if not check["ok"]:
        blocked_msg = f"[TOOL BLOCKED] {tool_name}: {check['reason']}"
        return {"messages": [HumanMessage(content=blocked_msg)]}

    # Execute
    start = time.time()
    result = execute_tool(tool_name, tool_args)
    duration_ms = (time.time() - start) * 1000

    # Post-tool hook (logging)
    post_tool_hook(tool_name, tool_args, result, duration_ms)

    # Store result
    tool_results = list(state.get("tool_results", []))
    tool_results.append({"tool": tool_name, "args": tool_args, "result": result[:2000]})

    return {
        "messages": [HumanMessage(content=f"[TOOL RESULT: {tool_name}]\n{result[:2000]}")],
        "tool_results": tool_results,
    }


def human_approval_node(state: AgentState) -> dict:
    """Pause and ask for human approval in the terminal."""
    last_msg = state["messages"][-1]
    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP — Approval Required")
    print("=" * 60)
    print(f"Agent wants to act:\n{last_msg.content[:500]}")
    print("=" * 60)

    choice = input("Approve? (y/n): ").strip().lower()
    approved = choice in ("y", "yes")

    return {
        "human_approved": approved,
        "messages": [
            HumanMessage(
                content=f"[HUMAN] {'Approved' if approved else 'Rejected'} the action."
            )
        ],
    }


def summarize_node(state: AgentState) -> dict:
    """Compact early messages into a summary to keep context short."""
    messages = list(state["messages"])
    if len(messages) <= 10:
        return {}  # nothing to compact

    early = messages[:8]
    kept = messages[8:]

    summary_text = "Summary of prior steps:\n"
    for m in early:
        role = m.__class__.__name__.replace("Message", "")
        summary_text += f"- [{role}] {m.content[:120]}\n"

    return {
        "messages": [SystemMessage(content=summary_text)] + kept,
    }


# ── Conditional edges ────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """Decide the next node after reasoning."""
    last_msg = state["messages"][-1]
    content = last_msg.content.strip()

    # Safety: cap iterations
    if state.get("step_count", 0) > 15:
        return "end"

    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        call = json.loads(content)
        tool_name = call.get("tool", "")
    except (json.JSONDecodeError, KeyError):
        return "end"  # unparseable → end

    if tool_name == "final_answer":
        return "end"

    # Require human approval for read_url (opening external pages)
    if tool_name == "read_url":
        return "needs_approval"

    return "use_tool"


def route_after_approval(state: AgentState) -> str:
    """After human review, either proceed to tool or go back to reason."""
    if state.get("human_approved", False):
        return "use_tool"
    return "reason"


def should_summarize(state: AgentState) -> str:
    """Check if context compaction is needed before reasoning."""
    if len(state.get("messages", [])) > 10:
        return "summarize"
    return "reason"