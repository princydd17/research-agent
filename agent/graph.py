"""Tasks 6-7 — Assemble the LangGraph state machine."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agent.state import AgentState
from agent.nodes import (
    reasoning_node,
    tool_node,
    human_approval_node,
    summarize_node,
    should_continue,
    route_after_approval,
    should_summarize,
)


def build_graph():
    """Build and compile the full agent graph with checkpointing."""

    graph = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────
    graph.add_node("summarize", summarize_node)
    graph.add_node("reason", reasoning_node)
    graph.add_node("tools", tool_node)
    graph.add_node("human_approval", human_approval_node)

    # ── Entry point → check if compaction needed ──────────────
    graph.set_entry_point("summarize")

    # After summarize → always go to reason
    graph.add_conditional_edges(
        "summarize",
        should_summarize,
        {
            "summarize": "reason",   # already summarized, go reason
            "reason": "reason",       # no summarization needed
        },
    )

    # After reasoning → decide next step
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "use_tool": "tools",
            "needs_approval": "human_approval",
            "end": END,
        },
    )

    # After tool execution → loop back to summarize (which routes to reason)
    graph.add_edge("tools", "summarize")

    # After human approval → proceed or loop back
    graph.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {
            "use_tool": "tools",
            "reason": "reason",
        },
    )

    # ── Compile with checkpointer ─────────────────────────────
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


# ── Run from terminal ─────────────────────────────────────────────

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    app = build_graph()

    print("Research Agent ready. Type your question (or 'quit' to exit).\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break

        result = app.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "query": user_input,
                "tool_results": [],
                "human_approved": False,
                "final_answer": "",
                "step_count": 0,
            },
            config={"configurable": {"thread_id": "main-thread"}},
        )

        # Print the last AI message as the answer
        for msg in reversed(result["messages"]):
            if hasattr(msg, "content"):
                print(f"\nAgent: {msg.content}\n")
                break
