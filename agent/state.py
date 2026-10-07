"""Task 1 — State definition for the Research Agent."""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    tool_results: list[dict]
    human_approved: bool
    final_answer: str
    step_count: int
