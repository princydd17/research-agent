"""Task 8 — Langfuse observability."""

import os
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler

load_dotenv()


def get_langfuse_handler(session_id: str = "default") -> CallbackHandler:
    """Return a Langfuse callback handler for LangChain tracing."""
    return CallbackHandler()
