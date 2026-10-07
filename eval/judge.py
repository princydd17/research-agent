"""Task 9 (Part 1) — LLM-as-a-judge evaluation for Error Recovery."""

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# ── Simulated historical traces ──────────────────────────────────

TRACES = [
    {
        "id": "trace-001",
        "steps": [
            "web_search('quantum computing 2025')",
            "read_url('https://example.com/quantum') -> 200 OK",
            "final_answer('Quantum computing advances include...')"
        ]
    },
    {
        "id": "trace-002",
        "steps": [
            "web_search('') -> ERROR: empty query",
            "web_search('AI safety research') -> results",
            "final_answer('AI safety research focuses on...')"
        ]
    },
    {
        "id": "trace-003",
        "steps": [
            "web_search('climate data') -> results",
            "read_url('http://localhost:8080/data') -> BLOCKED by guardrail",
            "read_url('https://climate.nasa.gov') -> 200 OK",
            "final_answer('NASA climate data shows...')"
        ]
    },
    {
        "id": "trace-004",
        "steps": [
            "web_search('LangGraph tutorial') -> results",
            "read_url('https://docs.example.com') -> TIMEOUT",
            "read_url('https://docs.example.com') -> TIMEOUT",
            "final_answer('I was unable to retrieve the page...')"
        ]
    },
    {
        "id": "trace-005",
        "steps": [
            "web_search('Python async patterns') -> results",
            "final_answer('Common async patterns include...')"
        ]
    },
    {
        "id": "trace-006",
        "steps": [
            "UNPARSEABLE OUTPUT: 'Let me think about this...'",
            "web_search('neural network architectures') -> results",
            "read_url('https://arxiv.org/abs/1234') -> 200 OK",
            "final_answer('Key neural architectures include...')"
        ]
    },
    {
        "id": "trace-007",
        "steps": [
            "web_search('stock prices') -> results",
            "read_url('https://finance.example.com') -> 403 Forbidden",
            "web_search('stock prices free API') -> results",
            "read_url('https://free-api.example.com') -> 200 OK",
            "final_answer('Current stock data from free sources...')"
        ]
    },
    {
        "id": "trace-008",
        "steps": [
            "web_search('machine learning') -> results",
            "read_url('https://example.com/ml') -> 200 OK",
            "read_url('https://example.com/dl') -> 200 OK",
            "read_url('https://example.com/rl') -> 200 OK",
            "read_url('https://example.com/nlp') -> 200 OK",
            "read_url('https://example.com/cv') -> 200 OK",
            "(hit step limit at 15)"
        ]
    },
    {
        "id": "trace-009",
        "steps": [
            "web_search('best restaurants NYC') -> results",
            "HUMAN REJECTED read_url('https://sketchy-site.com')",
            "read_url('https://nytimes.com/food') -> 200 OK",
            "final_answer('Top NYC restaurants include...')"
        ]
    },
    {
        "id": "trace-010",
        "steps": [
            "web_search('renewable energy stats') -> ERROR: network timeout",
            "web_search('renewable energy stats') -> results",
            "read_url('https://iea.org/renewables') -> 200 OK",
            "final_answer('Renewable energy now accounts for...')"
        ]
    },
]


JUDGE_PROMPT = """You are an expert evaluator. Score the following agent trace
on **Error Recovery** (1-5 scale):

1 = Agent crashed or gave up immediately on first error.
2 = Agent acknowledged error but didn't recover effectively.
3 = Agent partially recovered but the final answer was weak.
4 = Agent recovered from errors and delivered a decent answer.
5 = Agent handled all errors gracefully with an excellent final answer.

Trace ID: {trace_id}
Steps:
{steps}

Respond with ONLY a JSON object:
{{"score": <1-5>, "reasoning": "<one sentence>"}}
"""


def run_eval():
    """Score each trace using the LLM as a judge."""
    judge_llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )

    results = []
    for trace in TRACES:
        steps_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(trace["steps"]))
        prompt = JUDGE_PROMPT.format(trace_id=trace["id"], steps=steps_text)

        response = judge_llm.invoke(prompt)
        try:
            score_data = json.loads(response.content)
        except json.JSONDecodeError:
            score_data = {"score": "N/A", "reasoning": response.content[:200]}

        results.append({"trace_id": trace["id"], **score_data})
        print(f"  {trace['id']}: score={score_data.get('score')} — {score_data.get('reasoning', '')[:80]}")

    return results


if __name__ == "__main__":
    print("Running LLM-as-a-Judge evaluation on Error Recovery...\n")
    all_results = run_eval()
    print(f"\nDone. Evaluated {len(all_results)} traces.")
    avg = sum(r["score"] for r in all_results if isinstance(r["score"], int)) / max(
        sum(1 for r in all_results if isinstance(r["score"], int)), 1
    )
    print(f"Average Error Recovery score: {avg:.1f} / 5")