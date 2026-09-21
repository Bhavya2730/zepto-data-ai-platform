import os
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from .retrieval import retrieve


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class AssistantState(TypedDict):
    query: str
    intent: str
    response: AssistantResponse | None


POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


def classify_intent(state: AssistantState):
    query = state["query"].lower()

    intent = "general_question"

    for keyword in POLICY_KEYWORDS:
        if keyword in query:
            intent = "policy_question"
            break

    return {"intent": intent}


def retrieve_and_answer(state: AssistantState):
    results = retrieve(state["query"], top_k=3)

    top_chunk = results[0]

    snippet = top_chunk["text"][:200]

    response = AssistantResponse(
        answer=f"Based on the retrieved context: {snippet}",
        sources=[result["chunk_id"] for result in results],
        confidence=1.0,
    )

    return {"response": response}


def direct_answer(state: AssistantState):
    response = AssistantResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0,
    )

    return {"response": response}


def route_after_classification(state: AssistantState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


def build_graph():
    graph = StateGraph(AssistantState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.add_edge(START, "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )

    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()


MOCK_LLM = os.getenv("MOCK_LLM", "1")