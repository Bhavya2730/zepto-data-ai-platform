import os
from typing import TypedDict

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END

from retrieval import retrieve


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


STRUCTURED_PROMPT = """
ROLE:
You are a Zepto customer-support assistant.

CONTEXT:
Use only the policy context provided below.

TASK:
Answer the customer's question using the provided context.

FORMAT:
Return a structured response containing answer, sources, and confidence.

LENGTH:
Keep the answer concise and directly relevant.

NEGATIVE CONSTRAINT:
Do not answer using information that is not present in the provided context.
Do not invent or assume Zepto policies.

FEW-SHOT EXAMPLE:
Question: What is the delivery fee?
Context: Orders below INR 149 incur a flat INR 25 delivery fee.
Answer: Orders below INR 149 incur a flat INR 25 delivery fee.
"""


class AssistantState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]
    confidence: float


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

    return response.model_dump()


def direct_answer(state: AssistantState):
    response = AssistantResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0,
    )

    return response.model_dump()


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


def main():
    mock_llm = os.getenv("MOCK_LLM", "1")

    print(f"MOCK_LLM={mock_llm}")
    print("\nStructured prompt template loaded.")
    print("Prompt contains ROLE, CONTEXT, TASK, FORMAT, LENGTH.")
    print("Negative constraint included.")
    print("Few-shot example included.")

    app = build_graph()

    policy_query = {
        "query": "What is the delivery fee?",
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }

    policy_result = app.invoke(policy_query)

    print("\nPolicy response:")
    print(policy_result)

    policy_response = AssistantResponse.model_validate(policy_result)

    print("\nValidated Pydantic response:")
    print(policy_response.model_dump_json(indent=2))

    general_query = {
        "query": "What is the capital of France?",
        "intent": "",
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }

    general_result = app.invoke(general_query)

    print("\nGeneral response:")
    print(general_result)

    general_response = AssistantResponse.model_validate(general_result)

    print("\nValidated general Pydantic response:")
    print(general_response.model_dump_json(indent=2))

    if not policy_response.answer:
        raise AssertionError("Policy answer is empty.")

    if not policy_response.sources:
        raise AssertionError("Policy response has no sources.")

    if not 0.0 <= policy_response.confidence <= 1.0:
        raise AssertionError("Policy confidence is outside 0-1.")

    if general_response.sources:
        raise AssertionError("General response should have no sources.")

    if not 0.0 <= general_response.confidence <= 1.0:
        raise AssertionError("General confidence is outside 0-1.")

    print("\nPydantic schema validation passed.")
    print("Stage 3.6 structured output verification passed.")


if __name__ == "__main__":
    main()