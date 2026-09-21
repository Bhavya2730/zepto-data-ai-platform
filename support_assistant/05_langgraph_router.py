from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from retrieval import retrieve


class AssistantState(TypedDict):
    query: str
    intent: str
    answer: str
    sources: list[str]


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

    answer = f"Based on the retrieved context: {snippet}"

    sources = [result["chunk_id"] for result in results]

    return {
        "answer": answer,
        "sources": sources,
    }


def direct_answer(state: AssistantState):
    return {
        "answer": "I can only answer questions about Zepto policies right now.",
        "sources": [],
    }


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
    app = build_graph()

    policy_query = {
        "query": "What is the delivery fee?",
        "intent": "",
        "answer": "",
        "sources": [],
    }

    policy_result = app.invoke(policy_query)

    print("\nPolicy question:")
    print(policy_result)

    general_query = {
        "query": "What is the capital of France?",
        "intent": "",
        "answer": "",
        "sources": [],
    }

    general_result = app.invoke(general_query)

    print("\nGeneral question:")
    print(general_result)

    if policy_result["intent"] != "policy_question":
        raise AssertionError("Policy query was not classified correctly.")

    if general_result["intent"] != "general_question":
        raise AssertionError("General query was not classified correctly.")

    if not policy_result["sources"]:
        raise AssertionError("Policy query returned no sources.")

    if general_result["sources"]:
        raise AssertionError("General query should have no sources.")

    print("\nStage 3.5 LangGraph verification passed.")


if __name__ == "__main__":
    main()