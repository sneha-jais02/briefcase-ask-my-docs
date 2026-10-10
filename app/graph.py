import uuid
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app import answering, embeddings, reranking, store
from app.config import get_settings


class RAGState(TypedDict, total=False):
    question: str
    history: list[dict[str, str]]
    workspace_id: str
    jurisdiction: str | None
    hits: list[dict[str, Any]]
    result: dict[str, Any]


def build_graph():
    def retrieve(state: RAGState) -> dict[str, Any]:
        vector = embeddings.embed_texts([state["question"]])[0]
        hits = store.hybrid_search(
            uuid.UUID(state["workspace_id"]),
            state["question"],
            vector,
            state.get("jurisdiction"),
        )
        ranked = reranking.rerank(state["question"], hits)
        return {"hits": ranked[: get_settings().retrieval_results]}

    def answer(state: RAGState) -> dict[str, Any]:
        return {"result": answering.grounded_answer(state["question"], state["hits"], state.get("history", []))}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
