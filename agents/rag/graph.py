from langgraph.graph import StateGraph, END

from agents.state import AppState
from agents.classifier.graph import classifier_graph
from agents.rag.nodes import (
    retrieve_node,
    rerank_node,
    rewrite_node,
    generate_answer_node,
    no_answer_node,
)


def retrieval_router(state: AppState) -> str:
    """
    Decide what to do after retrieval.
    """

    if state["retrieved_chunks"]:
        return "rerank"

    if state["retry_count"] < 2:
        return "rewrite"

    return "no_answer"


builder = StateGraph(AppState)

# ------------------------
# Nodes
# ------------------------

builder.add_node("classifier", classifier_graph)

builder.add_node("retrieve", retrieve_node)

builder.add_node("rerank", rerank_node)

builder.add_node("rewrite", rewrite_node)

builder.add_node("generate_answer", generate_answer_node)

builder.add_node("no_answer", no_answer_node)

# ------------------------
# Edges
# ------------------------

builder.set_entry_point("classifier")

builder.add_edge("classifier", "retrieve")

builder.add_conditional_edges(
    "retrieve",
    retrieval_router,
    {
        "rerank": "rerank",
        "rewrite": "rewrite",
        "no_answer": "no_answer",
    },
)

builder.add_edge("rewrite", "retrieve")

builder.add_edge("rerank", "generate_answer")

builder.add_edge("generate_answer", END)

builder.add_edge("no_answer", END)

rag_graph = builder.compile()