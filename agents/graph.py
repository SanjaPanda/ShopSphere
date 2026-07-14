from langgraph.graph import StateGraph, END, START

from dotenv import load_dotenv

from shared.logging_config import setup_logger

from agents.state import AppState

load_dotenv()

from agents.sql.graph import sql_graph
from agents.rag.graph import rag_graph
from agents.classifier.nodes import (
    classifier_node,
    sql_guard_node,
    rag_guard_node,
    summary_node,
    reject_node,
)


logger = setup_logger("route")

builder = StateGraph(AppState)

builder.add_node("classifier", classifier_node)
builder.add_node("sql_guard", sql_guard_node)
builder.add_node("rag_guard", rag_guard_node)
builder.add_node("summary", summary_node)
builder.add_node("reject", reject_node)
builder.add_node("sql_agent", sql_graph)
builder.add_node("rag_agent", rag_graph)

builder.set_entry_point("classifier")


def route(state: AppState):
    logger.info("route determined as %s", state.get("route"))
    return state["route"]


builder.add_conditional_edges(
    "classifier",
    route,
    {
        "sql": "sql_guard",
        "rag": "rag_guard",
        "reject": "reject",

    },
)

builder.add_conditional_edges(
    "sql_guard",
    lambda state: "sql_agent" if state["allowed"] else "reject",
    {
        True: "sql_agent",
        False: "reject",
    },
)

builder.add_conditional_edges(
    "rag_guard",
    lambda state: "rag_agent" if state["allowed"] else "reject",
    {
        True: "rag_agent",
        False: "reject",
    },  
)

builder.add_edge("sql_agent", "summary")
builder.add_edge("rag_agent", "summary")

builder.add_edge("summary", END)
builder.add_edge("reject", END)

router_graph = builder.compile()