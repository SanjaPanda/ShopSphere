from langgraph.graph import StateGraph, END

from agents.state import AppState
from agents.sql.nodes import (
    retrieve_schema,
    generate_sql,
    summarize,
    sql_corrector,
    execute_generated_sql,
)



builder = StateGraph(AppState)

builder.add_node("retrieve_schema", retrieve_schema)
builder.add_node("generate_sql", generate_sql)
builder.add_node("execute_generated_sql", execute_generated_sql)
builder.add_node("sql_corrector", sql_corrector)
builder.add_node("summarize", summarize)

builder.set_entry_point("retrieve_schema")

builder.add_edge("retrieve_schema", "generate_sql")
builder.add_edge("generate_sql", "execute_generated_sql")




def router(state):
    MAX_RETRIES = state.get("max_retry_count", 3)

    # Success
    if state.get("sql_error") is None:
        return "success"

    # Retry if we still have attempts left
    if state.get("retry_count", 0) < MAX_RETRIES:
        return "error"

    # Give up
    return "failed"


builder.add_conditional_edges(
    "execute_generated_sql",
    router,
    {
        "success": "summarize",
        "error": "sql_corrector",
        "failed": "summarize",
    },
)

builder.add_edge("sql_corrector",   "execute_generated_sql")
builder.add_edge("summarize", END)

sql_graph = builder.compile()