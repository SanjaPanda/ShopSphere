from langgraph.graph import StateGraph, END

from agents.state import AppState
from agents.classifier.nodes import classifier_node

builder = StateGraph(AppState)

builder.add_node("classify", classifier_node)

builder.set_entry_point("classify")

builder.add_edge("classify", END)

classifier_graph = builder.compile()