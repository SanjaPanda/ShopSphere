from typing import TypedDict, Optional, Any, Literal


class AppState(TypedDict):
    """
    Shared state across all graphs.
    """

    # User Input
    user_query: str
    retry_count: int = 0
    max_retry_count: int = 3

    # classifier
    route: Literal["sql", "rag", "reject"]
    route_confidence: float
    route_reasoning: str
    allowed: bool

    # SQL
    schema: Optional[str]
    generated_sql: Optional[str]
    sql_result: Optional[Any]
    sql_error: Optional[str]
    executed_sql: Optional[str]    

    # RAG
    rewritten_question: Optional[str]
    retrieved_chunks: Optional[list[dict[str, Any]]]
    retrieval_scores: Optional[list[float]]


    # Final response
    answer: Optional[str]