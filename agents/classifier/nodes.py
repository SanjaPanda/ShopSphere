from pydantic import BaseModel, Field
from typing import Literal

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from dotenv import load_dotenv
import os

from shared.logging_config import setup_logger

load_dotenv()

class RouteDecision(BaseModel):

    route: Literal["sql", "rag", "reject"]

    route_confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence in routing"
    )

    route_reasoning: str

class SQLGuard(BaseModel):

    allowed: bool
    reason: str

class RAGGuard(BaseModel):

    allowed: bool
    reason: str

class Summary(BaseModel):

    answer: str

    grounded: bool

    citations: list[str]

    reasoning: str


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)




def classifier_node(state):
    logger = setup_logger("classifier_node")
    logger.info("classifier_node started")

    SYSTEM_PROMPT = """
            You are the routing agent for ShopSphere.

            ShopSphere ONLY answers questions related to:

            - products
            - customers
            - orders
            - payments
            - shipments
            - inventory
            - discounts
            - stores
            - documentation
            - APIs
            - architecture
            - deployment
            - troubleshooting
            - user guides

            Route using these rules.

            SQL:
            Questions asking for data, numbers, analytics, counts,
            filters, aggregations, rankings or customer/order information.

            Examples:
            - How many orders yesterday?
            - Top selling products
            - Revenue this month
            - Orders by customer

            RAG:
            Questions asking for documentation.

            Examples:
            - Explain authentication
            - How does checkout work?
            - What APIs exist?
            - Deployment guide
            - Troubleshooting payment failures

            Reject:
            Anything unrelated.

            Examples:
            - Write a poem
            - Tell me a joke
            - Weather
            - Politics
            - Movies

            Return ONLY structured output.
            """
    
    structured_llm = llm.with_structured_output(RouteDecision).with_config(run_name="Classifier")
    response = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=state["user_query"])
        ]
    )

    logger.info("classifier_node routed to %s with confidence %.2f", response.route, response.route_confidence)

    return {
        "route": response.route,
        "route_confidence": response.route_confidence,
        "route_reasoning": response.route_reasoning,
    }

def reject_node(state):
    logger = setup_logger("reject_node")
    logger.info("reject_node triggered")

    return {
        "final_answer": (
            "I'm designed to answer questions about ShopSphere's "
            "business data and technical documentation only."
        )
    }

def sql_guard_node(state):
    logger = setup_logger("sql_guard_node")
    logger.info("sql_guard_node started for query: %s", state.get("user_query"))

    SYSTEM_PROMPT = """
            Verify whether this question is suitable for SQL generation.

            Reject if:

            - requests source code
            - asks documentation
            - asks architecture
            - asks deployment
            - asks opinion
            - is unrelated

            Allow only if the answer can be obtained from database tables.
                        """

    structured_llm = llm.with_structured_output(SQLGuard).with_config(run_name="SQL_Guard")
    response = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User query: {state['user_query']}")
        ]
    )

    return {
        "allowed": response.allowed,
        "reason": response.reason,
    }

def rag_guard_node(state):
    logger = setup_logger("rag_guard_node")
    logger.info("rag_guard_node started for query: %s", state.get("user_query"))

    SYSTEM_PROMPT = """
            Allow ONLY questions that can be answered using
            ShopSphere documentation.

            Reject:

            - analytics
            - revenue
            - counts
            - database questions
            - unrelated questions
            """

    structured_llm = llm.with_structured_output(RAGGuard).with_config(run_name="RAG_Guard")
    response = structured_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"User query: {state['user_query']}")
        ]
    )

    return {
        "allowed": response.allowed,
        "reason": response.reason,
    }

def summary_node(state):
    logger = setup_logger("summary_node")
    logger.info("summary_node started for route=%s", state.get("route"))

    if state.get("route") == "sql":
        AI_Message = f"SQL: {state['executed_sql']}\nSQL Result: {state['sql_result']}"
    elif state.get("route") == "rag":
        AI_Message = f"Retrived Chunks: {state['retrieved_chunks']}\nretrieval scores: {state['retrieval_scores']}"
    else:
        AI_Message = "No answer generated."

    SYSTEM_PROMPT = """
            You are the final answer generator.

            Rules:

            Only use the supplied evidence.

            Never invent numbers.

            If evidence is insufficient:

            - say you don't know
            - do not guess

            If PII or sensitive information is present in the evidence, redact it.

            Return structured output.
            """

    summary_llm = llm.with_structured_output(Summary).with_config(run_name="Summary")
    response = summary_llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),            
            HumanMessage(content=state.get("user_query", "")),
            AIMessage(content=AI_Message),
        ]
    )

    logger.info("summary_node generated answer with grounded=%s", response.grounded)

    return {
        "answer": response.answer,
        "grounded": response.grounded,
        "citations": response.citations,
        "reasoning": response.reasoning,
    }


