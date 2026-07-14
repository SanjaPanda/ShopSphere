import json
import os
from pydantic import BaseModel, Field, schema
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

from shared.logging_config import setup_logger

from .retriever import search_documentation

from .reranker import Reranker
from pydantic import BaseModel, Field
load_dotenv()

reranker = Reranker()

class RetrievalInput(BaseModel):

    query: str = Field(description="Optimized search query.")

    latest_only: bool = Field(
        default=False,
        description="Whether only current/latest documentation should be searched."
    )

    k: int = Field(
        default=10,
        description="Number of documents to retrieve."
    )

class RewriteQuestion(BaseModel):
    rewritten_question: str = Field(description="Rewritten user question optimized for semantic search.")


class AnswerResponse(BaseModel):
    answer: str = Field( description="Answer generated using the provided context.")

    confidence: float = Field(description="Confidence score between 0 and 1.")

    sources: list[str] = Field(
        default_factory=list,
        description="Documents used to generate the answer."
    )


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def retrieve_node(state):
    """
    Retrieve relevant chunks from Chroma.
    """
    logger = setup_logger("retrieve_node")
    logger.info("retrieve_node started")

    retriver_llm = llm.with_structured_output(RetrievalInput).with_config(run_name="Retriever")

    if state["retry_count"] > 0:
        # If this is a retry, rewrite the question to improve retrieval.
        user_query = state.get("rewritten_question")
    
    else:
        user_query = state.get("user_query")

    SYSTEM_PROMPT = f"""
        You are a documentation retrieval assistant.

        Your task is to determine how to search the documentation.

        Always use the search_documentation tool.

        If the user requests:
        - latest version → latest_only=True

        Do not answer the question yourself.
         """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)]

    logger.info("retrieve_node user_query=%s retry_count=%s", user_query, state.get("retry_count", 0))

    response = retriver_llm.invoke(messages)
    logger.info("retrieve_node retrieval tool input=%s", response.model_dump())

    result = search_documentation.invoke(response.model_dump())
    logger.info("retrieve_node documents_count=%s scores_count=%s", len(result.get("documents", [])), len(result.get("scores", [])))

    return {
        "retrieved_chunks": result["documents"],
        "retrieval_scores": result["scores"],
    }


def rerank_node(state):
    """
    Rerank retrieved documents using a CrossEncoder.
    """
    logger = setup_logger("rerank_node")
    logger.info("rerank_node started")

    if state["retry_count"] > 0:
        # If this is a retry, rewrite the question to improve retrieval.
        user_query = state.get("rewritten_question")
    
    else:
        user_query = state.get("user_query")

    logger.info("rerank_node user_query=%s retrieved_chunk_count=%s", user_query, len(state.get("retrieved_chunks", [])))

    reranked_docs, rerank_scores = reranker.rerank(
        query=user_query,
        documents=state["retrieved_chunks"],
        top_k=5,
    )

    logger.info("rerank_node reranked_docs_count=%s rerank_scores=%s", len(reranked_docs), rerank_scores)

    return {
        "retrieved_chunks": reranked_docs,
        "rerank_scores": rerank_scores,
    }


def rewrite_node(state):
    """
    Rewrite the query to improve retrieval.
    """
    logger = setup_logger("rewrite_node")
    logger.info("rewrite_node started")

    rewrite_llm = llm.with_structured_output(RewriteQuestion).with_config(run_name="Rewrite")

    user_query = state.get("user_query")

    SYSTEM_PROMPT = f"""
        You are an expert search query rewriter.

        Your task is to rewrite the user's question so that it is easier for a vector database to retrieve relevant documentation.

        Rules:
        - Preserve the original intent.
        - Do not answer the question.
        - Expand abbreviations where appropriate.
        - Use technical terminology if applicable.
        - Return only the rewritten question.
         """

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)]

    response = rewrite_llm.invoke(messages)

    logger.info("rewrite_node original_question=%s rewritten_question=%s", user_query, response.rewritten_question)

    return {
        "rewritten_question": response.rewritten_question,
        "retry_count": state["retry_count"] + 1,
    }


def generate_answer_node(state):
    """
    Generate answer using retrieved context.
    """
    logger = setup_logger("generate_answer_node")
    logger.info("generate_answer_node started")

    summary_llm = llm.with_structured_output(AnswerResponse).with_config(run_name="RAG_Summary")

    logger.info("generate_answer_node retrieved_chunk_count=%s", len(state.get("retrieved_chunks", [])))

    context = "\n\n".join(
        doc.page_content
        for doc in state["retrieved_chunks"]
    )

    logger.info("generate_answer_node context_length=%s", len(context))
    user_query = state.get("user_query")

    SYSTEM_PROMPT = f"""You are an AI assistant answering questions using ONLY the provided documentation.

                Instructions:
                - Answer only from the supplied context.
                - Do not make up information.
                - If the answer cannot be found in the context, say so politely.
                - Provide a concise but complete answer.

                Context:
                {context}
            """
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_query)]

    response = summary_llm.invoke(messages)

    logger.info("generate_answer_node answer=%s", response.answer)

    return {
        "answer": response.answer
    }


def no_answer_node(state):
    """
    Return a graceful response when nothing relevant is found.
    """
    logger = setup_logger("no_answer_node")
    logger.info("no_answer_node triggered")

    return {
        "answer": (
            "I'm sorry, but I couldn't find any relevant "
            "information in the available documentation "
            "to answer your question."
        )
    }