from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel, Field

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


VECTOR_DB_DIRECTORY = (
    Path(__file__).parent.parent.parent / "data" / "vector_db"
)


def _build_where_clause(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert metadata filters into the Chroma-compatible where clause."""

    cleaned_filters = {
        key: value
        for key, value in (filters or {}).items()
        if value is not None
    }

    if not cleaned_filters:
        return None

    if len(cleaned_filters) == 1:
        key, value = next(iter(cleaned_filters.items()))
        return {key: value}

    return {"$and": [{key: value} for key, value in cleaned_filters.items()]}


class Retriever:
    """Wrapper around a Chroma vector store."""

    def __init__(
        self,
        persist_directory: str | Path = VECTOR_DB_DIRECTORY,
        k: int = 5,
    ) -> None:

        self.k = k

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self.vector_store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=self.embedding_model,
        )

    def retrieve(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> list[Document]:
        """
        Retrieve documents from the vector store.
        """

        where_clause = _build_where_clause(filters)

        return self.vector_store.similarity_search(
            query=query,
            filter=where_clause,
            k=k or self.k,
        )

    def retrieve_with_scores(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        k: int | None = None,
    ) -> tuple[list[Document], list[float]]:
        """
        Retrieve documents and similarity scores.
        """

        where_clause = _build_where_clause(filters)

        results = self.vector_store.similarity_search_with_score(
            query=query,
            filter=where_clause,
            k=k or self.k,
        )

        documents = [doc for doc, _ in results]
        scores = [score for _, score in results]

        return documents, scores


retriever = Retriever()


@tool(args_schema=RetrievalInput)
def search_documentation(
    query: str,
    latest_only: bool = False,
    k: int = 10,
):
    """
    Search technical documentation using semantic search and metadata filters.
    """

    filters = {}

    if latest_only:
        filters["status"] = "current"

    docs, scores = retriever.retrieve_with_scores(
        query=query,
        filters=filters,
        k=k,
    )

    return {
        "documents": docs,
        "scores": scores,
    }


if __name__ == "__main__":

    docs, scores = retriever.retrieve_with_scores(
        query="strategic vendor performance review frequency"
    )
    print(VECTOR_DB_DIRECTORY)

    for doc, score in zip(docs, scores):

        print("=" * 80)
        print(f"Score: {score:.4f}")
        print(doc.metadata)
        print(doc.page_content[:500])