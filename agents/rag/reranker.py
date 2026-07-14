from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


class Reranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> tuple[list[Document], list[float]]:

        if not documents:
            return [], []

        # Create query-document pairs
        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        # Higher score = more relevant
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        ranked_documents = [
            doc
            for doc, _ in ranked[:top_k]
        ]
        ranked_scores = [
            score
            for _, score in ranked[:top_k]
        ]

        return ranked_documents, ranked_scores