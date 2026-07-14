from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_DIRECTORY = Path(__file__).parent.parent.parent / "data" / "vector_db"

class Embedder:

    def __init__(self):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

    def create_vector_store(
        self,
        chunks,
        persist_directory=VECTOR_DB_DIRECTORY,
    ):

        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=persist_directory,
        )