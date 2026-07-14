from pathlib import Path
from uuid import uuid4

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

class DocumentChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        # RecursiveCharacterTextSplitter preserves the metadata
        # (h1, h2, h3, etc.) from MarkdownHeaderTextSplitter.
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def chunk(
        self,
        markdown: str,
        filename: str | Path,
        document_metadata: dict | None = None,
    ):

        sections = self.header_splitter.split_text(markdown)

        chunks = self.recursive_splitter.split_documents(
            sections
        )

        return self._add_metadata(
            chunks,
            filename,
            document_metadata,
        )

    def _add_metadata(
        self,
        chunks,
        filename: str | Path,
        document_metadata: dict | None = None,
    ):

        filename_path = Path(filename)

        for chunk in chunks:
            chunk.metadata.update(
                {
                    "chunk_id": str(uuid4()),
                    "document_name": filename_path.name,
                    "source": str(filename_path),
                }
            )

            if document_metadata is None:
                continue

            if isinstance(document_metadata, dict):
                chunk.metadata.update(
                    {
                        "title": document_metadata.get("title"),
                        "category": document_metadata.get("category") or document_metadata.get("product"),
                        "version": document_metadata.get("version"),
                        "release_date": document_metadata.get("release_date"),
                        "status": document_metadata.get("status"),
                        "summary": document_metadata.get("summary"),
                    }
                )
            else:
                chunk.metadata.update(
                    {
                        "title": getattr(document_metadata, "title", None),
                        "category": getattr(document_metadata, "category", None)
                        or getattr(document_metadata, "product", None),
                        "version": getattr(document_metadata, "version", None),
                        "release_date": getattr(document_metadata, "release_date", None),
                        "status": getattr(document_metadata, "status", None),
                        "summary": getattr(document_metadata, "summary", None),
                    }
                )

        return chunks