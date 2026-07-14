import os

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)


class DocumentMetadata(BaseModel):

    title: str | None = Field(
        default=None,
        description="Title of the document."
    )

    category: str | None = Field(
        default=None,
        description="Broad category the document belongs to."
    )

    version: str | None = Field(
        default=None,
        description="Document version if mentioned."
    )

    release_date: str | None = Field(
        default=None,
        description="Release or publication date."
    )

    status: str | None = Field(
        default="unknown",
        description="current, deprecated, draft or unknown"
    )

    summary: str | None = Field(
        default=None,
        description="One sentence summary."
    )



DOCUMENT_METADATA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are extracting metadata from technical documentation.

Extract:

- title
- product
- version
- release_date
- status
- summary

Rules:

- If a field is unavailable, return null.
- Infer status only if the document explicitly indicates it is deprecated,
  obsolete, archived or current.
- Do not guess.
"""
        ),
        (
            "human",
            """
Document:

{document}
"""
        )
    ]
)

class MetadataExtractor:

    def __init__(self):

        self.chain = (
            DOCUMENT_METADATA_PROMPT
            | llm.with_structured_output(DocumentMetadata)
        )

    def extract(
        self,
        markdown: str,
    ) -> DocumentMetadata:

        # First 1500-2000 chars are usually enough
        sample = markdown[:2000]

        return self.chain.invoke(
            {
                "document": sample
            }
        )