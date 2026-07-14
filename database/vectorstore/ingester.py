import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
CURRENT_DIR = Path(__file__).resolve().parent
for path in (str(ROOT_DIR), str(CURRENT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from docling.document_converter import DocumentConverter

try:
    from .chunker import DocumentChunker
    from .embedder import Embedder
    from .metadata_extractor import MetadataExtractor
except ImportError:
    from chunker import DocumentChunker
    from embedder import Embedder
    from metadata_extractor import MetadataExtractor

from shared.logging_config import setup_logger

# ------------------------------------------------------------------------------
# Logger
# ------------------------------------------------------------------------------

logger = setup_logger(__file__)

DATA_DIRECTORY = Path(__file__).parent.parent.parent / "data" / "pdfs"


def load_pdf(pdf_path: Path, converter: DocumentConverter) -> str:

    logger.info("Loading PDF: %s", pdf_path.name)

    try:
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
    except Exception:
        logger.exception("Failed to load PDF: %s", pdf_path.name)
        raise

    logger.info("Successfully loaded PDF: %s", pdf_path.name)
    return markdown


def process_pdf(
    pdf_path: Path,
    chunker: DocumentChunker,
    converter: DocumentConverter,
    metadata_extractor: MetadataExtractor,
):

    logger.info("Processing PDF: %s", pdf_path.name)

    markdown = load_pdf(pdf_path, converter)

    document_metadata = metadata_extractor.extract(markdown)

    chunks = chunker.chunk(
        markdown,
        pdf_path.name,
        document_metadata
    )

    logger.info("Created %d chunks for %s", len(chunks), pdf_path.name)

    return chunks


def main():

    logger.info("Scanning PDF directory: %s", DATA_DIRECTORY)

    pdf_files = sorted(
        Path(DATA_DIRECTORY).glob("*.pdf")
    )

    if not pdf_files:

        logger.warning("No PDF files found in %s", DATA_DIRECTORY)
        return

    logger.info("Found %d PDF file(s) to process", len(pdf_files))

    chunker = DocumentChunker()
    embedder = Embedder()
    metadata_extractor = MetadataExtractor()

    all_chunks = []
    converter = DocumentConverter()

    for pdf in pdf_files:

        chunks = process_pdf(
            pdf,
            chunker,
            converter,
            metadata_extractor
        )

        all_chunks.extend(chunks)

    logger.info("Total chunks generated: %d", len(all_chunks))

    embedder.create_vector_store(
        all_chunks
    )

    logger.info("Vector store created successfully")


if __name__ == "__main__":

    main()