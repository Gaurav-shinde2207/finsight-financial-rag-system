import logging
from pathlib import Path

from app.ingestion.chunker import FinancialDocumentChunker
from app.ingestion.loader import PDFLoader
from app.ingestion.models import IngestionError, IngestionResult
from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.rag.vector_store import ChromaVectorStore
from app.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Orchestrate PDF loading and citation-aware chunking."""

    def __init__(
        self,
        settings: Settings | None = None,
        loader: PDFLoader | None = None,
        chunker: FinancialDocumentChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.loader = loader or PDFLoader()
        self.chunker = chunker or FinancialDocumentChunker(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
        )

    def ingest_pdf(self, pdf_path: Path) -> IngestionResult:
        logger.info(
            "ingestion.pipeline_started",
            extra={"source_filename": pdf_path.name, "path": str(pdf_path)},
        )
        pages, total_pages = self.loader.load(pdf_path)
        if not pages:
            logger.warning(
                "ingestion.no_text_extracted",
                extra={"source_filename": pdf_path.name, "total_pages": total_pages},
            )
            raise IngestionError("No extractable text found in PDF.")

        chunks = self.chunker.chunk_pages(pages)
        if not chunks:
            raise IngestionError("PDF text was extracted but no chunks were created.")

        embeddings = self.embedding_service.embed_documents([chunk.text for chunk in chunks])
        stored_chunks = self.vector_store.upsert_chunks(chunks, embeddings)

        result = IngestionResult(
            source_path=pdf_path,
            filename=pdf_path.name,
            total_pages=total_pages,
            extracted_pages=len(pages),
            total_chunks=len(chunks),
            chunks=chunks,
        )
        logger.info(
            "ingestion.pipeline_completed",
            extra={
                "source_filename": result.filename,
                "total_pages": result.total_pages,
                "extracted_pages": result.extracted_pages,
                "total_chunks": result.total_chunks,
                "stored_chunks": stored_chunks,
            },
        )
        return result
