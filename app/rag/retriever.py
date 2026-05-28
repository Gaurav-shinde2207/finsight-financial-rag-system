import logging
from pathlib import Path
from typing import Protocol

from app.rag.embeddings import EmbeddingService, get_embedding_service
from app.rag.models import RetrievalResult
from app.rag.vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RetrieverSettings(Protocol):
    chroma_dir: Path
    top_k_retrieval: int


class SemanticRetriever:
    """Semantic retrieval layer independent from answer generation."""

    def __init__(
        self,
        settings: RetrieverSettings | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
    ) -> None:
        if settings is None:
            from app.utils.config import get_settings

            settings = get_settings()
        self.settings = settings
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or ChromaVectorStore(
            persist_dir=self.settings.chroma_dir,
        )

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Embed a question and return the most relevant stored chunks."""

        retrieval_k = top_k or self.settings.top_k_retrieval
        logger.info(
            "rag.retrieval_started",
            extra={"top_k": retrieval_k, "query_length": len(query)},
        )
        query_embedding = self.embedding_service.embed_query(query)
        records = self.vector_store.similarity_search(query_embedding, retrieval_k)
        results = [self._to_retrieval_result(record) for record in records]
        logger.info(
            "rag.retrieval_completed",
            extra={"top_k": retrieval_k, "result_count": len(results)},
        )
        return results

    @staticmethod
    def _to_retrieval_result(record: dict) -> RetrievalResult:
        metadata = record.get("metadata") or {}
        page = metadata.get("page")
        return RetrievalResult(
            text=record.get("text", ""),
            similarity_score=float(record.get("similarity_score", 0.0)),
            source_filename=str(metadata.get("source", "")),
            page_number=int(page) if page not in (None, "") else None,
            section_title=str(metadata.get("section") or "") or None,
            chunk_id=str(metadata.get("chunk_id", "")),
            metadata=metadata,
        )
