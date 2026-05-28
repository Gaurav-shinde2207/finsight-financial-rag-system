import json
import logging
from pathlib import Path
from typing import Any

from app.ingestion.models import DocumentChunk

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Small abstraction over a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "financial_documents",
        client: Any | None = None,
    ) -> None:
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self._client = client
        self._collection = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("chromadb is required for vector storage.") from exc

            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @property
    def collection(self) -> Any:
        if self._collection is None:
            logger.info(
                "rag.chroma_collection_init_started",
                extra={
                    "collection_name": self.collection_name,
                    "persist_dir": str(self.persist_dir),
                },
            )
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                "rag.chroma_collection_init_completed",
                extra={"collection_name": self.collection_name},
            )
        return self._collection

    def upsert_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: list[list[float]],
    ) -> int:
        """Upsert chunk texts, embeddings, and citation metadata."""

        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return 0

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [self._metadata_from_chunk(chunk) for chunk in chunks]

        logger.info(
            "rag.vector_upsert_started",
            extra={"collection_name": self.collection_name, "chunk_count": len(chunks)},
        )
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info(
            "rag.vector_upsert_completed",
            extra={"collection_name": self.collection_name, "chunk_count": len(chunks)},
        )
        return len(chunks)

    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Return raw Chroma similarity results for a query embedding."""

        logger.info(
            "rag.vector_search_started",
            extra={"collection_name": self.collection_name, "top_k": top_k},
        )
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        records = self._flatten_query_results(results)
        logger.info(
            "rag.vector_search_completed",
            extra={"collection_name": self.collection_name, "result_count": len(records)},
        )
        return records

    @staticmethod
    def _metadata_from_chunk(chunk: DocumentChunk) -> dict[str, Any]:
        metadata = {
            "source": chunk.source_filename,
            "page": chunk.page_number,
            "section": chunk.section_title or "",
            "chunk_id": chunk.chunk_id,
        }
        for key, value in chunk.metadata.items():
            if isinstance(value, str | int | float | bool) or value is None:
                metadata[key] = value
            else:
                metadata[key] = json.dumps(value, default=str)
        return metadata

    @staticmethod
    def _flatten_query_results(results: dict[str, Any]) -> list[dict[str, Any]]:
        documents = results.get("documents", [[]])[0] or []
        metadatas = results.get("metadatas", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []

        records: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) else {}
            distance = distances[index] if index < len(distances) else None
            records.append(
                {
                    "text": document,
                    "metadata": metadata or {},
                    "distance": distance,
                    "similarity_score": ChromaVectorStore._distance_to_similarity(distance),
                }
            )
        return records

    @staticmethod
    def _distance_to_similarity(distance: float | None) -> float:
        if distance is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(distance)))
