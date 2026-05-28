import logging
from functools import lru_cache
from typing import Protocol

logger = logging.getLogger(__name__)


class SentenceEmbeddingModel(Protocol):
    """Protocol implemented by SentenceTransformer and test doubles."""

    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ):
        """Return one embedding or a batch of embeddings."""


class EmbeddingService:
    """Reusable sentence-transformer embedding service."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        model: SentenceEmbeddingModel | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = model

    @property
    def model(self) -> SentenceEmbeddingModel:
        if self._model is None:
            logger.info(
                "rag.embedding_model_load_started",
                extra={"embedding_model": self.model_name, "device": self.device},
            )
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for embeddings.") from exc

            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(
                "rag.embedding_model_load_completed",
                extra={"embedding_model": self.model_name, "device": self.device},
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document chunks."""

        if not texts:
            return []
        logger.info(
            "rag.embed_documents_started",
            extra={"embedding_model": self.model_name, "document_count": len(texts)},
        )
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = self._to_float_matrix(embeddings)
        logger.info(
            "rag.embed_documents_completed",
            extra={"embedding_model": self.model_name, "document_count": len(result)},
        )
        return result

    def embed_query(self, query: str) -> list[float]:
        """Embed a single user query."""

        logger.info(
            "rag.embed_query_started",
            extra={"embedding_model": self.model_name, "query_length": len(query)},
        )
        embedding = self.model.encode(
            query,
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vector = self._to_float_vector(embedding)
        logger.info(
            "rag.embed_query_completed",
            extra={"embedding_model": self.model_name, "dimension": len(vector)},
        )
        return vector

    @staticmethod
    def _to_float_matrix(embeddings) -> list[list[float]]:
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [[float(value) for value in row] for row in embeddings]

    @staticmethod
    def _to_float_vector(embedding) -> list[float]:
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        return [float(value) for value in embedding]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return a shared embedding service for the process."""

    from app.utils.config import get_settings

    settings = get_settings()
    return EmbeddingService(model_name=settings.embedding_model, device="cpu")
