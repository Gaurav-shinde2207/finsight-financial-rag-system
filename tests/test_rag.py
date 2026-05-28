from pathlib import Path
from types import SimpleNamespace

from app.ingestion.models import DocumentChunk
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import SemanticRetriever
from app.rag.vector_store import ChromaVectorStore


class FakeSentenceModel:
    def encode(
        self,
        sentences: str | list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> list[float] | list[list[float]]:
        _ = batch_size, normalize_embeddings, show_progress_bar
        if isinstance(sentences, str):
            return [float(len(sentences)), 1.0]
        return [[float(len(sentence)), 1.0] for sentence in sentences]


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict] = {}

    def upsert(self, ids, documents, embeddings, metadatas) -> None:
        for index, chunk_id in enumerate(ids):
            self.records[chunk_id] = {
                "document": documents[index],
                "embedding": embeddings[index],
                "metadata": metadatas[index],
            }

    def query(self, query_embeddings, n_results, include):
        _ = query_embeddings, include
        rows = list(self.records.values())[:n_results]
        return {
            "documents": [[row["document"] for row in rows]],
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[0.1 for _ in rows]],
        }


class FakeChromaClient:
    def __init__(self) -> None:
        self.collection = FakeCollection()

    def get_or_create_collection(self, name: str, metadata: dict):
        _ = name, metadata
        return self.collection


def test_embedding_service_embeds_documents_and_query() -> None:
    service = EmbeddingService(
        model_name="fake-model",
        model=FakeSentenceModel(),
    )

    document_embeddings = service.embed_documents(["risk factors", "cash flow"])
    query_embedding = service.embed_query("risk")

    assert document_embeddings == [[12.0, 1.0], [9.0, 1.0]]
    assert query_embedding == [4.0, 1.0]


def test_vector_store_upserts_chunks_and_preserves_metadata() -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(persist_dir=Path("unused"), client=client)
    chunk = DocumentChunk(
        chunk_id="annual-report-p3-c0",
        source_filename="annual-report.pdf",
        page_number=3,
        section_title="Risk Factors",
        text="The company faces market and liquidity risks.",
        metadata={"source_path": "data/uploads/annual-report.pdf"},
    )

    inserted = store.upsert_chunks([chunk], [[0.1, 0.2]])
    records = store.similarity_search([0.1, 0.2], top_k=1)

    assert inserted == 1
    assert records[0]["text"] == chunk.text
    assert records[0]["metadata"]["source"] == "annual-report.pdf"
    assert records[0]["metadata"]["page"] == 3
    assert records[0]["metadata"]["section"] == "Risk Factors"
    assert records[0]["metadata"]["chunk_id"] == "annual-report-p3-c0"


def test_semantic_retriever_returns_structured_results() -> None:
    client = FakeChromaClient()
    store = ChromaVectorStore(persist_dir=Path("unused"), client=client)
    service = EmbeddingService(model_name="fake-model", model=FakeSentenceModel())
    chunk = DocumentChunk(
        chunk_id="annual-report-p5-c0",
        source_filename="annual-report.pdf",
        page_number=5,
        section_title="Risk Factors",
        text="Risk factors include credit risk and foreign exchange risk.",
    )
    store.upsert_chunks([chunk], service.embed_documents([chunk.text]))

    retriever = SemanticRetriever(
        settings=SimpleNamespace(chroma_dir=Path("unused"), top_k_retrieval=5),
        embedding_service=service,
        vector_store=store,
    )
    results = retriever.retrieve("company risk factors", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "annual-report-p5-c0"
    assert results[0].source_filename == "annual-report.pdf"
    assert results[0].page_number == 5
    assert results[0].section_title == "Risk Factors"
    assert results[0].similarity_score == 0.9
