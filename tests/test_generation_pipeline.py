from pathlib import Path
from types import SimpleNamespace

from app.rag.citation import CitationBuilder
from app.rag.context_builder import ContextBuilder
from app.rag.generator import GenerationResult
from app.rag.models import RetrievalResult
from app.rag.pipeline import RagPipeline


def make_chunk(
    chunk_id: str,
    text: str = "Revenue increased because enterprise demand improved.",
    page_number: int | None = 4,
    section_title: str | None = "Management Discussion",
) -> RetrievalResult:
    return RetrievalResult(
        text=text,
        similarity_score=0.92,
        source_filename="annual-report.pdf",
        page_number=page_number,
        section_title=section_title,
        chunk_id=chunk_id,
        metadata={"chunk_id": chunk_id},
    )


class FakeRetriever:
    def __init__(self, chunks: list[RetrievalResult]) -> None:
        self.chunks = chunks
        self.last_top_k: int | None = None

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        self.last_top_k = top_k
        return self.chunks


class FakeGenerator:
    def __init__(self) -> None:
        self.last_prompt = ""

    def generate(self, prompt: str) -> GenerationResult:
        self.last_prompt = prompt
        return GenerationResult(
            answer="Revenue increased due to improved enterprise demand. [1]",
            model="fake-model",
            prompt_chars=len(prompt),
            response_chars=55,
        )


def test_context_builder_preserves_order_and_citation_metadata() -> None:
    chunks = [
        make_chunk("chunk-1", text="First chunk text."),
        make_chunk("chunk-2", text="Second chunk text.", page_number=5, section_title="Liquidity"),
    ]

    context = ContextBuilder(max_context_chars=1000).build(chunks)

    assert not context.truncated
    assert "[1] Source: annual-report.pdf" in context.text
    assert "Page: 4" in context.text
    assert "Section: Management Discussion" in context.text
    assert context.text.index("First chunk text.") < context.text.index("Second chunk text.")


def test_context_builder_truncates_oversized_context() -> None:
    chunks = [
        make_chunk("chunk-1", text="A" * 200),
        make_chunk("chunk-2", text="B" * 200),
    ]

    context = ContextBuilder(max_context_chars=300).build(chunks)

    assert context.truncated
    assert len(context.blocks) == 1
    assert "A" * 20 in context.text


def test_citation_builder_deduplicates_by_source_page_and_section() -> None:
    chunks = [
        make_chunk("chunk-1"),
        make_chunk("chunk-2"),
        make_chunk("chunk-3", page_number=7, section_title="Risk Factors"),
    ]

    citations = CitationBuilder().from_chunks(chunks)

    assert len(citations) == 2
    assert citations[0].source_filename == "annual-report.pdf"
    assert citations[0].page_number == 4
    assert citations[0].section_title == "Management Discussion"
    assert citations[1].page_number == 7


def test_rag_pipeline_orchestrates_retrieval_context_generation_and_citations() -> None:
    chunks = [make_chunk("chunk-1")]
    retriever = FakeRetriever(chunks)
    generator = FakeGenerator()
    settings = SimpleNamespace(
        chroma_dir=Path("unused"),
        top_k_retrieval=5,
        max_context_chars=1000,
        generation_model="fake-model",
    )

    result = RagPipeline(
        settings=settings,
        retriever=retriever,
        generator=generator,
    ).answer("Why did revenue increase?", top_k=1)

    assert retriever.last_top_k == 1
    assert result.answer == "Revenue increased due to improved enterprise demand. [1]"
    assert result.citations[0].source_filename == "annual-report.pdf"
    assert result.retrieved_chunks == chunks
    assert "Why did revenue increase?" in generator.last_prompt
    assert "Retrieved context:" in generator.last_prompt
