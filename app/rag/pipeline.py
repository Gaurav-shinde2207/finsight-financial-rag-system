import logging
import time
from dataclasses import dataclass

from app.rag.citation import Citation, CitationBuilder
from app.rag.context_builder import ContextBuilder
from app.rag.generator import GenerationError, GenerationResult, GroqGenerationService
from app.rag.models import RetrievalResult
from app.rag.prompts import build_answer_prompt
from app.rag.retriever import SemanticRetriever
from app.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RagPipelineResult:
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]
    model: str
    context_truncated: bool
    prompt_chars: int
    response_chars: int


class RagPipeline:
    """Coordinate retrieval, context assembly, generation, and citations."""

    def __init__(
        self,
        settings: Settings | None = None,
        retriever: SemanticRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        generator: GroqGenerationService | None = None,
        citation_builder: CitationBuilder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or SemanticRetriever(settings=self.settings)
        self.context_builder = context_builder or ContextBuilder(
            max_context_chars=self.settings.max_context_chars,
        )
        self.generator = generator or GroqGenerationService(settings=self.settings)
        self.citation_builder = citation_builder or CitationBuilder()

    def answer(self, question: str, top_k: int | None = None) -> RagPipelineResult:
        retrieval_start = time.perf_counter()
        retrieval_k = top_k or self.settings.top_k_retrieval
        chunks = self.retriever.retrieve(question, top_k=retrieval_k)
        retrieval_ms = round((time.perf_counter() - retrieval_start) * 1000, 2)
        logger.info(
            "rag.pipeline_retrieval_completed",
            extra={"elapsed_ms": retrieval_ms, "result_count": len(chunks), "top_k": retrieval_k},
        )

        context = self.context_builder.build(chunks)
        prompt = build_answer_prompt(question=question, context=context.text)
        logger.info(
            "rag.prompt_prepared",
            extra={"prompt_chars": len(prompt), "context_chars": len(context.text)},
        )

        if not context.blocks:
            fallback_answer = (
                "The available document context is insufficient to answer this question. "
                "Upload or index relevant financial documents and try again."
            )
            generation = GenerationResult(
                answer=fallback_answer,
                model=self.settings.generation_model,
                prompt_chars=len(prompt),
                response_chars=len(fallback_answer),
            )
        else:
            try:
                generation = self.generator.generate(prompt)
            except GenerationError:
                raise

        citations = self.citation_builder.from_context(context)
        logger.info(
            "rag.pipeline_completed",
            extra={
                "citation_count": len(citations),
                "retrieved_count": len(chunks),
                "response_chars": generation.response_chars,
            },
        )
        return RagPipelineResult(
            question=question,
            answer=generation.answer,
            citations=citations,
            retrieved_chunks=chunks,
            model=generation.model,
            context_truncated=context.truncated,
            prompt_chars=generation.prompt_chars,
            response_chars=generation.response_chars,
        )
