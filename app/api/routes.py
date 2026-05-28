from pathlib import Path
import time

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.schemas import (
    AnswerResponse,
    EvaluateRequest,
    EvaluateResponse,
    EvaluationMetrics,
    HealthResponse,
    HallucinationWarning,
    QuestionRequest,
    RetrievedChunk,
    SourceCitation,
    UploadResponse,
)
from app.ingestion.models import IngestionError
from app.ingestion.pipeline import IngestionPipeline
from app.rag.generator import GenerationError
from app.rag.pipeline import RagPipeline
from app.evaluation.evaluator import RagEvaluator
from app.utils.config import get_settings

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        environment=settings.environment,
    )


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF uploads are supported in Phase 1.",
        )

    settings = get_settings()
    target_path = settings.upload_dir / Path(file.filename).name
    content = await file.read()
    target_path.write_bytes(content)

    try:
        ingestion_result = IngestionPipeline(settings=settings).ingest_pdf(target_path)
    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        filename=ingestion_result.filename,
        total_pages=ingestion_result.total_pages,
        extracted_pages=ingestion_result.extracted_pages,
        total_chunks=ingestion_result.total_chunks,
        message="PDF extracted, chunked, embedded, and stored successfully.",
    )


@router.post("/qa", response_model=AnswerResponse)
def answer_question(request: QuestionRequest) -> AnswerResponse:
    settings = get_settings()
    top_k = request.top_k or settings.top_k_retrieval
    try:
        result = RagPipeline(settings=settings).answer(request.question, top_k=top_k)
    except GenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return AnswerResponse(
        question=request.question,
        top_k=top_k,
        answer=result.answer,
        citations=[
            SourceCitation(
                source_filename=citation.source_filename,
                page_number=citation.page_number,
                section_title=citation.section_title,
                chunk_id=citation.chunk_id,
            )
            for citation in result.citations
        ],
        retrieved_chunks=[
            RetrievedChunk(
                text=chunk.text,
                similarity_score=chunk.similarity_score,
                source_filename=chunk.source_filename,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
                chunk_id=chunk.chunk_id,
                metadata=chunk.metadata,
            )
            for chunk in result.retrieved_chunks
        ],
        model=result.model,
        context_truncated=result.context_truncated,
        prompt_chars=result.prompt_chars,
        response_chars=result.response_chars,
    )


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_answer(request: EvaluateRequest) -> EvaluateResponse:
    """Evaluate a RAG answer with full metrics and hallucination analysis."""
    settings = get_settings()
    top_k = request.top_k or settings.top_k_retrieval

    eval_start = time.perf_counter()

    try:
        # Run RAG pipeline
        rag_result = RagPipeline(settings=settings).answer(request.question, top_k=top_k)

        # Evaluate result
        evaluator = RagEvaluator(settings=settings)
        eval_result = evaluator.evaluate(
            rag_result=rag_result,
            expected_keywords=request.expected_keywords,
        )

    except GenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(exc)}",
        ) from exc

    response_latency_ms = round((time.perf_counter() - eval_start) * 1000, 2)

    return EvaluateResponse(
        question=request.question,
        answer=eval_result.answer,
        citations=[
            SourceCitation(
                source_filename=citation.source_filename,
                page_number=citation.page_number,
                section_title=citation.section_title,
                chunk_id=citation.chunk_id,
            )
            for citation in rag_result.citations
        ],
        metrics=EvaluationMetrics(
            faithfulness=eval_result.metrics.faithfulness,
            answer_relevancy=eval_result.metrics.answer_relevancy,
            context_precision=eval_result.metrics.context_precision,
            context_recall=eval_result.metrics.context_recall,
            retrieval_precision=eval_result.metrics.retrieval_precision,
            retrieval_ndcg=eval_result.metrics.retrieval_ndcg,
        ),
        hallucination_warnings=[
            HallucinationWarning(
                sentence=h.sentence,
                confidence=h.confidence,
                explanation=h.explanation,
            )
            for h in eval_result.hallucinations
        ],
        hallucination_rate=eval_result.hallucination_rate,
        retrieved_count=len(rag_result.retrieved_chunks),
        retrieval_stats=eval_result.retrieval_stats,
        response_latency_ms=response_latency_ms,
    )
