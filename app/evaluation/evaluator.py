import logging
import time
from datetime import datetime, timezone

from app.evaluation.hallucination import detect_hallucinations, calculate_hallucination_rate
from app.evaluation.metrics import compute_metrics
from app.evaluation.models import EvaluationResult
from app.rag.pipeline import RagPipelineResult
from app.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


class RagEvaluator:
    """Orchestrate RAG answer evaluation."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def evaluate(
        self,
        rag_result: RagPipelineResult,
        expected_keywords: list[str] | None = None,
        relevant_chunk_ids: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a RAG pipeline result.

        Steps:
        1. Build context text from retrieved chunks
        2. Compute metrics (RAGAS + custom)
        3. Detect hallucinations
        4. Aggregate statistics
        5. Log structured evaluation data
        """
        eval_start = time.perf_counter()

        # Build context from retrieved chunks
        context = self._build_context(rag_result.retrieved_chunks)

        # Compute metrics
        metrics = compute_metrics(
            question=rag_result.question,
            answer=rag_result.answer,
            context=context,
            retrieved_chunks=rag_result.retrieved_chunks,
            expected_keywords=expected_keywords,
            relevant_chunk_ids=relevant_chunk_ids,
        )

        # Detect hallucinations
        hallucinations = detect_hallucinations(
            answer=rag_result.answer,
            context=context,
            retrieved_chunks=rag_result.retrieved_chunks,
            confidence_threshold=0.7,
        )

        hallucination_rate = calculate_hallucination_rate(
            answer=rag_result.answer,
            context=context,
            retrieved_chunks=rag_result.retrieved_chunks,
        )

        # Aggregate retrieval stats
        retrieval_stats = {
            "chunk_count": len(rag_result.retrieved_chunks),
            "context_chars": len(context),
            "context_truncated": rag_result.context_truncated,
            "avg_similarity_score": (
                sum(c.similarity_score for c in rag_result.retrieved_chunks)
                / len(rag_result.retrieved_chunks)
                if rag_result.retrieved_chunks
                else 0.0
            ),
        }

        eval_elapsed_ms = round((time.perf_counter() - eval_start) * 1000, 2)
        retrieval_stats["evaluation_latency_ms"] = eval_elapsed_ms

        result = EvaluationResult(
            question=rag_result.question,
            answer=rag_result.answer,
            metrics=metrics,
            hallucinations=hallucinations,
            hallucination_rate=hallucination_rate,
            retrieval_stats=retrieval_stats,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Log structured evaluation
        self._log_evaluation(result)

        return result

    def evaluate_batch(
        self,
        rag_results: list[RagPipelineResult],
        expected_keywords_list: list[list[str]] | None = None,
        relevant_chunk_ids_list: list[list[str]] | None = None,
    ) -> list[EvaluationResult]:
        """Evaluate multiple RAG results."""
        results = []

        for i, rag_result in enumerate(rag_results):
            expected_kw = (
                expected_keywords_list[i] if expected_keywords_list and i < len(expected_keywords_list) else None
            )
            relevant_ids = (
                relevant_chunk_ids_list[i] if relevant_chunk_ids_list and i < len(relevant_chunk_ids_list) else None
            )

            result = self.evaluate(
                rag_result=rag_result,
                expected_keywords=expected_kw,
                relevant_chunk_ids=relevant_ids,
            )
            results.append(result)

        return results

    @staticmethod
    def _build_context(chunks: list) -> str:
        """Build context text from retrieved chunks."""
        return "\n\n".join(
            f"[{chunk.source_filename}:{chunk.page_number}] {chunk.section_title or 'Content'}\n{chunk.text}"
            if chunk.section_title
            else f"[{chunk.source_filename}:{chunk.page_number}]\n{chunk.text}"
            for chunk in chunks
        )

    @staticmethod
    def _log_evaluation(result: EvaluationResult) -> None:
        """Log structured evaluation results."""
        logger.info(
            "evaluation.completed",
            extra={
                "question_len": len(result.question),
                "answer_len": len(result.answer),
                "faithfulness": round(result.metrics.faithfulness, 3),
                "answer_relevancy": round(result.metrics.answer_relevancy, 3),
                "context_precision": round(result.metrics.context_precision, 3),
                "context_recall": round(result.metrics.context_recall, 3),
                "retrieval_precision": round(result.metrics.retrieval_precision, 3),
                "hallucination_count": len(result.hallucinations),
                "hallucination_rate": round(result.hallucination_rate, 3),
                "chunk_count": result.retrieval_stats.get("chunk_count", 0),
                "context_chars": result.retrieval_stats.get("context_chars", 0),
            },
        )

        if result.hallucinations:
            logger.warning(
                "evaluation.hallucinations_detected",
                extra={
                    "count": len(result.hallucinations),
                    "avg_confidence": round(
                        sum(h.confidence for h in result.hallucinations) / len(result.hallucinations), 3
                    ),
                    "sentences": [h.sentence[:80] for h in result.hallucinations[:3]],
                },
            )
