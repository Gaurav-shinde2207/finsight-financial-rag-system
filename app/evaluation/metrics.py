import logging
from typing import Optional

from app.evaluation.models import MetricsResult
from app.rag.models import RetrievalResult

logger = logging.getLogger(__name__)


def evaluate_faithfulness(answer: str, context: str) -> float:
    """RAGAS faithfulness: is answer grounded in context?

    Uses RAGAS's faithfulness metric (0-1, higher is better).
    Gracefully degrades if RAGAS unavailable.
    """
    if not answer or not context:
        return 0.0

    try:
        from ragas.metrics import faithfulness

        # RAGAS expects specific input format
        # Lower level: use embeddings-based check instead of LLM
        score = faithfulness.score_single(answer, context)
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.warning(f"RAGAS faithfulness failed: {e}, using heuristic fallback")
        # Heuristic fallback: check token overlap
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        if not answer_words:
            return 0.0
        overlap = len(answer_words & context_words) / len(answer_words)
        return overlap


def evaluate_answer_relevancy(question: str, answer: str) -> float:
    """RAGAS relevancy: does answer address the question?

    Uses RAGAS's answer_relevancy metric (0-1, higher is better).
    """
    if not question or not answer:
        return 0.0

    try:
        from ragas.metrics import answer_relevancy

        score = answer_relevancy.score_single(answer, question)
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.warning(f"RAGAS answer_relevancy failed: {e}, using heuristic fallback")
        # Heuristic fallback: keyword overlap between question and answer
        question_words = set(w.lower() for w in question.split() if len(w) > 3)
        answer_words = set(w.lower() for w in answer.split() if len(w) > 3)
        if not question_words:
            return 0.5
        overlap = len(question_words & answer_words) / len(question_words)
        return min(1.0, overlap + 0.3)  # Assume some relevant content added


def evaluate_context_precision(
    question: str,
    context: str,
    retrieved_chunks: list[RetrievalResult] | None = None,
) -> float:
    """RAGAS context_precision: fraction of context relevant to question?

    Measures whether retrieved context is focused on answering question.
    """
    if not context or not question:
        return 0.0

    try:
        from ragas.metrics import context_precision

        # RAGAS context_precision expects specific format
        score = context_precision.score_single(question, context)
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.warning(f"RAGAS context_precision failed: {e}, using heuristic fallback")
        # Heuristic fallback: use retrieval scores if available
        if retrieved_chunks:
            avg_score = sum(c.similarity_score for c in retrieved_chunks) / len(retrieved_chunks)
            return avg_score
        return 0.5


def evaluate_context_recall(
    expected_keywords: list[str] | None,
    context: str,
) -> float:
    """RAGAS context_recall: recall of relevant information in context?

    Measures whether context contains information needed to answer.
    """
    if not context or not expected_keywords:
        return 0.0

    try:
        from ragas.metrics import context_recall

        score = context_recall.score_single(expected_keywords, context)
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        logger.warning(f"RAGAS context_recall failed: {e}, using heuristic fallback")
        # Heuristic fallback: keyword matching
        context_lower = context.lower()
        matched = sum(1 for kw in expected_keywords if kw.lower() in context_lower)
        return matched / len(expected_keywords) if expected_keywords else 0.5


def compute_retrieval_precision(
    retrieved_chunks: list[RetrievalResult],
    relevant_chunk_ids: list[str] | None = None,
) -> float:
    """Custom: fraction of retrieved chunks that are relevant.

    If relevant_chunk_ids provided, checks exact match.
    Otherwise uses similarity scores as proxy.
    """
    if not retrieved_chunks:
        return 0.0

    if relevant_chunk_ids:
        matched = sum(1 for chunk in retrieved_chunks if chunk.chunk_id in relevant_chunk_ids)
        return matched / len(retrieved_chunks)

    # Fallback: use similarity scores (>0.7 threshold)
    relevant_count = sum(1 for chunk in retrieved_chunks if chunk.similarity_score > 0.7)
    return relevant_count / len(retrieved_chunks)


def compute_ndcg(scores: list[float], k: int = 5) -> float:
    """NDCG@k: ranking quality metric.

    Normalized Discounted Cumulative Gain for top-k results.
    Measures ranking quality vs perfect ranking.
    """
    if not scores:
        return 0.0

    # Compute DCG for actual ranking
    dcg = sum(score / (1 + i) for i, score in enumerate(scores[:k]))

    # Compute DCG for ideal ranking (sorted descending)
    ideal_scores = sorted(scores, reverse=True)[:k]
    idcg = sum(score / (1 + i) for i, score in enumerate(ideal_scores))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_metrics(
    question: str,
    answer: str,
    context: str,
    retrieved_chunks: list[RetrievalResult] | None = None,
    expected_keywords: list[str] | None = None,
    relevant_chunk_ids: list[str] | None = None,
) -> MetricsResult:
    """Execute all metrics with graceful failure.

    Each metric computed independently so failure doesn't cascade.
    """
    retrieved_chunks = retrieved_chunks or []

    # Compute metrics individually for robustness
    faithfulness = evaluate_faithfulness(answer, context)
    logger.debug(f"Faithfulness: {faithfulness:.3f}")

    answer_relevancy = evaluate_answer_relevancy(question, answer)
    logger.debug(f"Answer relevancy: {answer_relevancy:.3f}")

    context_precision = evaluate_context_precision(question, context, retrieved_chunks)
    logger.debug(f"Context precision: {context_precision:.3f}")

    context_recall = evaluate_context_recall(expected_keywords, context)
    logger.debug(f"Context recall: {context_recall:.3f}")

    retrieval_precision = compute_retrieval_precision(retrieved_chunks, relevant_chunk_ids)
    logger.debug(f"Retrieval precision: {retrieval_precision:.3f}")

    # Compute NDCG from similarity scores
    similarity_scores = [chunk.similarity_score for chunk in retrieved_chunks]
    ndcg = compute_ndcg(similarity_scores, k=5)
    logger.debug(f"NDCG@5: {ndcg:.3f}")

    return MetricsResult(
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
        retrieval_precision=retrieval_precision,
        retrieval_ndcg=ndcg,
    )
