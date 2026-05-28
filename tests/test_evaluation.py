import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from app.evaluation.evaluator import RagEvaluator
from app.evaluation.models import MetricsResult
from app.rag.models import RetrievalResult
from app.rag.pipeline import RagPipelineResult


@dataclass
class FakeSettings:
    app_name = "FinSight"
    generation_model = "groq/mixtral"


def make_chunk(text: str, score: float = 0.9, filename: str = "test.pdf") -> RetrievalResult:
    """Factory for test chunks."""
    return RetrievalResult(
        text=text,
        similarity_score=score,
        source_filename=filename,
        page_number=1,
        section_title="Test Section",
        chunk_id="chunk_1",
    )


def make_rag_result(
    question: str = "What is revenue?",
    answer: str = "The revenue is $100 million.",
    chunks: list[RetrievalResult] | None = None,
) -> RagPipelineResult:
    """Factory for test RAG results."""
    if chunks is None:
        chunks = [make_chunk("Revenue is $100 million in 2023.")]

    return RagPipelineResult(
        question=question,
        answer=answer,
        citations=[],
        retrieved_chunks=chunks,
        model="test-model",
        context_truncated=False,
        prompt_chars=100,
        response_chars=50,
    )


def test_evaluator_init():
    """Test RagEvaluator initialization."""
    evaluator = RagEvaluator(settings=FakeSettings())
    assert evaluator.settings is not None


def test_evaluator_evaluate_basic():
    """Test basic evaluation flow."""
    evaluator = RagEvaluator(settings=FakeSettings())
    rag_result = make_rag_result()

    with patch("app.evaluation.evaluator.compute_metrics") as mock_metrics, \
         patch("app.evaluation.evaluator.detect_hallucinations") as mock_halluc:

        mock_metrics.return_value = MetricsResult(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            retrieval_precision=0.9,
            retrieval_ndcg=0.88,
        )
        mock_halluc.return_value = []

        result = evaluator.evaluate(rag_result)

        assert result.question == "What is revenue?"
        assert result.answer == "The revenue is $100 million."
        assert result.metrics.faithfulness == 0.9
        assert result.hallucination_rate == 0.0
        assert len(result.hallucinations) == 0


def test_evaluator_with_hallucinations():
    """Test evaluation with detected hallucinations."""
    from app.evaluation.models import HallucinationIssue

    evaluator = RagEvaluator(settings=FakeSettings())
    rag_result = make_rag_result(
        answer="Revenue is $100 million. Also, the company was founded in 1900."
    )

    with patch("app.evaluation.evaluator.compute_metrics") as mock_metrics, \
         patch("app.evaluation.evaluator.detect_hallucinations") as mock_halluc:

        mock_metrics.return_value = MetricsResult(
            faithfulness=0.7,
            answer_relevancy=0.8,
            context_precision=0.7,
            context_recall=0.7,
            retrieval_precision=0.8,
            retrieval_ndcg=0.85,
        )

        halluc_issue = HallucinationIssue(
            sentence_index=1,
            sentence="Also, the company was founded in 1900.",
            unsupported_tokens=["founded", "1900"],
            confidence=0.85,
            explanation="Founded year not mentioned in context",
        )
        mock_halluc.return_value = [halluc_issue]

        result = evaluator.evaluate(rag_result)

        assert len(result.hallucinations) == 1
        assert result.hallucinations[0].confidence == 0.85
        assert result.hallucination_rate > 0.0


def test_evaluator_batch():
    """Test batch evaluation."""
    evaluator = RagEvaluator(settings=FakeSettings())
    rag_results = [
        make_rag_result(question="Q1", answer="A1"),
        make_rag_result(question="Q2", answer="A2"),
    ]

    with patch("app.evaluation.metrics.compute_metrics") as mock_metrics, \
         patch("app.evaluation.hallucination.detect_hallucinations") as mock_halluc:

        mock_metrics.return_value = MetricsResult(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            retrieval_precision=0.9,
            retrieval_ndcg=0.88,
        )
        mock_halluc.return_value = []

        results = evaluator.evaluate_batch(rag_results)

        assert len(results) == 2
        assert results[0].question == "Q1"
        assert results[1].question == "Q2"


def test_evaluator_with_expected_keywords():
    """Test evaluation with expected keywords."""
    evaluator = RagEvaluator(settings=FakeSettings())
    rag_result = make_rag_result()

    with patch("app.evaluation.evaluator.compute_metrics") as mock_metrics, \
         patch("app.evaluation.evaluator.detect_hallucinations") as mock_halluc:

        mock_metrics.return_value = MetricsResult(
            faithfulness=0.9,
            answer_relevancy=0.85,
            context_precision=0.8,
            context_recall=0.75,
            retrieval_precision=0.9,
            retrieval_ndcg=0.88,
        )
        mock_halluc.return_value = []

        result = evaluator.evaluate(
            rag_result,
            expected_keywords=["revenue", "million"],
        )

        assert result is not None
        mock_metrics.assert_called_once()
        call_kwargs = mock_metrics.call_args[1]
        assert call_kwargs["expected_keywords"] == ["revenue", "million"]


def test_evaluator_build_context():
    """Test context building from chunks."""
    chunks = [
        make_chunk("Revenue data", filename="financial.pdf"),
        make_chunk("Expense data", filename="financial.pdf"),
    ]

    context = RagEvaluator._build_context(chunks)

    assert "Revenue data" in context
    assert "Expense data" in context
    assert "financial.pdf" in context
    assert "[financial.pdf:1]" in context


def test_evaluator_retrieval_stats():
    """Test retrieval stats computation."""
    chunks = [
        make_chunk("Text 1", score=0.95),
        make_chunk("Text 2", score=0.85),
        make_chunk("Text 3", score=0.75),
    ]
    rag_result = make_rag_result(chunks=chunks)

    evaluator = RagEvaluator(settings=FakeSettings())

    with patch("app.evaluation.metrics.compute_metrics") as mock_metrics, \
         patch("app.evaluation.hallucination.detect_hallucinations") as mock_halluc:

        mock_metrics.return_value = MetricsResult(
            faithfulness=0.9, answer_relevancy=0.85, context_precision=0.8,
            context_recall=0.75, retrieval_precision=0.9, retrieval_ndcg=0.88,
        )
        mock_halluc.return_value = []

        result = evaluator.evaluate(rag_result)

        assert result.retrieval_stats["chunk_count"] == 3
        assert result.retrieval_stats["context_truncated"] is False
        assert "avg_similarity_score" in result.retrieval_stats
        assert abs(result.retrieval_stats["avg_similarity_score"] - 0.85) < 0.01


def test_evaluator_empty_chunks():
    """Test evaluation with no retrieved chunks."""
    rag_result = make_rag_result(chunks=[])

    evaluator = RagEvaluator(settings=FakeSettings())

    with patch("app.evaluation.metrics.compute_metrics") as mock_metrics, \
         patch("app.evaluation.hallucination.detect_hallucinations") as mock_halluc:

        mock_metrics.return_value = MetricsResult(
            faithfulness=0.5, answer_relevancy=0.6, context_precision=0.0,
            context_recall=0.0, retrieval_precision=0.0, retrieval_ndcg=0.0,
        )
        mock_halluc.return_value = []

        result = evaluator.evaluate(rag_result)

        assert result.retrieval_stats["chunk_count"] == 0
        assert result.retrieval_stats["avg_similarity_score"] == 0.0
