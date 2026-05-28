import pytest
import json
import tempfile
from unittest.mock import MagicMock, patch
from pathlib import Path

from app.evaluation.benchmark import BenchmarkRunner
from app.evaluation.models import BenchmarkQuery, BenchmarkReport, MetricsResult, EvaluationResult
from app.rag.pipeline import RagPipelineResult
from app.rag.models import RetrievalResult


def make_chunk(text: str = "Revenue data") -> RetrievalResult:
    """Factory for test chunks."""
    return RetrievalResult(
        text=text,
        similarity_score=0.85,
        source_filename="test.pdf",
        page_number=1,
        section_title="Results",
        chunk_id="chunk_1",
    )


def make_rag_result(answer: str = "Answer text") -> RagPipelineResult:
    """Factory for test RAG results."""
    return RagPipelineResult(
        question="Test question?",
        answer=answer,
        citations=[],
        retrieved_chunks=[make_chunk()],
        model="test-model",
        context_truncated=False,
        prompt_chars=100,
        response_chars=50,
    )


def make_eval_result() -> EvaluationResult:
    """Factory for test evaluation results."""
    return EvaluationResult(
        question="Test question?",
        answer="Answer text",
        metrics=MetricsResult(
            faithfulness=0.85,
            answer_relevancy=0.80,
            context_precision=0.75,
            context_recall=0.70,
            retrieval_precision=0.90,
            retrieval_ndcg=0.88,
        ),
        hallucinations=[],
        hallucination_rate=0.0,
        retrieval_stats={"chunk_count": 1},
    )


def test_benchmark_runner_init():
    """Test BenchmarkRunner initialization."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    assert runner.rag_pipeline == mock_pipeline
    assert runner.evaluator == mock_evaluator


def test_benchmark_runner_empty_dataset():
    """Test benchmark with empty dataset."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    report = runner.run_benchmark(dataset=[], dataset_name="test")

    assert report.query_count == 0
    assert report.avg_faithfulness == 0.0


def test_benchmark_runner_single_query():
    """Test benchmark with single query."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    # Setup mocks
    mock_pipeline.answer.return_value = make_rag_result()
    mock_evaluator.evaluate.return_value = make_eval_result()

    dataset = [
        BenchmarkQuery(
            question="What is revenue?",
            expected_keywords=["revenue"],
        ),
    ]

    report = runner.run_benchmark(dataset=dataset, dataset_name="test")

    assert report.query_count == 1
    assert report.avg_faithfulness == 0.85
    assert len(report.query_results) == 1


def test_benchmark_runner_multiple_queries():
    """Test benchmark with multiple queries."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    # Setup mocks
    mock_pipeline.answer.side_effect = [
        make_rag_result(answer="Answer 1"),
        make_rag_result(answer="Answer 2"),
        make_rag_result(answer="Answer 3"),
    ]

    eval_results = [make_eval_result() for _ in range(3)]
    mock_evaluator.evaluate.side_effect = eval_results

    dataset = [
        BenchmarkQuery(question="Q1", expected_keywords=["key1"]),
        BenchmarkQuery(question="Q2", expected_keywords=["key2"]),
        BenchmarkQuery(question="Q3", expected_keywords=["key3"]),
    ]

    report = runner.run_benchmark(dataset=dataset, dataset_name="test")

    assert report.query_count == 3
    assert len(report.query_results) == 3
    assert mock_pipeline.answer.call_count == 3
    assert mock_evaluator.evaluate.call_count == 3


def test_benchmark_runner_max_queries():
    """Test benchmark with max_queries limit."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    mock_pipeline.answer.return_value = make_rag_result()
    mock_evaluator.evaluate.return_value = make_eval_result()

    dataset = [
        BenchmarkQuery(question=f"Q{i}") for i in range(10)
    ]

    report = runner.run_benchmark(dataset=dataset, max_queries=3, dataset_name="test")

    assert report.query_count == 3
    assert mock_pipeline.answer.call_count == 3


def test_benchmark_runner_error_handling():
    """Test benchmark error handling."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    # First query succeeds, second fails
    mock_pipeline.answer.side_effect = [
        make_rag_result(),
        Exception("Pipeline error"),
    ]
    mock_evaluator.evaluate.return_value = make_eval_result()

    dataset = [
        BenchmarkQuery(question="Q1"),
        BenchmarkQuery(question="Q2"),
    ]

    report = runner.run_benchmark(dataset=dataset, dataset_name="test")

    # Should complete without crashing
    assert len(report.query_results) == 2
    assert "error" in report.query_results[1]


def test_benchmark_runner_aggregation():
    """Test metric aggregation."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    mock_pipeline.answer.side_effect = [
        make_rag_result(),
        make_rag_result(),
    ]

    # Different metrics for each query
    eval1 = EvaluationResult(
        question="Q1", answer="A1",
        metrics=MetricsResult(
            faithfulness=0.90, answer_relevancy=0.85,
            context_precision=0.80, context_recall=0.75,
            retrieval_precision=0.95, retrieval_ndcg=0.92,
        ),
        hallucinations=[], hallucination_rate=0.0,
        retrieval_stats={},
    )

    eval2 = EvaluationResult(
        question="Q2", answer="A2",
        metrics=MetricsResult(
            faithfulness=0.80, answer_relevancy=0.75,
            context_precision=0.70, context_recall=0.65,
            retrieval_precision=0.85, retrieval_ndcg=0.83,
        ),
        hallucinations=[], hallucination_rate=0.0,
        retrieval_stats={},
    )

    mock_evaluator.evaluate.side_effect = [eval1, eval2]

    dataset = [
        BenchmarkQuery(question="Q1"),
        BenchmarkQuery(question="Q2"),
    ]

    report = runner.run_benchmark(dataset=dataset, dataset_name="test")

    assert abs(report.avg_faithfulness - 0.85) < 0.01  # (0.90 + 0.80) / 2
    assert abs(report.avg_answer_relevancy - 0.80) < 0.01  # (0.85 + 0.75) / 2
    assert abs(report.avg_retrieval_precision - 0.90) < 0.01


def test_benchmark_runner_save_results():
    """Test saving benchmark results."""
    mock_pipeline = MagicMock()
    mock_evaluator = MagicMock()

    runner = BenchmarkRunner(
        rag_pipeline=mock_pipeline,
        evaluator=mock_evaluator,
    )

    mock_pipeline.answer.return_value = make_rag_result()
    mock_evaluator.evaluate.return_value = make_eval_result()

    dataset = [BenchmarkQuery(question="Q1")]
    report = runner.run_benchmark(dataset=dataset, dataset_name="test")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "benchmark_results.json"
        runner.save_results(report, str(output_path))

        assert output_path.exists()

        # Verify content
        with open(output_path) as f:
            data = json.load(f)

        assert data["dataset_name"] == "test"
        assert data["query_count"] == 1
        assert "avg_faithfulness" in data


def test_benchmark_runner_compare_reports():
    """Test report comparison."""
    report1 = BenchmarkReport(
        dataset_name="test",
        query_count=10,
        timestamp="2024-01-01T00:00:00",
        avg_faithfulness=0.85,
        avg_answer_relevancy=0.80,
        avg_context_precision=0.75,
        avg_context_recall=0.70,
        avg_retrieval_precision=0.90,
        avg_retrieval_ndcg=0.88,
        avg_response_latency_ms=250,
        hallucination_rate=0.1,
        hallucination_severity=0.5,
        query_results=[],
    )

    report2 = BenchmarkReport(
        dataset_name="test",
        query_count=10,
        timestamp="2024-01-02T00:00:00",
        avg_faithfulness=0.80,  # Slight decrease
        avg_answer_relevancy=0.80,
        avg_context_precision=0.75,
        avg_context_recall=0.70,
        avg_retrieval_precision=0.90,
        avg_retrieval_ndcg=0.88,
        avg_response_latency_ms=260,
        hallucination_rate=0.12,  # Increase
        hallucination_severity=0.55,
        query_results=[],
    )

    comparison = BenchmarkRunner.compare_reports(report1, report2)

    assert comparison["faithfulness_delta"] < 0.0  # Decreased
    assert comparison["hallucination_rate_delta"] > 0.0  # Increased
    assert comparison["response_latency_delta_ms"] > 0.0  # Increased latency


def test_benchmark_runner_regression_detection():
    """Test regression detection in comparisons."""
    report1 = BenchmarkReport(
        dataset_name="test", query_count=10, timestamp="2024-01-01T00:00:00",
        avg_faithfulness=0.90, avg_answer_relevancy=0.85,
        avg_context_precision=0.80, avg_context_recall=0.75,
        avg_retrieval_precision=0.90, avg_retrieval_ndcg=0.88,
        avg_response_latency_ms=250, hallucination_rate=0.05,
        hallucination_severity=0.3, query_results=[],
    )

    # Significant regression
    report2 = BenchmarkReport(
        dataset_name="test", query_count=10, timestamp="2024-01-02T00:00:00",
        avg_faithfulness=0.80,  # >5% drop
        avg_answer_relevancy=0.85,
        avg_context_precision=0.80, avg_context_recall=0.75,
        avg_retrieval_precision=0.90, avg_retrieval_ndcg=0.88,
        avg_response_latency_ms=250, hallucination_rate=0.06,
        hallucination_severity=0.35, query_results=[],
    )

    comparison = BenchmarkRunner.compare_reports(report1, report2)

    assert comparison["regression_detected"] is True
