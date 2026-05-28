import json
import logging
import time
from datetime import datetime, timezone

from app.evaluation.evaluator import RagEvaluator
from app.evaluation.models import BenchmarkQuery, BenchmarkReport, EvaluationResult
from app.rag.pipeline import RagPipeline
from app.utils.config import Settings, get_settings

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Execute RAG benchmarks."""

    def __init__(
        self,
        rag_pipeline: RagPipeline,
        evaluator: RagEvaluator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.rag_pipeline = rag_pipeline
        self.evaluator = evaluator or RagEvaluator()
        self.settings = settings or get_settings()

    def run_benchmark(
        self,
        dataset: list[BenchmarkQuery],
        max_queries: int | None = None,
        dataset_name: str = "benchmark",
    ) -> BenchmarkReport:
        """Run benchmark on dataset.

        Returns BenchmarkReport with aggregated metrics.
        """
        queries_to_run = dataset if max_queries is None else dataset[:max_queries]

        logger.info(f"Starting benchmark: {dataset_name} with {len(queries_to_run)} queries")

        query_results = []
        start_time = time.perf_counter()

        for i, query in enumerate(queries_to_run):
            try:
                logger.debug(f"Running query {i+1}/{len(queries_to_run)}: {query.question[:60]}...")

                query_start = time.perf_counter()

                # Run RAG pipeline
                rag_result = self.rag_pipeline.answer(query.question)

                # Evaluate result
                eval_result = self.evaluator.evaluate(
                    rag_result=rag_result,
                    expected_keywords=query.expected_keywords,
                    relevant_chunk_ids=query.ground_truth_chunks,
                )

                query_elapsed_ms = round((time.perf_counter() - query_start) * 1000, 2)

                # Store detailed result
                query_results.append(
                    {
                        "question": query.question,
                        "answer": eval_result.answer,
                        "expected_answer": query.expected_answer,
                        "metrics": {
                            "faithfulness": eval_result.metrics.faithfulness,
                            "answer_relevancy": eval_result.metrics.answer_relevancy,
                            "context_precision": eval_result.metrics.context_precision,
                            "context_recall": eval_result.metrics.context_recall,
                            "retrieval_precision": eval_result.metrics.retrieval_precision,
                            "retrieval_ndcg": eval_result.metrics.retrieval_ndcg,
                        },
                        "hallucination_rate": eval_result.hallucination_rate,
                        "hallucination_count": len(eval_result.hallucinations),
                        "response_latency_ms": query_elapsed_ms,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to evaluate query {i+1}: {e}", exc_info=True)
                query_results.append(
                    {
                        "question": query.question,
                        "error": str(e),
                    }
                )

        total_elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Aggregate metrics
        successful_results = [r for r in query_results if "error" not in r]

        if not successful_results:
            logger.error("Benchmark: no successful queries")
            return self._create_empty_report(dataset_name, len(queries_to_run), query_results)

        # Compute averages
        avg_faithfulness = sum(r["metrics"]["faithfulness"] for r in successful_results) / len(successful_results)
        avg_answer_relevancy = (
            sum(r["metrics"]["answer_relevancy"] for r in successful_results) / len(successful_results)
        )
        avg_context_precision = (
            sum(r["metrics"]["context_precision"] for r in successful_results) / len(successful_results)
        )
        avg_context_recall = (
            sum(r["metrics"]["context_recall"] for r in successful_results) / len(successful_results)
        )
        avg_retrieval_precision = (
            sum(r["metrics"]["retrieval_precision"] for r in successful_results) / len(successful_results)
        )
        avg_retrieval_ndcg = sum(r["metrics"]["retrieval_ndcg"] for r in successful_results) / len(successful_results)
        avg_response_latency = sum(r["response_latency_ms"] for r in successful_results) / len(successful_results)

        hallucination_rate = sum(r["hallucination_rate"] for r in successful_results) / len(successful_results)
        total_hallucinations = sum(r["hallucination_count"] for r in successful_results)
        hallucination_severity = (
            total_hallucinations / len(successful_results) if successful_results else 0.0
        )

        report = BenchmarkReport(
            dataset_name=dataset_name,
            query_count=len(queries_to_run),
            timestamp=datetime.now(timezone.utc).isoformat(),
            avg_faithfulness=avg_faithfulness,
            avg_answer_relevancy=avg_answer_relevancy,
            avg_context_precision=avg_context_precision,
            avg_context_recall=avg_context_recall,
            avg_retrieval_precision=avg_retrieval_precision,
            avg_retrieval_ndcg=avg_retrieval_ndcg,
            avg_response_latency_ms=avg_response_latency,
            hallucination_rate=hallucination_rate,
            hallucination_severity=hallucination_severity,
            query_results=query_results,
        )

        logger.info(
            f"Benchmark completed: {dataset_name}",
            extra={
                "query_count": len(queries_to_run),
                "successful": len(successful_results),
                "total_time_ms": total_elapsed_ms,
                "avg_faithfulness": round(avg_faithfulness, 3),
                "avg_answer_relevancy": round(avg_answer_relevancy, 3),
                "hallucination_rate": round(hallucination_rate, 3),
            },
        )

        return report

    def save_results(self, report: BenchmarkReport, path: str) -> None:
        """Save benchmark results as JSON."""
        with open(path, "w") as f:
            json.dump(
                {
                    "dataset_name": report.dataset_name,
                    "query_count": report.query_count,
                    "timestamp": report.timestamp,
                    "avg_faithfulness": report.avg_faithfulness,
                    "avg_answer_relevancy": report.avg_answer_relevancy,
                    "avg_context_precision": report.avg_context_precision,
                    "avg_context_recall": report.avg_context_recall,
                    "avg_retrieval_precision": report.avg_retrieval_precision,
                    "avg_retrieval_ndcg": report.avg_retrieval_ndcg,
                    "avg_response_latency_ms": report.avg_response_latency_ms,
                    "hallucination_rate": report.hallucination_rate,
                    "hallucination_severity": report.hallucination_severity,
                    "query_results": report.query_results,
                },
                f,
                indent=2,
            )

        logger.info(f"Benchmark results saved to {path}")

    @staticmethod
    def _create_empty_report(
        dataset_name: str,
        query_count: int,
        query_results: list[dict],
    ) -> BenchmarkReport:
        """Create empty report for failed benchmark."""
        return BenchmarkReport(
            dataset_name=dataset_name,
            query_count=query_count,
            timestamp=datetime.now(timezone.utc).isoformat(),
            avg_faithfulness=0.0,
            avg_answer_relevancy=0.0,
            avg_context_precision=0.0,
            avg_context_recall=0.0,
            avg_retrieval_precision=0.0,
            avg_retrieval_ndcg=0.0,
            avg_response_latency_ms=0.0,
            hallucination_rate=0.0,
            hallucination_severity=0.0,
            query_results=query_results,
        )

    @staticmethod
    def compare_reports(report1: BenchmarkReport, report2: BenchmarkReport) -> dict:
        """Compare two benchmark runs for regression detection."""
        return {
            "dataset": (report1.dataset_name, report2.dataset_name),
            "timestamps": (report1.timestamp, report2.timestamp),
            "faithfulness_delta": report2.avg_faithfulness - report1.avg_faithfulness,
            "answer_relevancy_delta": report2.avg_answer_relevancy - report1.avg_answer_relevancy,
            "context_precision_delta": report2.avg_context_precision - report1.avg_context_precision,
            "context_recall_delta": report2.avg_context_recall - report1.avg_context_recall,
            "retrieval_precision_delta": report2.avg_retrieval_precision - report1.avg_retrieval_precision,
            "hallucination_rate_delta": report2.hallucination_rate - report1.hallucination_rate,
            "response_latency_delta_ms": report2.avg_response_latency_ms - report1.avg_response_latency_ms,
            "regression_detected": (
                report2.avg_faithfulness < report1.avg_faithfulness * 0.95
                or report2.hallucination_rate > report1.hallucination_rate * 1.1
            ),
        }
