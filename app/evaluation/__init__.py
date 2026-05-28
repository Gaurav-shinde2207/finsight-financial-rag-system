"""Evaluation, hallucination detection, and RAG observability module."""

from app.evaluation.benchmark import BenchmarkRunner
from app.evaluation.datasets import load_benchmark_dataset, save_benchmark_dataset, validate_dataset
from app.evaluation.evaluator import RagEvaluator
from app.evaluation.hallucination import detect_hallucinations, calculate_hallucination_rate
from app.evaluation.metrics import compute_metrics
from app.evaluation.models import (
    BenchmarkQuery,
    BenchmarkReport,
    EvaluationResult,
    HallucinationIssue,
    MetricsResult,
)

__all__ = [
    # Models
    "EvaluationResult",
    "HallucinationIssue",
    "MetricsResult",
    "BenchmarkQuery",
    "BenchmarkReport",
    # Core evaluation
    "RagEvaluator",
    # Hallucination detection
    "detect_hallucinations",
    "calculate_hallucination_rate",
    # Metrics
    "compute_metrics",
    # Benchmark
    "BenchmarkRunner",
    # Datasets
    "load_benchmark_dataset",
    "save_benchmark_dataset",
    "validate_dataset",
]
