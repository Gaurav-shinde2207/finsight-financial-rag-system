from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class HallucinationIssue:
    """A claim in the answer unsupported by retrieved context."""

    sentence_index: int
    sentence: str
    unsupported_tokens: list[str]
    confidence: float  # 0.0 (low) to 1.0 (high confidence issue)
    explanation: str


@dataclass(frozen=True)
class MetricsResult:
    """RAGAS and custom metrics for RAG evaluation."""

    faithfulness: float  # RAGAS: answer grounded in context [0, 1]
    answer_relevancy: float  # RAGAS: answer relevance to question [0, 1]
    context_precision: float  # RAGAS: precision of context [0, 1]
    context_recall: float  # RAGAS: recall of context [0, 1]
    retrieval_precision: float  # custom: fraction of relevant chunks [0, 1]
    retrieval_ndcg: float  # custom: NDCG@k ranking quality [0, 1]


@dataclass(frozen=True)
class EvaluationResult:
    """Complete evaluation of a RAG answer."""

    question: str
    answer: str
    metrics: MetricsResult
    hallucinations: list[HallucinationIssue] = field(default_factory=list)
    hallucination_rate: float = 0.0  # fraction of sentences with issues
    retrieval_stats: dict = field(default_factory=dict)  # timing, chunk count, etc
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class BenchmarkQuery:
    """A question for benchmark evaluation."""

    question: str
    expected_answer: str | None = None
    expected_keywords: list[str] = field(default_factory=list)
    document_source: str | None = None
    ground_truth_chunks: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregated results from benchmark evaluation."""

    dataset_name: str
    query_count: int
    timestamp: str

    # Aggregate metrics
    avg_faithfulness: float
    avg_answer_relevancy: float
    avg_context_precision: float
    avg_context_recall: float
    avg_retrieval_precision: float
    avg_retrieval_ndcg: float
    avg_response_latency_ms: float

    hallucination_rate: float  # % of answers with hallucinations
    hallucination_severity: float  # avg confidence of issues

    # Per-query details
    query_results: list[dict] = field(default_factory=list)
