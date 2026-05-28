import json
import logging
from pathlib import Path

from app.evaluation.models import BenchmarkQuery

logger = logging.getLogger(__name__)


def load_benchmark_dataset(path: str) -> list[BenchmarkQuery]:
    """Load benchmark dataset from JSON file.

    Expected JSON format:
    [
      {
        "question": "What is X?",
        "expected_answer": "X is...",
        "expected_keywords": ["keyword1", "keyword2"],
        "document_source": "filename.pdf",
        "ground_truth_chunks": ["chunk_id_1", "chunk_id_2"]
      },
      ...
    ]
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Dataset must be a JSON array")

        queries = []
        for item in data:
            if not isinstance(item.get("question"), str):
                logger.warning(f"Skipping item without valid question: {item}")
                continue

            query = BenchmarkQuery(
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                expected_keywords=item.get("expected_keywords", []),
                document_source=item.get("document_source"),
                ground_truth_chunks=item.get("ground_truth_chunks", []),
            )
            queries.append(query)

        logger.info(f"Loaded {len(queries)} benchmark queries from {path}")
        return queries

    except FileNotFoundError:
        logger.error(f"Dataset file not found: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in dataset: {e}")
        raise


def validate_dataset(dataset: list[BenchmarkQuery]) -> tuple[bool, list[str]]:
    """Validate dataset format and return errors."""
    errors = []

    if not dataset:
        errors.append("Dataset is empty")
        return False, errors

    for i, query in enumerate(dataset):
        if not query.question or len(query.question) < 5:
            errors.append(f"Query {i}: question too short or missing")

        if query.expected_keywords and not isinstance(query.expected_keywords, list):
            errors.append(f"Query {i}: expected_keywords must be a list")

    return len(errors) == 0, errors


def create_empty_benchmark_dataset() -> list[BenchmarkQuery]:
    """Create template for new benchmark dataset."""
    return [
        BenchmarkQuery(
            question="Example question about a financial metric?",
            expected_answer="The example answer with relevant details.",
            expected_keywords=["keyword1", "keyword2"],
            document_source="document.pdf",
            ground_truth_chunks=["chunk_1"],
        ),
    ]


def save_benchmark_dataset(dataset: list[BenchmarkQuery], path: str) -> None:
    """Save benchmark dataset to JSON file."""
    data = [
        {
            "question": q.question,
            "expected_answer": q.expected_answer,
            "expected_keywords": q.expected_keywords,
            "document_source": q.document_source,
            "ground_truth_chunks": q.ground_truth_chunks,
        }
        for q in dataset
    ]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved {len(dataset)} benchmark queries to {path}")
