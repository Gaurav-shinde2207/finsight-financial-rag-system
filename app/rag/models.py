from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved chunk with citation metadata and similarity score."""

    text: str
    similarity_score: float
    source_filename: str
    page_number: int | None
    section_title: str | None
    chunk_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
