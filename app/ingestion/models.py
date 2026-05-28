from pathlib import Path
from typing import Any

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from a single non-empty PDF page."""

    source_filename: str
    page_number: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DocumentChunk:
    """Citation-ready text chunk prepared for embedding and retrieval."""

    chunk_id: str
    source_filename: str
    text: str
    page_number: int
    section_title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IngestionResult:
    """Summary and chunk payload returned by the ingestion pipeline."""

    source_path: Path
    filename: str
    total_pages: int
    extracted_pages: int
    total_chunks: int
    chunks: list[DocumentChunk]


class IngestionError(Exception):
    """Raised when a document cannot be ingested into usable text chunks."""
