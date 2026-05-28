from dataclasses import dataclass

from app.rag.context_builder import BuiltContext
from app.rag.models import RetrievalResult


@dataclass(frozen=True)
class Citation:
    source_filename: str
    page_number: int | None = None
    section_title: str | None = None
    chunk_id: str | None = None


class CitationBuilder:
    """Build deduplicated citations from retrieved chunks."""

    def from_chunks(self, chunks: list[RetrievalResult]) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, int | None, str | None]] = set()

        for chunk in chunks:
            key = (chunk.source_filename, chunk.page_number, chunk.section_title)
            if key in seen:
                continue
            seen.add(key)
            citations.append(
                Citation(
                    source_filename=chunk.source_filename,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    chunk_id=chunk.chunk_id,
                )
            )

        return citations

    def from_context(self, context: BuiltContext) -> list[Citation]:
        return self.from_chunks([block.chunk for block in context.blocks])
