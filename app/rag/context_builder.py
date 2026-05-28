import logging
from dataclasses import dataclass

from app.rag.models import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContextBlock:
    reference_id: int
    chunk: RetrievalResult
    text: str


@dataclass(frozen=True)
class BuiltContext:
    text: str
    blocks: list[ContextBlock]
    truncated: bool


class ContextBuilder:
    """Assemble retrieved chunks into a bounded citation-aware prompt context."""

    def __init__(self, max_context_chars: int = 12000) -> None:
        self.max_context_chars = max_context_chars

    def build(self, chunks: list[RetrievalResult]) -> BuiltContext:
        blocks: list[ContextBlock] = []
        rendered_blocks: list[str] = []
        used_chars = 0
        truncated = False

        for index, chunk in enumerate(chunks, start=1):
            rendered = self._render_chunk(index, chunk)
            next_size = used_chars + len(rendered) + (2 if rendered_blocks else 0)
            if next_size > self.max_context_chars:
                truncated = True
                break

            blocks.append(ContextBlock(reference_id=index, chunk=chunk, text=rendered))
            rendered_blocks.append(rendered)
            used_chars = next_size

        context_text = "\n\n".join(rendered_blocks)
        logger.info(
            "rag.context_built",
            extra={
                "chunk_count": len(chunks),
                "included_count": len(blocks),
                "context_chars": len(context_text),
                "truncated": truncated,
            },
        )
        return BuiltContext(text=context_text, blocks=blocks, truncated=truncated)

    @staticmethod
    def _render_chunk(reference_id: int, chunk: RetrievalResult) -> str:
        page = chunk.page_number if chunk.page_number is not None else "unknown"
        section = chunk.section_title or "Unknown section"
        return (
            f"[{reference_id}] Source: {chunk.source_filename}\n"
            f"Page: {page}\n"
            f"Section: {section}\n"
            f"Text:\n{chunk.text.strip()}"
        )
