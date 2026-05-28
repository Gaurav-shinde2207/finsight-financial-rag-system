import hashlib
import logging
import re
from dataclasses import dataclass

from app.ingestion.models import DocumentChunk, ExtractedPage

logger = logging.getLogger(__name__)

SECTION_PATTERNS = (
    "management discussion",
    "financial statements",
    "balance sheet",
    "income statement",
    "cash flow",
    "risk factors",
    "notes to",
    "results of operations",
    "liquidity",
)


@dataclass(frozen=True)
class TextBlock:
    text: str
    section_title: str | None
    block_type: str


class FinancialDocumentChunker:
    """Create citation-friendly chunks while preserving headings and tables."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for page in pages:
            page_chunks = self._chunk_page(page)
            chunks.extend(page_chunks)

        logger.info(
            "ingestion.chunks_created",
            extra={
                "source_filename": pages[0].source_filename if pages else None,
                "total_chunks": len(chunks),
            },
        )
        return chunks

    def _chunk_page(self, page: ExtractedPage) -> list[DocumentChunk]:
        blocks = self._build_blocks(page.text)
        chunks: list[DocumentChunk] = []
        buffer: list[TextBlock] = []
        buffer_length = 0
        chunk_index = 0

        for block in blocks:
            block_length = len(block.text)
            should_flush = buffer and buffer_length + block_length + 2 > self.chunk_size
            if should_flush:
                chunks.append(self._create_chunk(page, buffer, chunk_index))
                chunk_index += 1
                buffer = self._overlap_blocks(buffer)
                buffer_length = sum(len(item.text) + 2 for item in buffer)

            if block_length > self.chunk_size and block.block_type != "table":
                if buffer:
                    chunks.append(self._create_chunk(page, buffer, chunk_index))
                    chunk_index += 1
                    buffer = []
                    buffer_length = 0
                for split_text in self._split_long_text(block.text):
                    split_block = TextBlock(
                        text=split_text,
                        section_title=block.section_title,
                        block_type=block.block_type,
                    )
                    chunks.append(self._create_chunk(page, [split_block], chunk_index))
                    chunk_index += 1
                continue

            buffer.append(block)
            buffer_length += block_length + 2

        if buffer:
            chunks.append(self._create_chunk(page, buffer, chunk_index))

        return chunks

    def _build_blocks(self, text: str) -> list[TextBlock]:
        blocks: list[TextBlock] = []
        current_section: str | None = None
        paragraph: list[str] = []
        table: list[str] = []

        def flush_paragraph() -> None:
            if paragraph:
                blocks.append(
                    TextBlock(
                        text=" ".join(paragraph).strip(),
                        section_title=current_section,
                        block_type="paragraph",
                    )
                )
                paragraph.clear()

        def flush_table() -> None:
            if table:
                blocks.append(
                    TextBlock(
                        text="\n".join(table).strip(),
                        section_title=current_section,
                        block_type="table",
                    )
                )
                table.clear()

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush_paragraph()
                flush_table()
                continue

            if self._is_heading(line):
                flush_paragraph()
                flush_table()
                current_section = line
                blocks.append(TextBlock(line, current_section, "heading"))
                continue

            if self._looks_like_table_row(line):
                flush_paragraph()
                table.append(line)
                continue

            flush_table()
            paragraph.append(line)

        flush_paragraph()
        flush_table()
        return blocks

    def _create_chunk(
        self,
        page: ExtractedPage,
        blocks: list[TextBlock],
        chunk_index: int,
    ) -> DocumentChunk:
        text = "\n\n".join(block.text for block in blocks).strip()
        section_title = self._dominant_section(blocks)
        chunk_id = self._chunk_id(page.source_filename, page.page_number, chunk_index, text)
        return DocumentChunk(
            chunk_id=chunk_id,
            source_filename=page.source_filename,
            page_number=page.page_number,
            section_title=section_title,
            text=text,
            metadata={
                **page.metadata,
                "chunk_index": chunk_index,
                "char_count": len(text),
                "block_types": sorted({block.block_type for block in blocks}),
            },
        )

    def _overlap_blocks(self, blocks: list[TextBlock]) -> list[TextBlock]:
        if self.chunk_overlap <= 0:
            return []

        overlap: list[TextBlock] = []
        total = 0
        for block in reversed(blocks):
            next_total = total + len(block.text)
            if next_total > self.chunk_overlap and overlap:
                break
            overlap.insert(0, block)
            total = next_total
            if total >= self.chunk_overlap:
                break
        return overlap

    def _split_long_text(self, text: str) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.chunk_size:
                current = f"{current} {sentence}".strip()
                continue
            if current:
                parts.append(current)
            current = sentence
        if current:
            parts.append(current)
        return parts

    @staticmethod
    def _is_heading(line: str) -> bool:
        normalized = line.lower()
        if any(pattern in normalized for pattern in SECTION_PATTERNS):
            return True
        if re.match(r"^(item\s+\d+[a-z]?\.?|part\s+[ivx]+\.?)\s+", normalized):
            return True
        if len(line) <= 90 and not line.endswith((".", ",", ";")):
            alpha_chars = [char for char in line if char.isalpha()]
            if alpha_chars and sum(char.isupper() for char in alpha_chars) / len(alpha_chars) > 0.75:
                return True
        return False

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        has_numeric_columns = len(re.findall(r"[-$()]?\d[\d,]*(?:\.\d+)?%?", line)) >= 2
        has_wide_spacing = bool(re.search(r"\S\s{2,}\S", line))
        return "|" in line or (has_numeric_columns and has_wide_spacing)

    @staticmethod
    def _dominant_section(blocks: list[TextBlock]) -> str | None:
        for block in reversed(blocks):
            if block.section_title:
                return block.section_title
        return None

    @staticmethod
    def _chunk_id(filename: str, page_number: int, chunk_index: int, text: str) -> str:
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        safe_name = re.sub(r"[^a-zA-Z0-9]+", "-", filename.rsplit(".", 1)[0]).strip("-")
        return f"{safe_name}-p{page_number}-c{chunk_index}-{digest}"
