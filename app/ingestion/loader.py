import logging
from pathlib import Path

from app.ingestion.models import ExtractedPage, IngestionError

logger = logging.getLogger(__name__)


class PDFLoader:
    """Extract page-level text from PDFs with PyMuPDF."""

    def load(self, pdf_path: Path) -> tuple[list[ExtractedPage], int]:
        """Return non-empty extracted pages and total PDF page count."""

        if not pdf_path.exists():
            raise IngestionError(f"PDF file does not exist: {pdf_path}")

        try:
            import fitz
        except ImportError as exc:
            raise IngestionError("PyMuPDF is required for PDF extraction.") from exc

        pages: list[ExtractedPage] = []
        logger.info(
            "ingestion.pdf_load_started",
            extra={"source_filename": pdf_path.name, "path": str(pdf_path)},
        )

        try:
            with fitz.open(pdf_path) as document:
                total_pages = document.page_count
                for page_index in range(total_pages):
                    page_number = page_index + 1
                    try:
                        page = document.load_page(page_index)
                        text = self._normalize_text(page.get_text("text"))
                    except Exception as exc:
                        logger.warning(
                            "ingestion.page_extraction_failed",
                            extra={
                                "source_filename": pdf_path.name,
                                "page_number": page_number,
                                "error": str(exc),
                            },
                        )
                        continue

                    if not text:
                        continue

                    pages.append(
                        ExtractedPage(
                            source_filename=pdf_path.name,
                            page_number=page_number,
                            text=text,
                            metadata={
                                "source_path": str(pdf_path),
                                "file_type": "pdf",
                            },
                        )
                    )
        except Exception as exc:
            logger.exception(
                "ingestion.pdf_load_failed",
                extra={"source_filename": pdf_path.name, "error": str(exc)},
            )
            raise IngestionError(f"Failed to read PDF: {pdf_path.name}") from exc

        logger.info(
            "ingestion.pdf_load_completed",
            extra={
                "source_filename": pdf_path.name,
                "total_pages": total_pages,
                "extracted_pages": len(pages),
            },
        )
        return pages, total_pages

    @staticmethod
    def _normalize_text(text: str) -> str:
        normalized_lines: list[str] = []
        previous_blank = False
        for raw_line in text.replace("\x00", "").splitlines():
            line = raw_line.strip()
            if not line:
                if not previous_blank:
                    normalized_lines.append("")
                previous_blank = True
                continue
            normalized_lines.append(line)
            previous_blank = False
        return "\n".join(normalized_lines).strip()
