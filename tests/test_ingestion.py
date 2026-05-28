from app.ingestion.chunker import FinancialDocumentChunker
from app.ingestion.models import ExtractedPage


def test_financial_chunker_preserves_citation_metadata_and_tables() -> None:
    page = ExtractedPage(
        source_filename="annual-report.pdf",
        page_number=7,
        text=(
            "MANAGEMENT DISCUSSION AND ANALYSIS\n"
            "Revenue increased due to higher enterprise demand. "
            "Operating margin improved year over year.\n\n"
            "Revenue        2025        2024\n"
            "Product        1200        950\n"
            "Services       800         700\n\n"
            "Liquidity remains strong with sufficient cash reserves."
        ),
        metadata={"source_path": "data/uploads/annual-report.pdf"},
    )
    chunker = FinancialDocumentChunker(chunk_size=220, chunk_overlap=60)

    chunks = chunker.chunk_pages([page])

    assert chunks
    assert all(chunk.source_filename == "annual-report.pdf" for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)
    assert chunks[0].section_title == "MANAGEMENT DISCUSSION AND ANALYSIS"
    assert any("Revenue        2025        2024" in chunk.text for chunk in chunks)
    assert all(chunk.chunk_id.startswith("annual-report-p7-c") for chunk in chunks)
    assert all("char_count" in chunk.metadata for chunk in chunks)
