# FinSight Architecture

FinSight is organized as a modular RAG application instead of a notebook workflow.

## Layers

- `app/api`: FastAPI entrypoint, route definitions, and request/response schemas.
- `app/frontend`: Streamlit client for local MVP usage.
- `app/ingestion`: PDF loading, text extraction, chunking, and document metadata.
- `app/rag`: Embeddings, Chroma vector storage, retrieval, prompting, and answer synthesis.
- `app/evaluation`: RAG quality checks. Ragas will be added after the MVP works end to end.
- `app/observability`: Logging and tracing utilities.
- `app/utils`: Shared configuration and small helpers.

## Phase 1 Flow

1. User uploads a PDF through Streamlit or the API.
2. API stores the PDF under `data/uploads`.
3. Ingestion extracts page text with PyMuPDF.
4. Chunking creates citation-friendly text windows with page metadata.
5. Embeddings are generated with a lightweight Sentence Transformers model.
6. Chroma stores embeddings and metadata under `data/chroma`.
7. Retrieval returns top matching chunks for a question.
8. Groq generates the answer using retrieved context.
9. API returns the answer and source citations.
