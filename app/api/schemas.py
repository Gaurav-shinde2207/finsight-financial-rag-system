from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    environment: str


class UploadResponse(BaseModel):
    filename: str
    total_pages: int
    extracted_pages: int
    total_chunks: int
    message: str


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    source_filename: str
    page_number: int | None = None
    section_title: str | None = None
    chunk_id: str | None = None


class RetrievedChunk(BaseModel):
    text: str
    similarity_score: float
    source_filename: str
    page_number: int | None = None
    section_title: str | None = None
    chunk_id: str
    metadata: dict


class RetrievalResponse(BaseModel):
    question: str
    top_k: int
    results: list[RetrievedChunk]


class AnswerResponse(BaseModel):
    question: str
    top_k: int
    answer: str
    citations: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunk]
    model: str
    context_truncated: bool
    prompt_chars: int
    response_chars: int


# Evaluation schemas
class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=3)
    expected_answer: str | None = Field(default=None)
    expected_keywords: list[str] | None = Field(default=None)
    top_k: int | None = Field(default=None, ge=1, le=20)


class HallucinationWarning(BaseModel):
    sentence: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str


class EvaluationMetrics(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevancy: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    retrieval_precision: float = Field(ge=0.0, le=1.0)
    retrieval_ndcg: float = Field(ge=0.0, le=1.0)


class EvaluateResponse(BaseModel):
    question: str
    answer: str
    citations: list[SourceCitation]

    # Evaluation results
    metrics: EvaluationMetrics
    hallucination_warnings: list[HallucinationWarning]
    hallucination_rate: float = Field(ge=0.0, le=1.0)

    # Retrieval stats
    retrieved_count: int
    retrieval_stats: dict
    response_latency_ms: float
