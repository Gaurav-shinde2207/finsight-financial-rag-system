import logging
import re
from typing import Callable

from app.evaluation.models import HallucinationIssue
from app.rag.models import RetrievalResult

logger = logging.getLogger(__name__)

# Common financial/document entities
FINANCIAL_TERMS = {
    "revenue", "net income", "profit", "loss", "asset", "liability",
    "equity", "cash flow", "margin", "ratio", "earnings", "dividend",
    "stock", "bond", "debt", "credit", "ebitda", "roe", "roi",
}

# Stopwords to exclude from key tokens
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "and", "or", "but", "if",
    "because", "as", "by", "for", "of", "in", "to", "from", "at", "up",
    "it", "this", "that", "which", "who", "what", "when", "where", "why",
    "how", "not", "no", "nor", "so", "than", "then", "there", "here",
}

# Financial amount pattern (e.g., $150M, 150 million, €500k)
AMOUNT_PATTERN = r'\$?\d+(?:[.,]\d{3})*(?:[.,]\d+)?(?:\s*(?:million|billion|thousand|M|B|K|k|m)?)?'

# Percentage pattern
PERCENT_PATTERN = r'\d+(?:\.\d+)?%'


def extract_named_entities(text: str) -> set[str]:
    """Extract named entities (numbers, percentages, financial terms, proper nouns)."""
    entities = set()

    # Numbers and amounts
    for match in re.finditer(AMOUNT_PATTERN, text):
        entities.add(match.group().strip())

    # Percentages
    for match in re.finditer(PERCENT_PATTERN, text):
        entities.add(match.group().strip())

    # Proper nouns (capitalized words that aren't sentence starts)
    words = text.split()
    for i, word in enumerate(words):
        # Clean punctuation
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word and clean_word[0].isupper() and len(clean_word) > 1:
            # Skip if it's start of sentence (usually position 0)
            if i > 0 or len(words) > 1:
                entities.add(clean_word.lower())

    return entities


def extract_key_tokens(text: str, exclude_stopwords: bool = True) -> set[str]:
    """Extract important tokens from text."""
    tokens = set()

    # Lowercase and split
    words = text.lower().split()

    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w]', '', word)

        if not clean_word or len(clean_word) < 2:
            continue

        # Skip stopwords if requested
        if exclude_stopwords and clean_word in STOPWORDS:
            continue

        # Include financial terms with extra weight (conceptually)
        if clean_word in FINANCIAL_TERMS or len(clean_word) > 4:
            tokens.add(clean_word)

    return tokens


def sentence_supported_by_context(
    sentence: str,
    context: str,
    retrieved_chunks: list[RetrievalResult] | None = None,
    support_threshold: float = 0.5,
) -> tuple[bool, float]:
    """Check if sentence is supported by context.

    Returns: (is_supported, confidence_score)
    - Confidence [0, 1] based on:
      - Fraction of key tokens found in context
      - Named entity matches
      - Chunk similarity scores (if available)
      - Special handling for numeric mismatches (unsupported numeric claims reduce confidence)
    """
    if not context.strip():
        return False, 0.0

    context_lower = context.lower()

    # Extract key components from sentence
    entities = extract_named_entities(sentence)
    tokens = extract_key_tokens(sentence)

    if not tokens:  # No meaningful tokens to match
        return True, 0.5

    # Check for unsupported numeric claims (high confidence issue)
    numeric_entities = {e for e in entities if re.search(r'\d', e)}
    unsupported_numeric = {e for e in numeric_entities if e.lower() not in context_lower}

    if unsupported_numeric:
        # If there's an unsupported numeric claim, confidence should be low
        return False, 0.2

    # Score entity matches
    entity_score = 0.0
    if entities:
        matching_entities = sum(1 for e in entities if e.lower() in context_lower)
        entity_score = matching_entities / len(entities)

    # Score token matches
    matching_tokens = sum(1 for t in tokens if t in context_lower)
    token_score = matching_tokens / len(tokens)

    # Average scores (weighted: entities more important for numeric claims)
    confidence = 0.6 * token_score + 0.4 * entity_score

    # Apply chunk similarity boost if available
    if retrieved_chunks:
        max_similarity = max((chunk.similarity_score for chunk in retrieved_chunks), default=0.0)
        if max_similarity > 0.8:
            confidence = min(1.0, confidence + 0.1)

    is_supported = confidence >= support_threshold

    return is_supported, confidence


def detect_hallucinations(
    answer: str,
    context: str,
    retrieved_chunks: list[RetrievalResult] | None = None,
    confidence_threshold: float = 0.7,
) -> list[HallucinationIssue]:
    """Detect unsupported claims in answer.

    Returns list of HallucinationIssue with sentence-level analysis.
    """
    hallucinations = []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

    for sent_idx, sentence in enumerate(sentences):
        if not sentence.strip():
            continue

        # Check if sentence is supported
        is_supported, confidence = sentence_supported_by_context(
            sentence,
            context,
            retrieved_chunks=retrieved_chunks,
            support_threshold=confidence_threshold,
        )

        # If not supported, add to hallucinations
        if not is_supported:
            # Find unsupported tokens
            tokens = extract_key_tokens(sentence)
            context_lower = context.lower()
            unsupported = [t for t in tokens if t not in context_lower]

            explanation = (
                f"Key terms not found in context: {', '.join(unsupported[:3])}"
                if unsupported
                else "Claim not well-supported by retrieved context"
            )

            hallucinations.append(
                HallucinationIssue(
                    sentence_index=sent_idx,
                    sentence=sentence.strip(),
                    unsupported_tokens=unsupported,
                    confidence=1.0 - confidence,  # Invert: higher = more confident it's a hallucination
                    explanation=explanation,
                )
            )

    return hallucinations


def calculate_hallucination_rate(
    answer: str,
    context: str,
    retrieved_chunks: list[RetrievalResult] | None = None,
) -> float:
    """Calculate fraction of sentences with hallucinations."""
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer.strip()) if s.strip()]

    if not sentences:
        return 0.0

    hallucinations = detect_hallucinations(answer, context, retrieved_chunks)
    return len(hallucinations) / len(sentences)
