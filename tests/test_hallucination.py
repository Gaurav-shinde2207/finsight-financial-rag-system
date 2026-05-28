import pytest

from app.evaluation.hallucination import (
    extract_key_tokens,
    extract_named_entities,
    sentence_supported_by_context,
    detect_hallucinations,
    calculate_hallucination_rate,
)
from app.rag.models import RetrievalResult


def test_extract_named_entities():
    """Test named entity extraction."""
    text = "Revenue was $150 million or €200k in 2023."
    entities = extract_named_entities(text)

    assert "$150 million" in entities or "$150" in entities  # Amount variations
    assert "2023" in entities


def test_extract_named_entities_empty():
    """Test extraction from empty text."""
    entities = extract_named_entities("")
    assert len(entities) == 0


def test_extract_key_tokens():
    """Test key token extraction."""
    text = "The company reported revenue of $100 million."
    tokens = extract_key_tokens(text)

    assert "company" in tokens
    assert "reported" in tokens
    assert "revenue" in tokens
    # Stopwords should be excluded
    assert "the" not in tokens
    assert "of" not in tokens


def test_extract_key_tokens_financial():
    """Test that financial terms are captured."""
    text = "EBITDA and operating margin are key metrics."
    tokens = extract_key_tokens(text)

    assert "ebitda" in tokens or "EBITDA" in tokens.lower()


def test_sentence_supported_by_context_match():
    """Test sentence support detection with matching context."""
    sentence = "Revenue was $100 million."
    context = "The company reported revenue of $100 million in 2023."

    supported, confidence = sentence_supported_by_context(sentence, context)

    assert supported is True
    assert confidence > 0.5


def test_sentence_supported_by_context_mismatch():
    """Test sentence support detection with non-matching context."""
    sentence = "The company was founded in 1800."
    context = "Revenue was $100 million. Operating margin improved."

    supported, confidence = sentence_supported_by_context(
        sentence, context, support_threshold=0.6
    )

    assert supported is False
    assert confidence < 0.6


def test_sentence_supported_by_context_empty_context():
    """Test with empty context."""
    sentence = "Some claim."
    supported, confidence = sentence_supported_by_context(sentence, "")

    assert supported is False
    assert confidence == 0.0


def test_sentence_supported_by_context_no_tokens():
    """Test with sentence that has no meaningful tokens."""
    sentence = "The the the."
    context = "This is context."

    supported, confidence = sentence_supported_by_context(sentence, context)

    # Should handle gracefully - no meaningful tokens
    assert supported is True  # No tokens to contradict
    assert confidence == 0.5  # Default fallback


def test_detect_hallucinations_none():
    """Test detection when no hallucinations present."""
    answer = "Revenue was $100 million."
    context = "The company reported revenue of $100 million in 2023."

    hallucinations = detect_hallucinations(answer, context)

    assert len(hallucinations) == 0


def test_detect_hallucinations_single():
    """Test detection of single hallucination."""
    answer = "Revenue was $100 million. The company was founded in 1800."
    context = "The company reported revenue of $100 million."

    hallucinations = detect_hallucinations(answer, context, confidence_threshold=0.6)

    assert len(hallucinations) >= 1
    assert any("founded" in h.sentence.lower() or "1800" in h.sentence for h in hallucinations)


def test_detect_hallucinations_with_chunks():
    """Test detection using similarity scores from chunks."""
    chunk1 = RetrievalResult(
        text="Revenue was $100 million in 2023.",
        similarity_score=0.95,
        source_filename="report.pdf",
        page_number=1,
        section_title="Results",
        chunk_id="chunk_1",
    )

    answer = "Revenue was $100 million in 2023."
    context = "Revenue was $100 million in 2023."

    hallucinations = detect_hallucinations(
        answer, context, retrieved_chunks=[chunk1], confidence_threshold=0.7
    )

    assert len(hallucinations) == 0


def test_detect_hallucinations_multiple_sentences():
    """Test detection across multiple sentences."""
    answer = (
        "Revenue was $100 million. "
        "Profits increased by 50%. "
        "The CEO is a famous Nobel laureate."
    )
    context = "Revenue was $100 million. Profits increased by 50%."

    hallucinations = detect_hallucinations(answer, context, confidence_threshold=0.6)

    assert len(hallucinations) >= 1
    # At least the Nobel laureate claim should be flagged
    assert any("nobel" in h.sentence.lower() or "laureate" in h.sentence.lower()
               for h in hallucinations)


def test_calculate_hallucination_rate_zero():
    """Test hallucination rate calculation with no issues."""
    answer = "Revenue increased. Margins improved."
    context = "Revenue increased by 10%. Margins improved by 5%."

    rate = calculate_hallucination_rate(answer, context)

    # Should be low or zero
    assert rate == 0.0 or rate < 0.5


def test_calculate_hallucination_rate_high():
    """Test hallucination rate calculation with many issues."""
    answer = (
        "Revenue was $100 million. "
        "CEO won Nobel Prize. "
        "Company founded in 1776. "
        "Traded on Mars stock exchange."
    )
    context = "Revenue was $100 million."

    rate = calculate_hallucination_rate(answer, context, )

    # Should be high
    assert rate > 0.5


def test_calculate_hallucination_rate_empty_answer():
    """Test with empty answer."""
    rate = calculate_hallucination_rate("", "Some context")
    assert rate == 0.0


def test_detect_hallucinations_numeric_mismatch():
    """Test detection of numeric mismatches."""
    answer = "Revenue was $200 million."
    context = "Revenue was $100 million."

    hallucinations = detect_hallucinations(answer, context, confidence_threshold=0.6)

    # Should detect the numeric discrepancy
    assert len(hallucinations) >= 1


def test_detect_hallucinations_case_insensitive():
    """Test that detection is case insensitive."""
    answer = "REVENUE WAS $100 MILLION."
    context = "revenue was $100 million."

    hallucinations = detect_hallucinations(answer, context)

    assert len(hallucinations) == 0


def test_hallucination_confidence_scoring():
    """Test that confidence scores are properly computed."""
    answer = "Uncertain claim about company history."
    context = "The company reported results today."

    hallucinations = detect_hallucinations(answer, context, confidence_threshold=0.3)

    if hallucinations:
        for h in hallucinations:
            assert 0.0 <= h.confidence <= 1.0
            assert len(h.explanation) > 0
