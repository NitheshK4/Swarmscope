from sandbox.schemas import Message
from sandbox.analytics.complexity_analyzer import ComplexityAnalyzer


def test_flesch_kincaid():
    """Verify Flesch-Kincaid grade level produces valid output."""
    analyzer = ComplexityAnalyzer()

    # Simple text should have low grade
    simple = "The cat sat on the mat. It was nice."
    fk_simple = analyzer.compute_flesch_kincaid(simple)
    assert fk_simple >= 0.0

    # Complex text should have higher grade
    complex_text = (
        "The epistemological ramifications of quantum entanglement necessitate "
        "a fundamental reconsideration of our ontological presuppositions regarding "
        "the nature of physical reality and deterministic causation."
    )
    fk_complex = analyzer.compute_flesch_kincaid(complex_text)
    assert fk_complex > fk_simple


def test_type_token_ratio():
    """Verify TTR calculation for vocabulary richness."""
    analyzer = ComplexityAnalyzer()

    # Repetitive text should have low TTR
    repetitive = "dog dog dog dog dog cat cat cat cat cat"
    ttr_rep = analyzer.compute_type_token_ratio(repetitive)

    # Diverse text should have higher TTR
    diverse = "the quick brown fox jumps over lazy sleeping red blue"
    ttr_div = analyzer.compute_type_token_ratio(diverse)

    assert ttr_rep < ttr_div
    assert 0.0 <= ttr_rep <= 1.0
    assert 0.0 <= ttr_div <= 1.0


def test_conversation_analysis():
    """Verify full conversation analysis produces expected structure."""
    analyzer = ComplexityAnalyzer()

    messages = [
        Message(sender="Alice", receiver="Bob", content="Hello, I would like to discuss the sophisticated technical proposal regarding our infrastructure.", turn=1),
        Message(sender="Bob", receiver="Alice", content="Sure, let's evaluate the comprehensive deployment strategy.", turn=2),
        Message(sender="Alice", receiver="Bob", content="OK. Deal.", turn=3),
        Message(sender="Bob", receiver="Alice", content="Yes. Done.", turn=4),
    ]

    result = analyzer.analyze_conversation(messages)

    assert "agent_summaries" in result
    assert "Alice" in result["agent_summaries"]
    assert "Bob" in result["agent_summaries"]
    assert "vocabulary_overlap" in result
    assert "has_degradation" in result


def test_empty_input():
    """Verify graceful handling of empty input."""
    analyzer = ComplexityAnalyzer()

    result = analyzer.analyze_conversation([])
    assert "error" in result
