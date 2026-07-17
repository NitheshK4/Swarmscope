import pytest
from sandbox.schemas import Message
from sandbox.analytics.token_tracker import TokenTracker


def test_token_tracking():
    """Verify token usage tracking produces correct structure."""
    tracker = TokenTracker()

    messages = [
        Message(sender="Alice", receiver="Bob", content="Hello, I would like to negotiate a price for the car.", turn=1),
        Message(sender="Bob", receiver="Alice", content="Sure, my starting price is fifteen thousand dollars.", turn=2),
        Message(sender="Alice", receiver="Bob", content="That's too high.", turn=3),
        Message(sender="Bob", receiver="Alice", content="OK how about twelve thousand?", turn=4),
    ]

    result = tracker.track_token_usage(messages)

    assert result["total_tokens"] > 0
    assert result["total_messages"] == 4
    assert "Alice" in result["agent_summaries"]
    assert "Bob" in result["agent_summaries"]
    assert "gini_coefficient" in result
    assert 0.0 <= result["gini_coefficient"] <= 1.0
    assert result["fairness_label"] in ["Balanced", "Slightly Uneven", "Dominated"]


def test_token_estimate():
    """Verify token estimation heuristic."""
    tracker = TokenTracker()

    tokens = tracker.estimate_tokens("This is a simple sentence with eight words.")
    assert tokens > 0
    assert tokens >= 8  # at least word count


def test_empty_messages():
    """Verify graceful handling of empty messages."""
    tracker = TokenTracker()
    result = tracker.track_token_usage([])
    assert "error" in result


def test_gini_coefficient():
    """Verify Gini coefficient calculation."""
    tracker = TokenTracker()

    # Equal distribution
    equal_gini = tracker._compute_gini([100, 100, 100])
    assert equal_gini < 0.01

    # Highly unequal distribution
    unequal_gini = tracker._compute_gini([1, 1, 1000])
    assert unequal_gini > 0.3
