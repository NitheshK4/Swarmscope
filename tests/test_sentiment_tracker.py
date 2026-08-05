from sandbox.analytics.sentiment_tracker import SentimentTracker
from sandbox.schemas import Message


def test_analyze_message_sentiment_cooperative():
    tracker = SentimentTracker()
    text = "I agree to this deal and accept your reasonable compromise."
    score = tracker.analyze_message_sentiment(text)
    assert score > 0.0

def test_analyze_message_sentiment_hostile():
    tracker = SentimentTracker()
    text = "This is unacceptable! I refuse and will walk away from this negotiation."
    score = tracker.analyze_message_sentiment(text)
    assert score < 0.0

def test_analyze_message_sentiment_neutral():
    tracker = SentimentTracker()
    text = "The quick brown fox jumps over the lazy dog."
    score = tracker.analyze_message_sentiment(text)
    assert score == 0.0

def test_track_conversation_sentiment():
    tracker = SentimentTracker()
    messages = [
        Message(sender="AgentA", receiver="AgentB", content="I agree to compromise.", turn=1, timestamp="12:00"),
        Message(sender="AgentB", receiver="AgentA", content="This is unacceptable and rigid.", turn=2, timestamp="12:01"),
        Message(sender="AgentA", receiver="AgentB", content="The proposal is standard.", turn=3, timestamp="12:02")
    ]

    trajectory = tracker.track_conversation_sentiment(messages)
    assert len(trajectory) == 3

    assert trajectory[0]["sender"] == "AgentA"
    assert trajectory[0]["classification"] == "Cooperative"
    assert trajectory[0]["sentiment_score"] > 0

    assert trajectory[1]["sender"] == "AgentB"
    assert trajectory[1]["classification"] == "Tense/Hostile"
    assert trajectory[1]["sentiment_score"] < 0

    assert trajectory[2]["classification"] == "Neutral"
    assert trajectory[2]["sentiment_score"] == 0.0


def test_sentiment_multi_word_phrases_and_caching():
    tracker = SentimentTracker()
    text = "We can find a middle ground and work together."
    score1 = tracker.analyze_message_sentiment(text)
    assert score1 > 0.0

    # Test caching hit
    score2 = tracker.analyze_message_sentiment(text)
    assert score1 == score2

    text_hostile = "I will walk away because this offer is too low and no way acceptable."
    score_h = tracker.analyze_message_sentiment(text_hostile)
    assert score_h < 0.0

