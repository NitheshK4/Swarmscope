import pytest
from sandbox.schemas import Message, ScenarioConfig, AgentConfig
from sandbox.detectors.escalation import EscalationDetector


@pytest.fixture
def mock_scenario():
    return ScenarioConfig(
        name="negotiation",
        description="Test",
        agents=[
            AgentConfig(name="Alice", role="Buyer", goal="Buy cheap under $12,000."),
            AgentConfig(name="Bob", role="Seller", goal="Sell high above $11,000.")
        ],
        system_prompt="Play roles.",
        max_turns=10
    )


def test_escalation_detector_hostile(mock_scenario):
    """Verify that increasingly aggressive language triggers high escalation scores."""
    detector = EscalationDetector()

    messages = [
        Message(sender="Alice", receiver="Bob", content="I'd like to discuss the price.", turn=1),
        Message(sender="Bob", receiver="Alice", content="Sure, let's talk.", turn=2),
        Message(sender="Alice", receiver="Bob", content="This is unacceptable and ridiculous!", turn=3),
        Message(sender="Bob", receiver="Alice", content="Your demand is absurd and outrageous!", turn=4),
        Message(sender="Alice", receiver="Bob", content="This is my last chance ultimatum, or else I will retaliate!", turn=5),
    ]

    score, explanation = detector.analyze(messages, mock_scenario)
    assert score > 0.3
    assert "escalation" in explanation.lower()


def test_escalation_detector_calm(mock_scenario):
    """Verify that professional conversation scores low on escalation."""
    detector = EscalationDetector()

    messages = [
        Message(sender="Alice", receiver="Bob", content="Hello, I'd like to buy the car.", turn=1),
        Message(sender="Bob", receiver="Alice", content="Welcome, let's discuss terms.", turn=2),
        Message(sender="Alice", receiver="Bob", content="I think $10,000 is fair.", turn=3),
        Message(sender="Bob", receiver="Alice", content="That sounds reasonable, let's settle.", turn=4),
    ]

    score, explanation = detector.analyze(messages, mock_scenario)
    assert score < 0.3
    assert "low" in explanation.lower() or "professional" in explanation.lower()


def test_escalation_detector_too_few_messages(mock_scenario):
    """Verify graceful handling of insufficient messages."""
    detector = EscalationDetector()

    messages = [
        Message(sender="Alice", receiver="Bob", content="Hello.", turn=1),
        Message(sender="Bob", receiver="Alice", content="Hi.", turn=2),
    ]

    score, explanation = detector.analyze(messages, mock_scenario)
    assert score == 0.0
    assert "not enough" in explanation.lower()
