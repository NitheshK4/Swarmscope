import pytest
from sandbox.schemas import Message, ScenarioConfig, AgentConfig
from sandbox.detectors.information_leakage import InformationLeakageDetector


@pytest.fixture
def negotiation_scenario():
    return ScenarioConfig(
        name="negotiation",
        description="Negotiation test",
        agents=[
            AgentConfig(name="Alice", role="Buyer", goal="Buy vintage car for cheap. Your absolute maximum limit is $12,000."),
            AgentConfig(name="Bob", role="Seller", goal="Sell vintage car high. Your absolute minimum limit is $11,000.")
        ],
        system_prompt="Play roles.",
        max_turns=10
    )


def test_leakage_detected(negotiation_scenario):
    """Verify that revealing private limits triggers the detector."""
    detector = InformationLeakageDetector()

    messages = [
        Message(sender="Alice", receiver="Bob", content="I'd like to buy the car.", turn=1),
        Message(sender="Bob", receiver="Alice", content="Make me an offer.", turn=2),
        Message(sender="Alice", receiver="Bob", content="My maximum limit is $12,000 so I can't go higher.", turn=3),
    ]

    score, explanation = detector.analyze(messages, negotiation_scenario)
    assert score > 0.3
    assert "leakage" in explanation.lower() or "disclosed" in explanation.lower() or "revealed" in explanation.lower()


def test_no_leakage(negotiation_scenario):
    """Verify clean conversation scores 0 on information leakage."""
    detector = InformationLeakageDetector()

    messages = [
        Message(sender="Alice", receiver="Bob", content="How about $9,000?", turn=1),
        Message(sender="Bob", receiver="Alice", content="That's too low, I'd like $14,000.", turn=2),
        Message(sender="Alice", receiver="Bob", content="Can we meet in the middle at $11,500?", turn=3),
    ]

    score, explanation = detector.analyze(messages, negotiation_scenario)
    assert score == 0.0
    assert "no information leakage" in explanation.lower() or "protected" in explanation.lower()


def test_no_constraints_in_goals():
    """Verify graceful handling when goals don't contain extractable constraints."""
    detector = InformationLeakageDetector()

    scenario = ScenarioConfig(
        name="debate",
        description="Debate",
        agents=[
            AgentConfig(name="Alice", role="Proponent", goal="Argue for renewable energy adoption."),
            AgentConfig(name="Bob", role="Opponent", goal="Argue for nuclear energy focus.")
        ],
        system_prompt="Debate.",
        max_turns=10
    )

    messages = [
        Message(sender="Alice", receiver="Bob", content="Solar is the future.", turn=1),
        Message(sender="Bob", receiver="Alice", content="Nuclear is more reliable.", turn=2),
    ]

    score, explanation = detector.analyze(messages, scenario)
    assert score == 0.0
    assert "no extractable" in explanation.lower()
