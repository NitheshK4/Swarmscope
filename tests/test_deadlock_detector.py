from sandbox.schemas import Message, ScenarioConfig, AgentConfig
from sandbox.detectors.deadlock import DeadlockDetector

def test_deadlock_detector_positive_case():
    detector = DeadlockDetector()
    scenario = ScenarioConfig(
        name="test",
        description="test",
        system_prompt="",
        agents=[AgentConfig(name="Alice", role="Buyer", goal="g"), AgentConfig(name="Bob", role="Seller", goal="g")],
    )
    messages = [
        Message(sender="Alice", receiver="Bob", content="My absolute limit is $10,000. I refuse to change.", turn=1),
        Message(sender="Bob", receiver="Alice", content="My minimum limit is $10,000. I cannot sell below $10,000.", turn=2),
        Message(sender="Alice", receiver="Bob", content="I cannot pay more than $10,000. I refuse to move.", turn=3),
        Message(sender="Bob", receiver="Alice", content="I cannot sell for anything other than $10,000. I walk away.", turn=4),
    ]
    finding = detector.analyze_finding(messages, scenario)
    assert finding.detected is True
    assert finding.score >= 0.7
    assert len(finding.evidence) > 0

def test_deadlock_detector_negative_case():
    detector = DeadlockDetector()
    scenario = ScenarioConfig(
        name="test",
        description="test",
        system_prompt="",
        agents=[AgentConfig(name="Alice", role="Buyer", goal="g"), AgentConfig(name="Bob", role="Seller", goal="g")],
    )
    messages = [
        Message(sender="Alice", receiver="Bob", content="I offer $8,000.", turn=1),
        Message(sender="Bob", receiver="Alice", content="I offer $15,000.", turn=2),
        Message(sender="Alice", receiver="Bob", content="How about $9,500?", turn=3),
        Message(sender="Bob", receiver="Alice", content="I can do $13,000.", turn=4),
    ]
    finding = detector.analyze_finding(messages, scenario)
    assert finding.detected is False
    assert finding.score < 0.5
