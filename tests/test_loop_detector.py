from sandbox.detectors.loop import LoopDetector
from sandbox.schemas import AgentConfig, Message, ScenarioConfig


def test_loop_detector_positive_case():
    detector = LoopDetector(threshold=0.5)
    scenario = ScenarioConfig(
        name="test",
        description="test",
        system_prompt="",
        agents=[AgentConfig(name="A", role="r", goal="g"), AgentConfig(name="B", role="r", goal="g")],
    )
    messages = [
        Message(sender="A", receiver="B", content="I offer $10,000 for the car", turn=1),
        Message(sender="B", receiver="A", content="I refuse and demand $15,000", turn=2),
        Message(sender="A", receiver="B", content="I offer $10,000 for the car", turn=3),
        Message(sender="B", receiver="A", content="I refuse and demand $15,000", turn=4),
    ]
    finding = detector.analyze_finding(messages, scenario)
    assert finding.detected is True
    assert finding.score > 0.7
    assert len(finding.evidence) > 0

def test_loop_detector_negative_case():
    detector = LoopDetector(threshold=0.5)
    scenario = ScenarioConfig(
        name="test",
        description="test",
        system_prompt="",
        agents=[AgentConfig(name="A", role="r", goal="g"), AgentConfig(name="B", role="r", goal="g")],
    )
    messages = [
        Message(sender="A", receiver="B", content="Hello, let's start negotiating.", turn=1),
        Message(sender="B", receiver="A", content="Sure, my asking price is $15,000.", turn=2),
        Message(sender="A", receiver="B", content="That is quite expensive. Can you do $12,000?", turn=3),
        Message(sender="B", receiver="A", content="I can meet you at $13,500.", turn=4),
    ]
    finding = detector.analyze_finding(messages, scenario)
    assert finding.detected is False
    assert finding.score < 0.5
