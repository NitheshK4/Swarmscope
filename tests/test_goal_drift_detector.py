from sandbox.schemas import Message, ScenarioConfig, AgentConfig
from sandbox.detectors.goal_drift import GoalDriftDetector

def test_goal_drift_detector_positive_case():
    detector = GoalDriftDetector()
    scenario = ScenarioConfig(
        name="car_negotiation",
        description="Negotiate vintage car",
        system_prompt="",
        agents=[
            AgentConfig(name="Alice", role="Buyer", goal="Buy vintage car cheap under limit"),
            AgentConfig(name="Bob", role="Seller", goal="Sell vintage car high above limit"),
        ],
    )
    # Start talking about car, then completely drift off to random topics like weather and cooking
    messages = [
        Message(sender="Alice", receiver="Bob", content="I want to buy the vintage car for a good price.", turn=1),
        Message(sender="Bob", receiver="Alice", content="I am selling the vintage car for $15,000.", turn=2),
        Message(sender="Alice", receiver="Bob", content="Have you seen the weather outside today? It looks sunny.", turn=3),
        Message(sender="Bob", receiver="Alice", content="Yes, I love pasta recipes for dinner on sunny days.", turn=4),
        Message(sender="Alice", receiver="Bob", content="Cooking pasta is great. What are your favorite ingredients?", turn=5),
        Message(sender="Bob", receiver="Alice", content="Garlic and olive oil make the best combination.", turn=6),
    ]
    finding = detector.analyze_finding(messages, scenario)
    assert finding.detected is True
    assert finding.score > 0.3
    assert len(finding.evidence) > 0

def test_goal_drift_detector_negative_case():
    detector = GoalDriftDetector()
    scenario = ScenarioConfig(
        name="car_negotiation",
        description="Negotiate vintage car",
        system_prompt="",
        agents=[
            AgentConfig(name="Alice", role="Buyer", goal="Buy vintage car cheap under limit"),
            AgentConfig(name="Bob", role="Seller", goal="Sell vintage car high above limit"),
        ],
    )
    messages = [
        Message(sender="Alice", receiver="Bob", content="I offer $9,000 for your vintage car.", turn=1),
        Message(sender="Bob", receiver="Alice", content="My vintage car price limit is $15,000.", turn=2),
        Message(sender="Alice", receiver="Bob", content="I can increase my bid to $10,500 for the car.", turn=3),
        Message(sender="Bob", receiver="Alice", content="My lowest price for this vintage car is $13,000.", turn=4),
    ]
    finding = detector.analyze_finding(messages, scenario)
    assert finding.detected is False
    assert finding.score < 0.3
