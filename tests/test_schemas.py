from sandbox.schemas import (
    AgentConfig,
    ConversationEvent,
    DetectorFinding,
    Message,
    ScenarioConfig,
    SimulationConfig,
    SimulationMetadata,
    SimulationRun,
)


def test_domain_model_instantiation():
    agent = AgentConfig(name="Alice", role="Buyer", goal="Buy cheap")
    assert agent.name == "Alice"
    assert agent.traits["assertiveness"] == 0.5

    scenario = ScenarioConfig(
        name="test_scenario",
        description="A test scenario",
        agents=[agent],
        system_prompt="Test system prompt",
        max_turns=5,
    )
    assert len(scenario.agents) == 1
    assert scenario.max_turns == 5

    config = SimulationConfig(scenario_name="test_scenario", backend="dummy", random_seed=42)
    assert config.random_seed == 42
    assert config.max_turns == 10

    event = ConversationEvent(turn=1, sender="Alice", receiver="Bob", content="Hello")
    assert event.event_type == "message"
    assert event.turn == 1

    finding = DetectorFinding(
        detector_name="loop",
        score=0.85,
        detected=True,
        severity="HIGH",
        explanation="Loop detected",
        evidence=["Hello", "Hello"],
    )
    assert finding.detected is True
    assert finding.severity == "HIGH"

    meta = SimulationMetadata(
        simulation_id="sim-123",
        scenario_name="test_scenario",
        timestamp="2026-01-01T00:00:00Z",
        total_turns=2,
        backend="dummy",
        temperature=0.7,
        status="COMPLETED",
    )
    msg = Message(sender="Alice", receiver="Bob", content="Hello", turn=1)
    run = SimulationRun(metadata=meta, messages=[msg], findings=[finding])
    assert len(run.messages) == 1
    assert len(run.findings) == 1
    assert run.findings[0].detector_name == "loop"
