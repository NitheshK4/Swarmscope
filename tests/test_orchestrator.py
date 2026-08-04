from sandbox.schemas import ScenarioConfig, AgentConfig
from sandbox.simulation import Simulation

def test_simulation_orchestrator():
    scenario = ScenarioConfig(
        name="orchestrator_test",
        description="Orchestrator test scenario",
        system_prompt="Test system prompt",
        max_turns=3,
        agents=[
            AgentConfig(name="Alice", role="Buyer", goal="Buy cheap"),
            AgentConfig(name="Bob", role="Seller", goal="Sell high"),
        ],
    )

    sim = Simulation(scenario=scenario, backend_name="dummy", random_seed=42)
    run = sim.run(max_turns=3)

    assert run.metadata.scenario_name == "orchestrator_test"
    assert run.metadata.total_turns == 3
    assert len(run.messages) == 3
    assert len(run.events) == 3
    assert len(run.findings) > 0
    assert run.metadata.duration_ms >= 0
