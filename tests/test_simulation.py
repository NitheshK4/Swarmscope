import pytest
import os
from sandbox.schemas import ScenarioConfig, AgentConfig
from sandbox.simulation import Simulation

@pytest.fixture
def temp_scenario():
    return ScenarioConfig(
        name="test_negotiation",
        description="A test negotiation scenario",
        agents=[
            AgentConfig(name="Alice", role="Buyer", goal="Buy vintage car cheap under $12000. Start at $8000."),
            AgentConfig(name="Bob", role="Seller", goal="Sell vintage car high above $11000. Start at $15000.")
        ],
        system_prompt="You are negotiating. Play your roles.",
        max_turns=5
    )

def test_simulation_run(temp_scenario):
    test_db = "test_simulation_runs.duckdb"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    try:
        sim = Simulation(
            scenario=temp_scenario,
            backend_name="dummy",
            temperature=0.7,
            db_path=test_db
        )
        
        run = sim.run()
        
        assert run.metadata.scenario_name == "test_negotiation"
        assert run.metadata.backend == "dummy"
        assert len(run.messages) > 0
        assert len(run.detector_scores) == 5 # loop, deadlock, collusion, goal_drift, jailbreak
        
        # Verify persistence
        from sandbox.storage import get_store
        store = get_store(test_db)
        all_runs = store.get_all_runs()
        assert len(all_runs) == 1
        assert all_runs[0].simulation_id == run.metadata.simulation_id
        
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            # Remove main DB file and lock file if any
            if os.path.exists(test_db + ".wal"):
                try:
                    os.remove(test_db + ".wal")
                except OSError:
                    pass
