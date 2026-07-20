import pytest
import os
from sandbox.storage.duckdb_store import DuckDBStore
from sandbox.schemas import SimulationRun, SimulationMetadata, Message

@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_runs.duckdb")
    store = DuckDBStore(db_path=db_file)
    yield store

def test_duckdb_save_and_get_run(temp_db):
    metadata = SimulationMetadata(
        simulation_id="sim-test-1",
        scenario_name="negotiation",
        timestamp="2026-07-20T12:00:00Z",
        total_turns=2,
        backend="dummy",
        temperature=0.7,
        status="completed"
    )
    messages = [
        Message(sender="A", receiver="B", content="Hello", turn=1, timestamp="12:00"),
        Message(sender="B", receiver="A", content="World", turn=2, timestamp="12:01")
    ]
    scores = {"loop": 0.1, "collusion": 0.85, "jailbreak": 0.0}
    explanations = {"collusion": "High collusion detected"}

    run = SimulationRun(
        metadata=metadata,
        messages=messages,
        detector_scores=scores,
        detector_explanations=explanations
    )

    temp_db.save_run(run)

    retrieved = temp_db.get_run("sim-test-1")
    assert retrieved.metadata.simulation_id == "sim-test-1"
    assert retrieved.metadata.scenario_name == "negotiation"
    assert len(retrieved.messages) == 2
    assert retrieved.messages[0].content == "Hello"
    assert retrieved.detector_scores["collusion"] == 0.85

def test_duckdb_get_all_runs(temp_db):
    m1 = SimulationMetadata(
        simulation_id="sim-1", scenario_name="test1", timestamp="10:00",
        total_turns=1, backend="dummy", temperature=0.7, status="completed"
    )
    m2 = SimulationMetadata(
        simulation_id="sim-2", scenario_name="test2", timestamp="11:00",
        total_turns=2, backend="dummy", temperature=0.5, status="completed"
    )
    run1 = SimulationRun(metadata=m1, messages=[], detector_scores={}, detector_explanations={})
    run2 = SimulationRun(metadata=m2, messages=[], detector_scores={}, detector_explanations={})

    temp_db.save_run(run1)
    temp_db.save_run(run2)

    all_runs = temp_db.get_all_runs()
    assert len(all_runs) == 2
    ids = [r.simulation_id for r in all_runs]
    assert "sim-1" in ids
    assert "sim-2" in ids

def test_duckdb_delete_run(temp_db):
    m = SimulationMetadata(
        simulation_id="sim-del", scenario_name="test", timestamp="10:00",
        total_turns=1, backend="dummy", temperature=0.7, status="completed"
    )
    run = SimulationRun(metadata=m, messages=[], detector_scores={}, detector_explanations={})
    temp_db.save_run(run)

    temp_db.delete_run("sim-del")

    with pytest.raises(ValueError):
        temp_db.get_run("sim-del")

def test_duckdb_get_nonexistent_run(temp_db):
    with pytest.raises(ValueError):
        temp_db.get_run("sim-nonexistent")
