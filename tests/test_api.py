import pytest
from fastapi.testclient import TestClient
from sandbox.api import app
from sandbox.schemas import SimulationRun, SimulationMetadata, Message

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data

def test_list_scenarios_endpoint():
    response = client.get("/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert "scenarios" in data
    assert isinstance(data["scenarios"], list)
    assert data["total"] >= 1

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_runs" in data

def test_predict_risk_endpoint():
    response = client.post("/predict", json={
        "scenario_name": "negotiation",
        "backend": "dummy",
        "temperature": 0.7,
        "turns": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "failure_probability" in data
    assert "risk_level" in data

def test_simulate_endpoint():
    response = client.post("/simulate", json={
        "scenario_name": "negotiation",
        "backend": "dummy",
        "temperature": 0.5,
        "turns": 2
    })
    assert response.status_code == 200
    data = response.json()
    assert "metadata" in data
    assert "messages" in data
    assert len(data["messages"]) > 0

def test_runs_lifecycle_endpoints():
    # 1. Register run
    create_res = client.post("/runs", json={
        "scenario_name": "negotiation",
        "backend": "custom",
        "temperature": 0.7
    })
    assert create_res.status_code == 200
    sim_id = create_res.json()["simulation_id"]
    assert sim_id

    # 2. Add message to run
    msg_res = client.post(f"/runs/{sim_id}/messages", json={
        "sender": "AgentA",
        "receiver": "AgentB",
        "content": "I offer 10 units",
        "turn": 1
    })
    assert msg_res.status_code == 200
    assert "detector_scores" in msg_res.json()

    # 3. Get run details
    get_res = client.get(f"/runs/{sim_id}")
    assert get_res.status_code == 200
    assert get_res.json()["metadata"]["simulation_id"] == sim_id

    # 4. Export run
    export_res = client.get(f"/runs/{sim_id}/export?format=json")
    assert export_res.status_code == 200

    # 5. Get markdown safety report
    report_res = client.get(f"/runs/{sim_id}/report")
    assert report_res.status_code == 200

    # 6. Delete run
    del_res = client.delete(f"/runs/{sim_id}")
    assert del_res.status_code == 200

    # Verify deleted
    get_after = client.get(f"/runs/{sim_id}")
    assert get_after.status_code == 404
