import pytest
from unittest.mock import patch, MagicMock
import json
import io
from sdk.python.swarmscope_client import SwarmScopeClient

def test_sdk_check_health():
    client = SwarmScopeClient("http://localhost:8000")
    
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps({"status": "ok"}).encode("utf-8")
    mock_resp.headers.get_content_type.return_value = "application/json"
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        assert client.check_health() is True

def test_sdk_run_simulation():
    client = SwarmScopeClient("http://localhost:8000")
    
    mock_run_data = {
        "metadata": {"simulation_id": "sdk-sim-1"},
        "messages": [],
        "detector_scores": {},
        "detector_explanations": {}
    }
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_run_data).encode("utf-8")
    mock_resp.headers.get_content_type.return_value = "application/json"
    
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp
        res = client.run_simulation("negotiation", turns=2)
        assert res["metadata"]["simulation_id"] == "sdk-sim-1"

def test_sdk_register_and_post_external_message():
    client = SwarmScopeClient("http://localhost:8000")

    # 1. Register run mock
    mock_resp_reg = MagicMock()
    mock_resp_reg.read.return_value = json.dumps({"simulation_id": "sim-ext-99"}).encode("utf-8")
    mock_resp_reg.headers.get_content_type.return_value = "application/json"

    # 2. Post message mock
    mock_resp_msg = MagicMock()
    mock_resp_msg.read.return_value = json.dumps({
        "simulation_id": "sim-ext-99",
        "detector_scores": {"collusion": 0.2},
        "status": "running"
    }).encode("utf-8")
    mock_resp_msg.headers.get_content_type.return_value = "application/json"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value.__enter__.return_value = mock_resp_reg
        sim_id = client.register_external_run("negotiation")
        assert sim_id == "sim-ext-99"

        mock_urlopen.return_value.__enter__.return_value = mock_resp_msg
        msg_res = client.post_external_message(sim_id, "AgentA", "AgentB", "Offer 10", 1)
        assert msg_res["detector_scores"]["collusion"] == 0.2
