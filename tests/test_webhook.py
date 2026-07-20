import pytest
from unittest.mock import patch, MagicMock
from sandbox.notifications.webhook import WebhookNotifier

def test_webhook_disabled_by_default():
    notifier = WebhookNotifier(url="", enabled=False)
    assert notifier.should_alert({"collusion": 0.9}) is False

def test_should_alert_threshold():
    notifier = WebhookNotifier(url="https://hooks.slack.com/services/test", enabled=True, threshold=0.6)
    assert notifier.should_alert({"collusion": 0.5, "escalation": 0.3}) is False
    assert notifier.should_alert({"collusion": 0.7, "escalation": 0.3}) is True

def test_build_payload_structure():
    notifier = WebhookNotifier(url="https://hooks.slack.com/services/test", enabled=True, threshold=0.6)
    scores = {"collusion": 0.85, "escalation": 0.4}
    explanations = {"collusion": "High collusion detected between agents."}
    
    payload = notifier.build_payload(
        simulation_id="sim-12345",
        scenario_name="negotiation",
        detector_scores=scores,
        detector_explanations=explanations,
        backend="dummy",
        temperature=0.7,
        status="completed"
    )

    assert "CRITICAL" in payload["text"]
    assert payload["swarmscope"]["simulation_id"] == "sim-12345"
    assert payload["swarmscope"]["triggered_detectors"]["collusion"] == 0.85
    assert len(payload["blocks"]) >= 4

def test_send_alert_success():
    notifier = WebhookNotifier(url="https://example.com/webhook", enabled=True, threshold=0.5)
    scores = {"jailbreak": 0.7}
    explanations = {"jailbreak": "Jailbreak risk detected"}

    with patch("requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        res = notifier.send_alert(
            simulation_id="sim-999",
            scenario_name="crisis_response",
            detector_scores=scores,
            detector_explanations=explanations,
            backend="dummy",
            temperature=0.8,
            status="completed"
        )

        assert res is not None
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert args[0] == "https://example.com/webhook"
        assert kwargs["json"]["swarmscope"]["simulation_id"] == "sim-999"

def test_send_alert_no_alert_when_low_risk():
    notifier = WebhookNotifier(url="https://example.com/webhook", enabled=True, threshold=0.8)
    scores = {"jailbreak": 0.3}

    with patch("requests.post") as mock_post:
        res = notifier.send_alert(
            simulation_id="sim-000",
            scenario_name="negotiation",
            detector_scores=scores,
            detector_explanations={},
            backend="dummy",
            temperature=0.7,
            status="completed"
        )
        assert res is None
        mock_post.assert_not_called()
