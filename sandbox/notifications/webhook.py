import logging
from typing import Dict, Any, Optional
from sandbox.config import Config

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Configurable webhook integration for simulation risk alerts.

    When a simulation exceeds risk thresholds, fires an HTTP POST to a
    configured URL with the run summary. Supports Slack, Discord, and
    custom webhook endpoints.

    Configuration via environment variables:
        WEBHOOK_URL: Target URL for POST requests
        WEBHOOK_ENABLED: Set to 'true' to enable (default: false)
        RISK_ALERT_THRESHOLD: Minimum risk score to trigger alert (default: 0.6)
    """

    def __init__(self, url: str = None, enabled: bool = None, threshold: float = None):
        self.url = url or getattr(Config, "WEBHOOK_URL", "")
        self.enabled = enabled if enabled is not None else getattr(Config, "WEBHOOK_ENABLED", False)
        self.threshold = threshold if threshold is not None else getattr(Config, "RISK_ALERT_THRESHOLD", 0.6)

    def should_alert(self, detector_scores: Dict[str, float]) -> bool:
        """Checks if any detector score exceeds the alert threshold."""
        if not self.enabled or not self.url:
            return False
        return any(score >= self.threshold for score in detector_scores.values())

    def build_payload(self, simulation_id: str, scenario_name: str,
                      detector_scores: Dict[str, float],
                      detector_explanations: Dict[str, str],
                      backend: str, temperature: float,
                      status: str) -> Dict[str, Any]:
        """Builds the webhook payload with alert details."""
        # Find the highest risk detectors
        triggered = {k: v for k, v in detector_scores.items() if v >= self.threshold}
        max_detector = max(triggered, key=triggered.get) if triggered else "none"
        max_score = triggered.get(max_detector, 0.0)

        risk_level = "🔴 CRITICAL" if max_score > 0.8 else "🟠 HIGH" if max_score > 0.6 else "🟡 MEDIUM"

        # Slack-compatible payload (also works for Discord and custom endpoints)
        payload = {
            "text": f"{risk_level} SwarmScope Alert: Simulation `{simulation_id}` triggered risk threshold",
            "blocks": [
                {
                    "type": "header",
                    "text": "⚠️ SwarmScope Risk Alert"
                },
                {
                    "type": "section",
                    "fields": [
                        f"*Simulation ID:* `{simulation_id}`",
                        f"*Scenario:* `{scenario_name}`",
                        f"*Backend:* `{backend}`",
                        f"*Temperature:* `{temperature}`",
                        f"*Status:* `{status}`",
                        f"*Risk Level:* {risk_level}"
                    ]
                },
                {
                    "type": "section",
                    "text": "*Triggered Detectors:*"
                }
            ],
            # Raw data for custom webhook consumers
            "swarmscope": {
                "simulation_id": simulation_id,
                "scenario_name": scenario_name,
                "backend": backend,
                "temperature": temperature,
                "status": status,
                "risk_level": risk_level,
                "triggered_detectors": triggered,
                "all_scores": detector_scores,
                "explanations": {k: detector_explanations.get(k, "") for k in triggered}
            }
        }

        # Add triggered detector details
        for det, score in sorted(triggered.items(), key=lambda x: x[1], reverse=True):
            explanation = detector_explanations.get(det, "No explanation available.")
            payload["blocks"].append({
                "type": "section",
                "text": f"• *{det.upper()}*: `{score:.2f}` — {explanation}"
            })

        return payload

    def send_alert(self, simulation_id: str, scenario_name: str,
                   detector_scores: Dict[str, float],
                   detector_explanations: Dict[str, str],
                   backend: str, temperature: float,
                   status: str) -> Optional[Dict[str, Any]]:
        """Sends a webhook alert if risk threshold is exceeded.

        Returns the payload that was sent, or None if no alert was triggered.
        """
        if not self.should_alert(detector_scores):
            return None

        payload = self.build_payload(
            simulation_id=simulation_id,
            scenario_name=scenario_name,
            detector_scores=detector_scores,
            detector_explanations=detector_explanations,
            backend=backend,
            temperature=temperature,
            status=status
        )

        try:
            import requests
            response = requests.post(
                self.url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            logger.info(
                f"Webhook alert sent for simulation {simulation_id}. "
                f"Status: {response.status_code}"
            )
            return payload
        except ImportError:
            logger.warning(
                "requests library not installed. Webhook alert skipped. "
                "Install with: pip install requests"
            )
            return payload
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return payload
