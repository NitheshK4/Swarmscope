import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional, List

class SwarmScopeClient:
    """Python Client SDK for interacting with the SwarmScope API server."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        payload = json.dumps(data).encode("utf-8") if data is not None else None

        req = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                resp_text = resp.read().decode("utf-8")
                if resp.headers.get_content_type() == "application/json":
                    return json.loads(resp_text)
                return resp_text
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code} Error for {method} {endpoint}: {error_body}")
        except Exception as e:
            raise RuntimeError(f"Failed to communicate with SwarmScope API at {url}: {e}")

    def check_health(self) -> bool:
        """Check if the SwarmScope API server is reachable and healthy."""
        try:
            res = self._request("GET", "/health")
            return res.get("status") == "ok"
        except Exception:
            return False

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """Retrieve all available simulation scenarios."""
        res = self._request("GET", "/scenarios")
        return res.get("scenarios", [])

    def run_simulation(self, scenario_name: str = "negotiation", backend: str = "dummy",
                       temperature: float = 0.7, turns: Optional[int] = None) -> Dict[str, Any]:
        """Trigger an execution of a multi-agent simulation run."""
        payload = {
            "scenario_name": scenario_name,
            "backend": backend,
            "temperature": temperature
        }
        if turns is not None:
            payload["turns"] = turns
        return self._request("POST", "/simulate", payload)

    def predict_risk(self, scenario_name: str, backend: str = "dummy",
                     temperature: float = 0.7, turns: int = 5) -> Dict[str, Any]:
        """Predict failure probability for a given simulation configuration."""
        payload = {
            "scenario_name": scenario_name,
            "backend": backend,
            "temperature": temperature,
            "turns": turns
        }
        return self._request("POST", "/predict", payload)

    def register_external_run(self, scenario_name: str, backend: str = "custom",
                              temperature: float = 0.7) -> str:
        """Register a new external simulation run for turn-by-turn monitoring."""
        res = self._request("POST", "/runs", {
            "scenario_name": scenario_name,
            "backend": backend,
            "temperature": temperature
        })
        return res["simulation_id"]

    def post_external_message(self, simulation_id: str, sender: str, receiver: str,
                              content: str, turn: int) -> Dict[str, Any]:
        """Post a turn message to an external run and receive live detector scores."""
        return self._request("POST", f"/runs/{simulation_id}/messages", {
            "sender": sender,
            "receiver": receiver,
            "content": content,
            "turn": turn
        })

    def get_run(self, simulation_id: str) -> Dict[str, Any]:
        """Get full details of a recorded simulation run."""
        return self._request("GET", f"/runs/{simulation_id}")

    def export_run(self, simulation_id: str, format: str = "json") -> str:
        """Export a simulation run in json, csv, jsonl, yaml, or markdown format."""
        return self._request("GET", f"/runs/{simulation_id}/export?format={format}")
