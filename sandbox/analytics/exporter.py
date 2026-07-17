import csv
import json
import io
from typing import List, Dict, Any
from sandbox.schemas import SimulationRun, Message
from sandbox.analytics.sentiment_tracker import SentimentTracker


class ConversationExporter:
    """Exports simulation runs in multiple formats: JSON, CSV, and JSONL.

    Includes full message logs, detector scores, metadata, and optional
    sentiment analysis data for each export format.
    """

    def __init__(self):
        self.sentiment_tracker = SentimentTracker()

    def _enrich_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Adds sentiment scores to each message."""
        enriched = []
        for msg in messages:
            sentiment = self.sentiment_tracker.analyze_message_sentiment(msg.content)
            enriched.append({
                "turn": msg.turn,
                "sender": msg.sender,
                "receiver": msg.receiver,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "sentiment_score": round(sentiment, 3),
                "sentiment_label": "Cooperative" if sentiment > 0.1 else "Tense" if sentiment < -0.1 else "Neutral"
            })
        return enriched

    def to_json(self, run: SimulationRun, include_sentiment: bool = True) -> str:
        """Exports a simulation run as a formatted JSON string."""
        data = {
            "metadata": {
                "simulation_id": run.metadata.simulation_id,
                "scenario_name": run.metadata.scenario_name,
                "timestamp": run.metadata.timestamp,
                "total_turns": run.metadata.total_turns,
                "backend": run.metadata.backend,
                "temperature": run.metadata.temperature,
                "status": run.metadata.status
            },
            "detector_scores": run.detector_scores,
            "detector_explanations": run.detector_explanations,
            "messages": self._enrich_messages(run.messages) if include_sentiment else [
                {
                    "turn": m.turn,
                    "sender": m.sender,
                    "receiver": m.receiver,
                    "content": m.content,
                    "timestamp": m.timestamp
                } for m in run.messages
            ]
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def to_csv(self, run: SimulationRun) -> str:
        """Exports simulation messages as CSV with metadata columns."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header row
        writer.writerow([
            "simulation_id", "scenario_name", "backend", "temperature", "status",
            "turn", "sender", "receiver", "content", "timestamp",
            "sentiment_score", "sentiment_label"
        ])

        enriched = self._enrich_messages(run.messages)
        for msg in enriched:
            writer.writerow([
                run.metadata.simulation_id,
                run.metadata.scenario_name,
                run.metadata.backend,
                run.metadata.temperature,
                run.metadata.status,
                msg["turn"],
                msg["sender"],
                msg["receiver"],
                msg["content"],
                msg["timestamp"],
                msg["sentiment_score"],
                msg["sentiment_label"]
            ])

        return output.getvalue()

    def to_jsonl(self, run: SimulationRun) -> str:
        """Exports simulation as JSON Lines (one JSON object per line).

        First line: metadata + detector scores.
        Subsequent lines: one per message with sentiment data.
        """
        lines = []

        # Line 1: metadata
        meta_line = {
            "type": "metadata",
            "simulation_id": run.metadata.simulation_id,
            "scenario_name": run.metadata.scenario_name,
            "timestamp": run.metadata.timestamp,
            "total_turns": run.metadata.total_turns,
            "backend": run.metadata.backend,
            "temperature": run.metadata.temperature,
            "status": run.metadata.status,
            "detector_scores": run.detector_scores,
            "detector_explanations": run.detector_explanations
        }
        lines.append(json.dumps(meta_line, ensure_ascii=False))

        # Remaining lines: one per message
        enriched = self._enrich_messages(run.messages)
        for msg in enriched:
            msg_line = {"type": "message", **msg}
            lines.append(json.dumps(msg_line, ensure_ascii=False))

        return "\n".join(lines) + "\n"

    def export(self, run: SimulationRun, fmt: str = "json") -> str:
        """Exports a simulation run in the specified format.

        Args:
            run: The simulation run to export.
            fmt: Export format — 'json', 'csv', or 'jsonl'.

        Returns:
            Formatted string of the export.

        Raises:
            ValueError: If format is not supported.
        """
        fmt = fmt.lower().strip()
        if fmt == "json":
            return self.to_json(run)
        elif fmt == "csv":
            return self.to_csv(run)
        elif fmt == "jsonl":
            return self.to_jsonl(run)
        else:
            raise ValueError(f"Unsupported export format '{fmt}'. Use 'json', 'csv', or 'jsonl'.")
