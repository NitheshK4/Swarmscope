from typing import Dict, Any, List, Optional
from sandbox.schemas import SimulationRun
from sandbox.storage import get_store
from sandbox.analytics.sentiment_tracker import SentimentTracker
from sandbox.analytics.token_tracker import TokenTracker


class SimulationComparator:
    """Compares two simulation runs side-by-side.

    Produces a comprehensive diff of detector scores, sentiment trajectories,
    turn counts, token usage, and outcomes to identify which configuration
    performed better and why.
    """

    def __init__(self, db_path: str = "simulation_runs.duckdb"):
        self.store = get_store(db_path)
        self.sentiment_tracker = SentimentTracker()
        self.token_tracker = TokenTracker()

    def compare(self, run_id_a: str, run_id_b: str) -> Dict[str, Any]:
        """Produces a side-by-side comparison of two simulation runs.

        Args:
            run_id_a: First simulation run ID.
            run_id_b: Second simulation run ID.

        Returns:
            Dictionary with comparison results, deltas, and winner determination.
        """
        run_a = self.store.get_run(run_id_a)
        run_b = self.store.get_run(run_id_b)

        # Metadata comparison
        meta_comparison = {
            "run_a": {
                "id": run_a.metadata.simulation_id,
                "scenario": run_a.metadata.scenario_name,
                "backend": run_a.metadata.backend,
                "temperature": run_a.metadata.temperature,
                "turns": run_a.metadata.total_turns,
                "status": run_a.metadata.status
            },
            "run_b": {
                "id": run_b.metadata.simulation_id,
                "scenario": run_b.metadata.scenario_name,
                "backend": run_b.metadata.backend,
                "temperature": run_b.metadata.temperature,
                "turns": run_b.metadata.total_turns,
                "status": run_b.metadata.status
            }
        }

        # Detector score comparison
        all_detectors = set(list(run_a.detector_scores.keys()) + list(run_b.detector_scores.keys()))
        detector_comparison = {}
        for det in all_detectors:
            score_a = run_a.detector_scores.get(det, 0.0)
            score_b = run_b.detector_scores.get(det, 0.0)
            delta = score_b - score_a
            detector_comparison[det] = {
                "run_a": round(score_a, 3),
                "run_b": round(score_b, 3),
                "delta": round(delta, 3),
                "better": "A" if score_a < score_b else "B" if score_b < score_a else "Tie"
            }

        # Overall risk comparison
        max_a = max(run_a.detector_scores.values()) if run_a.detector_scores else 0.0
        max_b = max(run_b.detector_scores.values()) if run_b.detector_scores else 0.0

        # Sentiment comparison
        sentiment_a = self.sentiment_tracker.track_conversation_sentiment(run_a.messages)
        sentiment_b = self.sentiment_tracker.track_conversation_sentiment(run_b.messages)

        avg_sent_a = sum(s["sentiment_score"] for s in sentiment_a) / len(sentiment_a) if sentiment_a else 0.0
        avg_sent_b = sum(s["sentiment_score"] for s in sentiment_b) / len(sentiment_b) if sentiment_b else 0.0

        # Token usage comparison
        tokens_a = self.token_tracker.track_token_usage(run_a.messages)
        tokens_b = self.token_tracker.track_token_usage(run_b.messages)

        # Winner determination
        a_wins = sum(1 for d in detector_comparison.values() if d["better"] == "A")
        b_wins = sum(1 for d in detector_comparison.values() if d["better"] == "B")

        if a_wins > b_wins:
            overall_winner = "A"
            winner_reason = f"Run A wins {a_wins}/{len(detector_comparison)} detector comparisons with lower risk scores."
        elif b_wins > a_wins:
            overall_winner = "B"
            winner_reason = f"Run B wins {b_wins}/{len(detector_comparison)} detector comparisons with lower risk scores."
        else:
            # Tiebreaker: lower max risk
            if max_a < max_b:
                overall_winner = "A"
                winner_reason = f"Tie on detector wins. Run A has lower peak risk ({max_a:.2f} vs {max_b:.2f})."
            elif max_b < max_a:
                overall_winner = "B"
                winner_reason = f"Tie on detector wins. Run B has lower peak risk ({max_b:.2f} vs {max_a:.2f})."
            else:
                overall_winner = "Tie"
                winner_reason = "Both runs produced identical risk profiles."

        return {
            "metadata": meta_comparison,
            "detector_comparison": detector_comparison,
            "risk_summary": {
                "peak_risk_a": round(max_a, 3),
                "peak_risk_b": round(max_b, 3),
                "delta": round(max_b - max_a, 3)
            },
            "sentiment_summary": {
                "avg_sentiment_a": round(avg_sent_a, 3),
                "avg_sentiment_b": round(avg_sent_b, 3),
                "delta": round(avg_sent_b - avg_sent_a, 3)
            },
            "token_summary": {
                "total_tokens_a": tokens_a.get("total_tokens", 0),
                "total_tokens_b": tokens_b.get("total_tokens", 0),
                "gini_a": tokens_a.get("gini_coefficient", 0),
                "gini_b": tokens_b.get("gini_coefficient", 0)
            },
            "verdict": {
                "winner": overall_winner,
                "reason": winner_reason,
                "a_detector_wins": a_wins,
                "b_detector_wins": b_wins
            }
        }
