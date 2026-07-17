import re
from typing import List, Dict, Any
from sandbox.schemas import Message


class TokenTracker:
    """Estimates token usage and tracks verbosity metrics per agent per turn.

    Provides turn-level and agent-level statistics including:
    - Estimated token count per message (word-count based proxy)
    - Cumulative token usage per agent
    - Verbosity trends (are messages getting longer or shorter?)
    - Gini coefficient of turn distribution (conversation fairness)
    - Agent airtime dominance metrics
    """

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimates token count from text using a word-count based heuristic.
        Roughly, 1 word ≈ 1.3 tokens for English text."""
        words = len(re.findall(r"\S+", text))
        return int(words * 1.3)

    def track_token_usage(self, messages: List[Message]) -> Dict[str, Any]:
        """Produces comprehensive token usage analytics for a conversation."""
        if not messages:
            return {"error": "No messages to analyze."}

        # Per-message tracking
        per_message = []
        agent_totals: Dict[str, int] = {}
        agent_counts: Dict[str, int] = {}
        agent_trajectories: Dict[str, List[int]] = {}

        for msg in messages:
            tokens = self.estimate_tokens(msg.content)
            per_message.append({
                "turn": msg.turn,
                "sender": msg.sender,
                "estimated_tokens": tokens,
                "word_count": len(re.findall(r"\S+", msg.content)),
                "char_count": len(msg.content)
            })

            agent_totals[msg.sender] = agent_totals.get(msg.sender, 0) + tokens
            agent_counts[msg.sender] = agent_counts.get(msg.sender, 0) + 1
            agent_trajectories.setdefault(msg.sender, []).append(tokens)

        total_tokens = sum(agent_totals.values())

        # Agent-level summaries
        agent_summaries = {}
        for agent, total in agent_totals.items():
            trajectory = agent_trajectories[agent]
            avg_tokens = total / agent_counts[agent] if agent_counts[agent] > 0 else 0
            share = total / total_tokens if total_tokens > 0 else 0

            # Verbosity trend: compare first half vs second half
            mid = len(trajectory) // 2
            if mid > 0:
                first_half = sum(trajectory[:mid]) / mid
                second_half = sum(trajectory[mid:]) / (len(trajectory) - mid)
                trend = "increasing" if second_half > first_half * 1.1 else \
                        "decreasing" if second_half < first_half * 0.9 else "stable"
            else:
                trend = "insufficient_data"

            agent_summaries[agent] = {
                "total_tokens": total,
                "message_count": agent_counts[agent],
                "avg_tokens_per_message": round(avg_tokens, 1),
                "share_of_total": round(share, 3),
                "verbosity_trend": trend,
                "min_tokens": min(trajectory),
                "max_tokens": max(trajectory)
            }

        # Gini coefficient for conversation fairness
        gini = self._compute_gini(list(agent_totals.values()))

        # Identify dominant speaker
        dominant = max(agent_totals, key=agent_totals.get) if agent_totals else None

        return {
            "total_tokens": total_tokens,
            "total_messages": len(messages),
            "avg_tokens_per_turn": round(total_tokens / len(messages), 1),
            "agent_summaries": agent_summaries,
            "gini_coefficient": round(gini, 3),
            "fairness_label": "Balanced" if gini < 0.15 else "Slightly Uneven" if gini < 0.3 else "Dominated",
            "dominant_speaker": dominant,
            "per_message": per_message
        }

    @staticmethod
    def _compute_gini(values: List[int]) -> float:
        """Computes the Gini coefficient to measure inequality in token distribution.
        0.0 = perfectly equal, 1.0 = maximally unequal."""
        if not values or sum(values) == 0:
            return 0.0

        sorted_vals = sorted(values)
        n = len(sorted_vals)
        cumulative = sum((i + 1) * v for i, v in enumerate(sorted_vals))
        total = sum(sorted_vals)

        if total == 0:
            return 0.0

        gini = (2 * cumulative) / (n * total) - (n + 1) / n
        return max(0.0, min(gini, 1.0))
