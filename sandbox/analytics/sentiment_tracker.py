import re
from typing import List, Dict, Any
from sandbox.schemas import Message

class SentimentTracker:
    def __init__(self):
        self.cooperative_words = {
            "agree", "accept", "deal", "compromise", "pleased", "great", 
            "fair", "reasonable", "happy to", "settle", "partner", 
            "understand", "flexible", "consensus", "middle ground"
        }
        self.hostile_words = {
            "refuse", "cannot", "impossible", "walk away", "too low", "too high",
            "unacceptable", "bottleneck", "demanding", "no way", "unreasonable",
            "rigid", "stubborn", "disagree", "reject", "limits", "frustrated"
        }

    def analyze_message_sentiment(self, text: str) -> float:
        """Calculates sentiment polarity from -1.0 (tense/hostile) to +1.0 (cooperative)."""
        words = set(re.findall(r"\b\w+\b", text.lower()))
        
        coop_count = len(words.intersection(self.cooperative_words))
        hostile_count = len(words.intersection(self.hostile_words))
        
        total = coop_count + hostile_count
        if total == 0:
            return 0.0 # Neutral
            
        return (coop_count - hostile_count) / total

    def track_conversation_sentiment(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Returns a list of turn-by-turn sentiment scores for plotting."""
        trajectory = []
        for msg in messages:
            score = self.analyze_message_sentiment(msg.content)
            trajectory.append({
                "turn": msg.turn,
                "sender": msg.sender,
                "sentiment_score": score,
                "classification": "Cooperative" if score > 0.1 else "Tense/Hostile" if score < -0.1 else "Neutral"
            })
        return trajectory
