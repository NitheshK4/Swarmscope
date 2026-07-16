import numpy as np
from typing import List, Dict, Any, Tuple
from sandbox.schemas import SimulationRun
from sandbox.analytics.sentiment_tracker import SentimentTracker

class MarkovChainAnalyzer:
    def __init__(self):
        self.states = ["Neutral", "Cooperative", "Tense", "Terminated"]
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        
        # Default logical transition matrix if training data is sparse
        # Rows: Neutral, Cooperative, Tense, Terminated
        self.transition_matrix = np.array([
            [0.50, 0.30, 0.15, 0.05], # Neutral
            [0.20, 0.65, 0.10, 0.05], # Cooperative
            [0.25, 0.05, 0.50, 0.20], # Tense
            [0.00, 0.00, 0.00, 1.00]  # Terminated (absorbing state)
        ])
        
    def _classify_message_state(self, content: str, tracker: SentimentTracker) -> str:
        score = tracker.analyze_message_sentiment(content)
        if score > 0.15:
            return "Cooperative"
        elif score < -0.15:
            return "Tense"
        else:
            return "Neutral"

    def fit_from_runs(self, runs: List[SimulationRun]) -> None:
        """Computes the transition matrix based on actual historical conversation transitions."""
        if len(runs) < 2:
            return # Keep using default transition matrix
            
        tracker = SentimentTracker()
        counts = np.zeros((4, 4))
        
        for run in runs:
            # Map messages to state sequence
            sequence = []
            for msg in run.messages:
                state = self._classify_message_state(msg.content, tracker)
                sequence.append(state)
                
            if run.metadata.status == "terminated":
                sequence.append("Terminated")
                
            # Count transitions
            for i in range(len(sequence) - 1):
                s_from = sequence[i]
                s_to = sequence[i+1]
                idx_from = self.state_to_idx[s_from]
                idx_to = self.state_to_idx[s_to]
                counts[idx_from, idx_to] += 1
                
        # Ensure Terminated is absorbing
        counts[3, 3] += 1
        
        # Normalize to get probabilities
        for i in range(4):
            row_sum = np.sum(counts[i])
            if row_sum > 0:
                self.transition_matrix[i] = counts[i] / row_sum
            else:
                # Fallback to default row if state was never visited
                pass

    def predict_absorption_risk(self, recent_messages: List[str], steps: int = 10) -> Dict[str, float]:
        """Calculates the probability of reaching the Terminated/Walkaway state in K steps."""
        tracker = SentimentTracker()
        
        # Get current state from the last message
        current_state = "Neutral"
        if recent_messages:
            current_state = self._classify_message_state(recent_messages[-1], tracker)
            
        idx = self.state_to_idx[current_state]
        
        # Start state probability vector
        v = np.zeros(4)
        v[idx] = 1.0
        
        # Power iteration: v * P^steps
        for _ in range(steps):
            v = np.dot(v, self.transition_matrix)
            
        return {
            "starting_state": current_state,
            "neutral_prob": float(v[0]),
            "cooperative_prob": float(v[1]),
            "tense_prob": float(v[2]),
            "termination_risk": float(v[3])
        }

    def get_matrix_dict(self) -> Dict[str, Dict[str, float]]:
        """Returns the transition matrix as a nested dictionary for visualization."""
        matrix_dict = {}
        for i, s_from in enumerate(self.states):
            matrix_dict[s_from] = {}
            for j, s_to in enumerate(self.states):
                matrix_dict[s_from][s_to] = float(self.transition_matrix[i, j])
        return matrix_dict
