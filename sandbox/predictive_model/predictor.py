import os
import pickle
from sandbox.predictive_model.train import train_and_save, MODEL_FILE, get_scenario_agent_count

class FailurePredictor:
    def __init__(self):
        self.model_data = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_FILE):
            # Proactively train if model file is missing
            train_and_save()
        
        with open(MODEL_FILE, "rb") as f:
            self.model_data = pickle.load(f)

    def predict_probability(self, scenario_name: str, backend: str, temperature: float, total_turns: int) -> float:
        """Predicts the likelihood of an emergent failure occurring, without running a simulation."""
        if not self.model_data:
            self._load_model()
            
        clf = self.model_data["classifier"]
        meta = self.model_data["encoding_meta"]
        
        # Build encoded feature vector
        row = []
        row.append(float(temperature))
        row.append(float(total_turns))
        row.append(float(get_scenario_agent_count(scenario_name)))
        
        # Scenario one-hot
        for s in meta["scenarios"]:
            row.append(1.0 if scenario_name == s else 0.0)
            
        # Backend one-hot
        for b in meta["backends"]:
            row.append(1.0 if backend == b else 0.0)
            
        # Predict probability of class 1 (failure)
        try:
            probabilities = clf.predict_proba([row])[0]
            # Class indices depend on trained target, but generally index 1 represents positive class
            # Let's check classes order
            if len(clf.classes_) > 1:
                class_1_idx = list(clf.classes_).index(1)
                return float(probabilities[class_1_idx])
            else:
                return float(clf.classes_[0])
        except Exception:
            # Fallback basic risk formula if prediction fails
            risk = 0.05
            if temperature > 0.9:
                risk += 0.35
            if total_turns > 12:
                risk += 0.25
            if backend in ["dummy", "ollama"]:
                risk += 0.15
            return min(risk, 0.99)
