import pytest
import os
from sandbox.predictive_model.train import train_and_save, MODEL_FILE
from sandbox.predictive_model.predictor import FailurePredictor

def test_predictive_model():
    # Force retrain on synthetic data
    test_db = "test_prediction.duckdb"
    if os.path.exists(test_db):
        os.remove(test_db)
        
    try:
        # Delete old model file to test recreate flow
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
            
        predictor = FailurePredictor()
        # Should automatically trigger train and save
        assert os.path.exists(MODEL_FILE)
        
        prob = predictor.predict_probability(
            scenario_name="negotiation",
            backend="dummy",
            temperature=0.8,
            total_turns=15
        )
        
        assert 0.0 <= prob <= 1.0
        
    finally:
        if os.path.exists(test_db):
            os.remove(test_db)
            if os.path.exists(test_db + ".wal"):
                try:
                    os.remove(test_db + ".wal")
                except OSError:
                    pass

def test_markov_model():
    from sandbox.predictive_model.markov_model import MarkovChainAnalyzer
    
    analyzer = MarkovChainAnalyzer()
    
    # Verify classification and absorption math
    messages = [
        "Let's settle a fair agreement.",
        "No, I refuse. That limit is unacceptable and too high."
    ]
    
    res = analyzer.predict_absorption_risk(messages, steps=3)
    assert res["starting_state"] == "Tense"
    assert 0.0 <= res["termination_risk"] <= 1.0
    
    matrix = analyzer.get_matrix_dict()
    assert "Neutral" in matrix
    assert "Terminated" in matrix["Neutral"]
    assert matrix["Terminated"]["Terminated"] == 1.0
