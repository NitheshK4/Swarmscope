import duckdb
import pickle
import os
import random
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

MODEL_FILE = os.path.join(os.path.dirname(__file__), "failure_predictor.pkl")

def get_scenario_agent_count(scenario_name: str) -> int:
    # Helper to map agent counts
    counts = {
        "negotiation": 2,
        "resource_allocation": 2,
        "debate_consensus": 2
    }
    return counts.get(scenario_name, 3)

def generate_synthetic_data(num_samples: int = 100) -> List[Dict[str, Any]]:
    """Generates synthetic runs to seed the classifier training if the DB is empty."""
    data = []
    backends = ["dummy", "ollama", "openai", "anthropic"]
    scenarios = ["negotiation", "resource_allocation", "debate_consensus"]
    
    for _ in range(num_samples):
        temp = random.uniform(0.1, 1.2)
        turns = random.randint(5, 20)
        backend = random.choice(backends)
        scenario = random.choice(scenarios)
        
        # Simple logical rules to assign synthetic labels
        # 1. Higher temp and higher turns lead to loop and goal drift failures
        # 2. Ollama / dummy backend might have higher failure rates in our simulation
        failure_prob = 0.1
        if temp > 0.9:
            failure_prob += 0.3
        if turns > 12:
            failure_prob += 0.25
        if backend in ["dummy", "ollama"]:
            failure_prob += 0.15
        if scenario == "debate_consensus" and temp < 0.3:
            # Low temperature debate might deadlock
            failure_prob += 0.2
            
        has_failure = 1 if random.random() < failure_prob else 0
        
        data.append({
            "scenario_name": scenario,
            "backend": backend,
            "temperature": temp,
            "total_turns": turns,
            "has_failure": has_failure
        })
    return data

def extract_features_and_labels(db_path: str = "simulation_runs.duckdb") -> Tuple[List[Dict[str, Any]], List[int]]:
    """Extracts features from the DuckDB store, falling back to synthetic data if empty."""
    features = []
    labels = []
    
    # Try fetching from DB first
    runs_exist = False
    if os.path.exists(db_path):
        conn = duckdb.connect(db_path)
        try:
            # Check table and count
            res = conn.execute("SELECT count(*) FROM runs").fetchone()
            if res and res[0] > 3:
                runs_exist = True
                rows = conn.execute("""
                    SELECT scenario_name, backend, temperature, total_turns,
                           loop_score, deadlock_score, collusion_score, goal_drift_score
                    FROM runs
                """).fetchall()
                
                for row in rows:
                    scenario_name, backend, temp, turns, s1, s2, s3, s4 = row
                    has_failure = 1 if max(s1, s2, s3, s4) > 0.5 else 0
                    
                    features.append({
                        "scenario_name": scenario_name,
                        "backend": backend,
                        "temperature": temp,
                        "total_turns": turns
                    })
                    labels.append(has_failure)
        except Exception:
            pass
        finally:
            conn.close()
            
    if not runs_exist:
        # Fall back to synthetic data
        synth_data = generate_synthetic_data(120)
        for row in synth_data:
            features.append({
                "scenario_name": row["scenario_name"],
                "backend": row["backend"],
                "temperature": row["temperature"],
                "total_turns": row["total_turns"]
            })
            labels.append(row["has_failure"])
            
    return features, labels

def encode_features(raw_features: List[Dict[str, Any]]) -> Tuple[List[List[float]], Dict[str, Any]]:
    """Manual one-hot encoding of categorical variables to keep model footprint small and lightweight."""
    scenarios = ["negotiation", "resource_allocation", "debate_consensus"]
    backends = ["dummy", "ollama", "openai", "anthropic"]
    
    encoded = []
    for f in raw_features:
        row = []
        # Numeric features
        row.append(float(f["temperature"]))
        row.append(float(f["total_turns"]))
        row.append(float(get_scenario_agent_count(f["scenario_name"])))
        
        # Scenario one-hot
        for s in scenarios:
            row.append(1.0 if f["scenario_name"] == s else 0.0)
            
        # Backend one-hot
        for b in backends:
            row.append(1.0 if f["backend"] == b else 0.0)
            
        encoded.append(row)
        
    meta = {
        "scenarios": scenarios,
        "backends": backends
    }
    return encoded, meta

def train_and_save(db_path: str = "simulation_runs.duckdb") -> Dict[str, Any]:
    raw_feats, labels = extract_features_and_labels(db_path)
    X, encoding_meta = encode_features(raw_feats)
    y = labels
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Save model artifacts
    model_data = {
        "classifier": clf,
        "encoding_meta": encoding_meta,
        "accuracy": accuracy,
        "evaluation_report": report
    }
    
    os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)
        
    return {
        "accuracy": accuracy,
        "model_path": MODEL_FILE,
        "report": report,
        "samples_trained": len(X)
    }
