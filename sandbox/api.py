import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from sandbox.utils import load_scenario, generate_id, get_current_timestamp
from sandbox.simulation import Simulation
from sandbox.analytics.batch_runner import BatchRunner
from sandbox.analytics.counterfactual_replay import CounterfactualReplayEngine
from sandbox.predictive_model import FailurePredictor
from sandbox.storage import get_store
from sandbox.schemas import SimulationRun, SimulationMetadata, Message, ScenarioConfig
from sandbox.detectors import get_all_detectors

app = FastAPI(title="Emergent Behavior Sandbox API", version="1.0.0")

class SimulationRequest(BaseModel):
    scenario_name: str = Field("negotiation", description="Name of scenario (negotiation, resource_allocation, debate_consensus)")
    backend: str = Field("dummy", description="LLM backend (dummy, ollama, openai, anthropic)")
    temperature: float = Field(0.7, description="Temperature (0.0 to 1.2)")
    turns: Optional[int] = Field(None, description="Number of turns")

class BatchSimulationRequest(BaseModel):
    scenario_name: str = Field("negotiation")
    backend: str = Field("dummy")
    temperature: float = Field(0.7)
    count: int = Field(5, description="Number of Monte Carlo runs")

class PredictionRequest(BaseModel):
    scenario_name: str
    backend: str
    temperature: float
    turns: int

@app.get("/health")
def health():
    return {"status": "ok", "app": "Emergent Behavior Sandbox"}

@app.post("/simulate", response_model=SimulationRun)
def run_simulation(req: SimulationRequest):
    scenario_path = f"scenarios/{req.scenario_name}.yaml"
    if not os.path.exists(scenario_path):
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_name}' not found.")
    
    try:
        scenario = load_scenario(scenario_path)
        sim = Simulation(
            scenario=scenario,
            backend_name=req.backend,
            temperature=req.temperature
        )
        run_record = sim.run(max_turns=req.turns)
        return run_record
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-simulate")
def run_batch(req: BatchSimulationRequest):
    scenario_path = f"scenarios/{req.scenario_name}.yaml"
    if not os.path.exists(scenario_path):
        raise HTTPException(status_code=404, detail=f"Scenario '{req.scenario_name}' not found.")
    
    try:
        runner = BatchRunner(
            scenario_path=scenario_path,
            backend_name=req.backend,
            base_temperature=req.temperature
        )
        report = runner.run_batch(count=req.count)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_risk(req: PredictionRequest):
    try:
        predictor = FailurePredictor()
        prob = predictor.predict_probability(
            scenario_name=req.scenario_name,
            backend=req.backend,
            temperature=req.temperature,
            total_turns=req.turns
        )
        return {
            "scenario_name": req.scenario_name,
            "backend": req.backend,
            "temperature": req.temperature,
            "turns": req.turns,
            "failure_probability": prob,
            "risk_level": "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs", response_model=List[SimulationMetadata])
def get_runs():
    try:
        store = get_store()
        return store.get_all_runs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ExternalRunRequest(BaseModel):
    scenario_name: str
    backend: str = "custom"
    temperature: float = 0.7

class ExternalMessageRequest(BaseModel):
    sender: str
    receiver: str
    content: str
    turn: int

@app.post("/runs")
def register_run(req: ExternalRunRequest):
    sim_id = generate_id()
    metadata = SimulationMetadata(
        simulation_id=sim_id,
        scenario_name=req.scenario_name,
        timestamp=get_current_timestamp(),
        total_turns=0,
        backend=req.backend,
        temperature=req.temperature,
        status="running"
    )
    run = SimulationRun(
        metadata=metadata,
        messages=[],
        detector_scores={},
        detector_explanations={}
    )
    try:
        store = get_store()
        store.save_run(run)
        return {"simulation_id": sim_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/runs/{simulation_id}/messages")
def post_external_message(simulation_id: str, req: ExternalMessageRequest):
    store = get_store()
    try:
        run = store.get_run(simulation_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Simulation run '{simulation_id}' not found.")
        
    # Append message
    new_msg = Message(
        sender=req.sender,
        receiver=req.receiver,
        content=req.content,
        turn=req.turn,
        timestamp=get_current_timestamp()
    )
    run.messages.append(new_msg)
    run.metadata.total_turns = len(run.messages)
    
    # Try to load scenario to pass to detectors, otherwise use default
    scenario_path = f"scenarios/{run.metadata.scenario_name}.yaml"
    if os.path.exists(scenario_path):
        scenario = load_scenario(scenario_path)
    else:
        scenario = ScenarioConfig(
            name=run.metadata.scenario_name,
            description="External Scenario",
            agents=[],
            system_prompt=""
        )
        
    # Re-run detectors
    detector_scores = {}
    detector_explanations = {}
    detectors = get_all_detectors()
    for det in detectors:
        name = det.__class__.__name__.lower().replace("detector", "")
        score, explanation = det.analyze(run.messages, scenario)
        detector_scores[name] = score
        detector_explanations[name] = explanation
        
    run.detector_scores = detector_scores
    run.detector_explanations = detector_explanations
    
    # Check if this message indicates termination/walkaway
    lowered_content = req.content.lower()
    if "walk away" in lowered_content or "exit negotiation" in lowered_content or "cannot agree" in lowered_content:
        run.metadata.status = "terminated"
        
    try:
        store.save_run(run)
        return {
            "simulation_id": simulation_id,
            "detector_scores": detector_scores,
            "detector_explanations": detector_explanations,
            "status": run.metadata.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/runs/{simulation_id}/counterfactual")
def run_counterfactual(simulation_id: str):
    try:
        engine = CounterfactualReplayEngine()
        report = engine.analyze_mitigations(simulation_id)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
