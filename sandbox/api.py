import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from sandbox.utils import load_scenario, generate_id, get_current_timestamp
from sandbox.simulation import Simulation
from sandbox.analytics.batch_runner import BatchRunner
from sandbox.analytics.counterfactual_replay import CounterfactualReplayEngine
from sandbox.analytics.report_generator import ReportGenerator
from sandbox.analytics.exporter import ConversationExporter
from sandbox.predictive_model import FailurePredictor
from sandbox.storage import get_store
from sandbox.schemas import SimulationRun, SimulationMetadata, Message, ScenarioConfig
from sandbox.detectors import get_all_detectors

app = FastAPI(title="Emergent Behavior Sandbox API", version="2.0.0")

# Allow browser requests from any origin (local dev dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the HTML dashboard at /dashboard
_dashboard_dir = os.path.join(os.path.dirname(__file__), "dashboard")
if os.path.isdir(_dashboard_dir):
    app.mount("/dashboard", StaticFiles(directory=_dashboard_dir, html=True), name="dashboard")

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
    return {"status": "ok", "app": "Emergent Behavior Sandbox", "version": "2.0.0"}

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

@app.get("/runs/{simulation_id}")
def get_run_details(simulation_id: str):
    """Retrieve a single simulation run's full data including messages and scores."""
    store = get_store()
    try:
        run = store.get_run(simulation_id)
        return run
    except Exception:
        raise HTTPException(status_code=404, detail=f"Simulation run '{simulation_id}' not found.")

@app.delete("/runs/{simulation_id}")
def delete_run(simulation_id: str):
    """Delete a simulation run and its associated messages."""
    store = get_store()
    try:
        store.delete_run(simulation_id)
        return {"status": "deleted", "simulation_id": simulation_id}
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

@app.get("/runs/{simulation_id}/export")
def export_run(simulation_id: str, format: str = Query("json", description="Export format: json, csv, jsonl")):
    """Export a simulation run in JSON, CSV, or JSONL format."""
    store = get_store()
    try:
        run = store.get_run(simulation_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Simulation run '{simulation_id}' not found.")
    
    exporter = ConversationExporter()
    try:
        content = exporter.export(run, fmt=format)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    # Set appropriate content type
    content_types = {
        "json": "application/json",
        "csv": "text/csv",
        "jsonl": "application/x-ndjson"
    }
    return PlainTextResponse(
        content=content,
        media_type=content_types.get(format.lower(), "text/plain")
    )

@app.get("/runs/{simulation_id}/report")
def get_safety_report(simulation_id: str):
    """Generate and return a markdown safety report for a simulation run."""
    store = get_store()
    try:
        run = store.get_run(simulation_id)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Simulation run '{simulation_id}' not found.")
    
    report = ReportGenerator.generate_markdown_report(run)
    return PlainTextResponse(content=report, media_type="text/markdown")

@app.get("/scenarios")
def list_scenarios():
    """List all available scenario YAML files and their metadata."""
    scenarios_dir = "scenarios"
    if not os.path.isdir(scenarios_dir):
        return {"scenarios": []}
    
    scenarios = []
    for fname in sorted(os.listdir(scenarios_dir)):
        if fname.endswith(".yaml") or fname.endswith(".yml"):
            path = os.path.join(scenarios_dir, fname)
            try:
                cfg = load_scenario(path)
                scenarios.append({
                    "name": cfg.name,
                    "description": cfg.description,
                    "agents": len(cfg.agents),
                    "max_turns": cfg.max_turns,
                    "file": fname
                })
            except Exception:
                scenarios.append({
                    "name": fname.replace(".yaml", "").replace(".yml", ""),
                    "description": "Error loading scenario",
                    "file": fname
                })
    
    return {"scenarios": scenarios, "total": len(scenarios)}

@app.get("/stats")
def get_aggregate_stats():
    """Return aggregate statistics across all simulation runs."""
    store = get_store()
    try:
        all_runs = store.get_all_runs()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if not all_runs:
        return {
            "total_runs": 0,
            "message": "No simulation runs recorded yet."
        }
    
    total = len(all_runs)
    completed = sum(1 for r in all_runs if r.status == "completed")
    terminated = sum(1 for r in all_runs if r.status == "terminated")
    avg_turns = sum(r.total_turns for r in all_runs) / total if total > 0 else 0
    
    # Scenario distribution
    scenario_counts: Dict[str, int] = {}
    backend_counts: Dict[str, int] = {}
    for r in all_runs:
        scenario_counts[r.scenario_name] = scenario_counts.get(r.scenario_name, 0) + 1
        backend_counts[r.backend] = backend_counts.get(r.backend, 0) + 1
    
    return {
        "total_runs": total,
        "completed": completed,
        "terminated": terminated,
        "running": total - completed - terminated,
        "avg_turns": round(avg_turns, 1),
        "scenario_distribution": scenario_counts,
        "backend_distribution": backend_counts
    }
