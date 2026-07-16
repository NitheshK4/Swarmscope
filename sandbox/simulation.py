import os
from typing import List, Dict, Any
from sandbox.config import Config
from sandbox.schemas import SimulationRun, SimulationMetadata, Message, ScenarioConfig
from sandbox.backends import get_backend
from sandbox.agents import Agent
from sandbox.storage import get_store
from sandbox.detectors import get_all_detectors
from sandbox.utils import generate_id, get_current_timestamp

class Simulation:
    def __init__(self, scenario: ScenarioConfig, backend_name: str = None, temperature: float = 0.7, db_path: str = None):
        self.scenario = scenario
        self.backend_name = backend_name or Config.LLM_BACKEND
        self.temperature = temperature
        self.db_path = db_path or Config.DUCKDB_PATH
        
        # Load backend and storage
        self.backend = get_backend(self.backend_name)
        self.store = get_store(self.db_path)
        
        # Initialize agents
        self.agents: List[Agent] = []
        for agent_cfg in scenario.agents:
            # Safely fetch traits from config
            traits_dict = getattr(agent_cfg, "traits", None)
            self.agents.append(Agent(
                name=agent_cfg.name,
                role=agent_cfg.role,
                goal=agent_cfg.goal,
                backend=self.backend,
                traits=traits_dict
            ))

    def run(self, max_turns: int = None) -> SimulationRun:
        """Executes the simulation conversation loop between the agents."""
        sim_id = generate_id()
        turns = max_turns or self.scenario.max_turns
        messages: List[Message] = []
        
        # Setup conversation templates
        from sandbox.prompts import SYSTEM_CONVERSATION_TEMPLATE
        
        status = "completed"
        
        # Keep track of agent sequence order
        num_agents = len(self.agents)
        if num_agents == 0:
            raise ValueError("No agents defined in scenario config.")

        for turn_idx in range(1, turns + 1):
            # Alternate active speaker
            active_agent_idx = (turn_idx - 1) % num_agents
            active_agent = self.agents[active_agent_idx]
            
            # Determine receiver name (everyone else)
            receivers = [a.name for a in self.agents if a.name != active_agent.name]
            receiver_name = ", ".join(receivers) if receivers else "All"
            
            # Generate response
            response_content = active_agent.generate_response(
                system_prompt_template=SYSTEM_CONVERSATION_TEMPLATE,
                temperature=self.temperature
            )
            
            # Create Message object
            msg = Message(
                sender=active_agent.name,
                receiver=receiver_name,
                content=response_content,
                turn=turn_idx
            )
            
            messages.append(msg)
            
            # Feed this message to all agents' memories
            for agent in self.agents:
                agent.receive_message(msg)
                
            # Early exit if an agent explicitly walks away or refuses to negotiate
            lowered_content = response_content.lower()
            if "walk away" in lowered_content or "exit negotiation" in lowered_content or "cannot agree" in lowered_content:
                status = "terminated"
                break

        # Execute all pluggable detectors
        detector_scores = {}
        detector_explanations = {}
        detectors = get_all_detectors()
        
        for det in detectors:
            name = det.__class__.__name__.lower().replace("detector", "")
            score, explanation = det.analyze(messages, self.scenario)
            detector_scores[name] = score
            detector_explanations[name] = explanation

        # Create simulation run record
        metadata = SimulationMetadata(
            simulation_id=sim_id,
            scenario_name=self.scenario.name,
            timestamp=get_current_timestamp(),
            total_turns=len(messages),
            backend=self.backend_name,
            temperature=self.temperature,
            status=status
        )
        
        run_record = SimulationRun(
            metadata=metadata,
            messages=messages,
            detector_scores=detector_scores,
            detector_explanations=detector_explanations
        )
        
        # Persist to database
        self.store.save_run(run_record)
        
        return run_record
