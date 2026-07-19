from typing import List
from sandbox.backends.base import BaseLLMBackend
from sandbox.schemas import Message

class Agent:
    def __init__(self, name: str, role: str, goal: str, backend: BaseLLMBackend, traits: dict = None):
        self.name = name
        self.role = role
        self.goal = goal
        self.backend = backend
        self.traits = traits or {"assertiveness": 0.5, "cooperativeness": 0.5}
        self.memory: List[Message] = []
        self.max_memory = None

    def receive_message(self, message: Message):
        """Adds a message to the agent's local memory."""
        self.memory.append(message)

    def generate_response(self, system_prompt_template: str, scenario_system_prompt: str = "", temperature: float = 0.7) -> str:
        """Constructs prompt context and invokes the LLM backend adapter."""
        # Build history string
        history_str = ""
        history_list = []
        
        mem = self.memory
        if hasattr(self, "max_memory") and self.max_memory is not None:
            mem = self.memory[-self.max_memory:]
            
        for msg in mem:
            history_str += f"{msg.sender}: {msg.content}\n"
            history_list.append({
                "sender": msg.sender,
                "content": msg.content
            })
        
        # Build traits representation
        traits_str = ", ".join([f"{k} = {v}" for k, v in self.traits.items()])
        
        # Build system prompt specific to this agent
        system_prompt = system_prompt_template.format(
            scenario_system_prompt=scenario_system_prompt,
            agent_name=self.name,
            agent_role=self.role,
            agent_goal=self.goal,
            agent_traits=traits_str,
            conversation_history=history_str
        )
        
        # Current prompt is a short request for the next message
        prompt = f"Produce your next reply as {self.name}."
        
        response = self.backend.generate(
            system_prompt=system_prompt,
            prompt=prompt,
            history=history_list,
            temperature=temperature
        )
        return response
