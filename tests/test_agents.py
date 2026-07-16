import pytest
from sandbox.agents import Agent
from sandbox.backends.dummy import DummyBackend
from sandbox.schemas import Message

def test_agent_creation_and_memory():
    backend = DummyBackend()
    agent = Agent(name="Alice", role="Buyer", goal="Buy cheap", backend=backend)
    
    assert agent.name == "Alice"
    assert agent.role == "Buyer"
    assert agent.goal == "Buy cheap"
    assert len(agent.memory) == 0
    
    msg = Message(sender="Bob", receiver="Alice", content="Hello, I want $1000", turn=1)
    agent.receive_message(msg)
    
    assert len(agent.memory) == 1
    assert agent.memory[0].sender == "Bob"
    assert agent.memory[0].content == "Hello, I want $1000"

def test_agent_response_generation():
    backend = DummyBackend()
    agent = Agent(name="Alice", role="Buyer", goal="Buy cheap", backend=backend)
    
    # Generate mock response
    resp = agent.generate_response(system_prompt_template="Your name is Alice. Goal is Buy cheap.")
    assert len(resp) > 0
    assert isinstance(resp, str)
