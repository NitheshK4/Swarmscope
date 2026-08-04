from sandbox.backends.dummy import DummyBackend

def test_dummy_backend_determinism():
    b1 = DummyBackend(seed=42)
    res1_1 = b1.generate("You are Alice buyer", "negotiation", [], temperature=0.7)
    
    b2 = DummyBackend(seed=42)
    res2_1 = b2.generate("You are Alice buyer", "negotiation", [], temperature=0.7)

    assert res1_1 == res2_1

def test_dummy_backend_conversation_patterns():
    backend = DummyBackend(seed=100)
    # Negotiation pattern
    neg_response = backend.generate(
        system_prompt="Your name is Alice buyer assertiveness=0.8",
        prompt="negotiation offer",
        history=[{"sender": "Bob", "content": "I am selling for $15,000"}],
    )
    assert "$" in neg_response or "expensive" in neg_response or "accept" in neg_response

    # Debate pattern
    debate_response = backend.generate(
        system_prompt="Your name is Alice NoSQL",
        prompt="mongodb debate",
        history=[],
    )
    assert "NoSQL" in debate_response or "MongoDB" in debate_response
