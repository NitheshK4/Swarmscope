from sandbox.backends import get_backend
from sandbox.backends.anthropic_backend import AnthropicBackend
from sandbox.backends.base import BackendError, BaseLLMBackend, LLMResponse
from sandbox.backends.dummy import DummyBackend
from sandbox.backends.ollama_backend import OllamaBackend
from sandbox.backends.openai_backend import OpenAIBackend


def test_backend_instantiation_and_types():
    dummy = get_backend("dummy")
    assert isinstance(dummy, BaseLLMBackend)
    assert isinstance(dummy, DummyBackend)

    ollama = get_backend("ollama")
    assert isinstance(ollama, BaseLLMBackend)
    assert isinstance(ollama, OllamaBackend)

    openai_backend = get_backend("openai")
    assert isinstance(openai_backend, BaseLLMBackend)
    assert isinstance(openai_backend, OpenAIBackend)

    anthropic_backend = get_backend("anthropic")
    assert isinstance(anthropic_backend, BaseLLMBackend)
    assert isinstance(anthropic_backend, AnthropicBackend)

    unknown = get_backend("nonexistent")
    assert isinstance(unknown, DummyBackend)

def test_normalized_llm_response():
    dummy = DummyBackend()
    res = dummy.generate_response(system_prompt="Test", prompt="Hello", history=[], temperature=0.7)
    assert isinstance(res, LLMResponse)
    assert res.content != ""
    assert res.backend_name in ["dummy", "base"]

def test_backend_error_structure():
    err = BackendError("openai", "Missing API key")
    assert "openai" in str(err)
    assert err.backend_name == "openai"


def test_backend_retry_success_and_failure():
    import pytest

    dummy = DummyBackend(max_retries=3, backoff_factor=0.01)
    attempts = 0

    def flaky_func():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise RuntimeError("Transient error")
        return "success"

    res = dummy.with_retry(flaky_func)
    assert res == "success"
    assert attempts == 2

    def failing_func():
        raise ValueError("Persistent error")

    with pytest.raises(BackendError) as exc_info:
        dummy.with_retry(failing_func)
    assert "Failed after 3 attempts" in str(exc_info.value)

