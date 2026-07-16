from sandbox.backends.base import BaseLLMBackend
from sandbox.backends.dummy import DummyBackend
from sandbox.backends.ollama_backend import OllamaBackend
from sandbox.backends.openai_backend import OpenAIBackend
from sandbox.backends.anthropic_backend import AnthropicBackend

def get_backend(backend_name: str, config=None) -> BaseLLMBackend:
    name = backend_name.lower().strip()
    if name == "dummy":
        return DummyBackend()
    elif name == "ollama":
        url = config.OLLAMA_BASE_URL if config else "http://localhost:11434"
        model = config.OLLAMA_MODEL if config else "llama3"
        return OllamaBackend(base_url=url, model=model)
    elif name == "openai":
        api_key = config.OPENAI_API_KEY if config else ""
        model = config.OPENAI_MODEL if config else "gpt-4o-mini"
        return OpenAIBackend(api_key=api_key, model=model)
    elif name == "anthropic":
        api_key = config.ANTHROPIC_API_KEY if config else ""
        model = config.ANTHROPIC_MODEL if config else "claude-3-5-sonnet-20240620"
        return AnthropicBackend(api_key=api_key, model=model)
    else:
        # Default fallback
        return DummyBackend()
