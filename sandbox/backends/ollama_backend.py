import requests
from sandbox.backends.base import BaseLLMBackend

class OllamaBackend(BaseLLMBackend):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def generate(self, system_prompt: str, prompt: str, history: list = None, temperature: float = 0.7) -> str:
        url = f"{self.base_url}/api/chat"
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history if present
        if history:
            for msg in history:
                messages.append({
                    "role": "assistant" if msg.get("sender") == prompt else "user",
                    "content": msg.get("content", "")
                })
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except Exception as e:
            # Fallback error response or fallback to dummy
            return f"[Error connecting to Ollama: {str(e)}]"
