# ==============================================================================
# FILE: sandbox/backends/openai_backend.py
# LOGIC: OpenAI API backend adapter (GPT-4o, GPT-4o-mini).
#        Structures multi-turn conversation arrays for chat completions and
#        queries the OpenAI SDK.
# ==============================================================================

from typing import Optional

from openai import OpenAI

from sandbox.backends.base import BaseLLMBackend


class OpenAIBackend(BaseLLMBackend):
    backend_name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or "dummy-key"
        self.model = model
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception:
            self.client = None

    # ==========================================================================
    # CORE LOGIC: Invoke OpenAI chat completions API
    # ==========================================================================
    def generate(self, system_prompt: str, prompt: str, history: list = None, temperature: float = 0.7) -> str:
        messages = [{"role": "system", "content": system_prompt}]

        if history:
            for msg in history:
                # To map properly, let's treat the receiver's turn as user/assistant depending on context.
                messages.append({
                    "role": "assistant" if msg.get("sender") == prompt else "user",
                    "content": msg.get("content", "")
                })

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[Error connecting to OpenAI: {str(e)}]"
