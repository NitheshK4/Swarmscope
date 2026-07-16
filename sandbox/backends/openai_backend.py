from openai import OpenAI
from sandbox.backends.base import BaseLLMBackend

class OpenAIBackend(BaseLLMBackend):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

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
