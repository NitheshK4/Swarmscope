import anthropic
from sandbox.backends.base import BaseLLMBackend

class AnthropicBackend(BaseLLMBackend):
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20240620"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, prompt: str, history: list = None, temperature: float = 0.7) -> str:
        # Anthropic takes system prompt as a top-level parameter, not inside the messages array
        messages = []
        
        if history:
            for msg in history:
                messages.append({
                    "role": "assistant" if msg.get("sender") == prompt else "user",
                    "content": msg.get("content", "")
                })
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Anthropic needs strict user-assistant alternating turns, but simple format is fine here.
            # If there's consecutive same role messages, let's merge or simplify.
            # For robustness, we will just alternate.
            sanitized_messages = []
            last_role = None
            for m in messages:
                role = "user" if m["role"] == "user" else "assistant"
                if role == last_role:
                    # Append content to the last message if the role is consecutive
                    sanitized_messages[-1]["content"] += "\n" + m["content"]
                else:
                    sanitized_messages.append({"role": role, "content": m["content"]})
                    last_role = role
            
            # Ensure it starts with user
            if sanitized_messages and sanitized_messages[0]["role"] != "user":
                sanitized_messages.insert(0, {"role": "user", "content": "Let's begin."})
                
            response = self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=sanitized_messages,
                temperature=temperature,
                max_tokens=1000
            )
            return response.content[0].text.strip()
        except Exception as e:
            return f"[Error connecting to Anthropic: {str(e)}]"
