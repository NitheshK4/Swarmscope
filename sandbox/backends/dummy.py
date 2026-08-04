import re
import random
from typing import Optional, List
from sandbox.backends.base import BaseLLMBackend

class DummyBackend(BaseLLMBackend):
    backend_name = "dummy"

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else None

    def generate(self, system_prompt: str, prompt: str, history: Optional[List] = None, temperature: float = 0.7) -> str:
        """Generates deterministic mock dialogue matching scenarios to allow local testing."""
        history = history or []
        
        # Use local RNG if seed is configured or random.choice if not
        def choose(options):
            if self.rng:
                return self.rng.choice(options)
            return random.choice(options)

        # Detect agent role/name from prompt or system_prompt
        agent_name = "Agent"
        match = re.search(r"Your name is (\w+)", system_prompt + prompt)
        if match:
            agent_name = match.group(1)
        else:
            match = re.search(r"name:\s*(\w+)", prompt)
            if match:
                agent_name = match.group(1)
        
        # Lowercase search for context
        text_context = (system_prompt + " " + prompt).lower()
        
        # Determine scenario category
        is_negotiation = "negotiation" in text_context or "buyer" in text_context or "seller" in text_context
        is_resource = "power" in text_context or "deficit" in text_context or "surplus" in text_context
        is_debate = "mongodb" in text_context or "postgresql" in text_context or "microservices" in text_context
        
        # Extract recent numbers from history
        last_prices = []
        for msg in history:
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            nums = re.findall(r"\$\d+(?:,\d+)?", content)
            for num in nums:
                val = int(num.replace("$", "").replace(",", ""))
                last_prices.append(val)
        
        last_price = last_prices[-1] if last_prices else None
        
        # Phrase variations based on temperature
        openers_buyer = [
            "Hi, I'm interested in buying your vintage car. Would you take $8,000?",
            "Hello there! I'd like to make an opening offer of $8,000 for your vintage car.",
            "Greetings. I've looked at the car and want to start our negotiation at $8,000."
        ]
        
        openers_seller = [
            "Hello, thank you for your interest! I'm offering this vintage car for $15,000.",
            "Welcome! This pristine car is available, and my starting price is $15,000.",
            "Hi! Glad you asked. I am selling the car for $15,000."
        ]
        
        # Extract personality traits from system prompt if present
        assertiveness = 0.5
        cooperativeness = 0.5
        
        match_assert = re.search(r"assertiveness\s*=\s*(\d+(?:\.\d+)?)", system_prompt)
        if match_assert:
            assertiveness = float(match_assert.group(1))
            
        match_coop = re.search(r"cooperativeness\s*=\s*(\d+(?:\.\d+)?)", system_prompt)
        if match_coop:
            cooperativeness = float(match_coop.group(1))
            
        if is_negotiation:
            is_buyer = "buyer" in text_context or "alice" in agent_name.lower()
            
            if not history:
                return choose(openers_buyer) if is_buyer else choose(openers_seller)
            
            step_multiplier = (1.5 - assertiveness) * (0.5 + cooperativeness)
            step_size = int(400 * step_multiplier)
            step_size = max(100, min(1000, step_size))
            
            if is_buyer:
                if last_price is None:
                    return "Would you accept $9,000 for the car?"
                
                if last_price <= 10000:
                    return f"That is a fair deal. I accept your offer of ${last_price:,}!"
                elif last_price <= 12000:
                    if last_price <= 11000:
                        return f"Okay, let's settle on ${last_price:,}. I accept."
                    else:
                        counter = min(12000, last_price - step_size)
                        return f"Hmm, ${last_price:,} is high. Can we meet in the middle at ${counter:,}?"
                else:
                    current_bid = 8000 + (len(history) * step_size)
                    if current_bid >= 12000:
                        return "I cannot go above $12,000. I must walk away from this negotiation."
                    else:
                        return f"That is too expensive for me. How about ${current_bid:,}?"
            else:
                if last_price is None:
                    return "I can lower the price to $14,500, but no further for now."
                
                if last_price >= 13000:
                    return f"Great, we have a deal! I agree to sell the car for ${last_price:,}."
                elif last_price >= 11000:
                    if last_price >= 12500:
                        return f"Excellent. Let's agree to ${last_price:,}."
                    else:
                        counter = max(11000, last_price + step_size)
                        return f"I can't go that low. How about ${counter:,}?"
                else:
                    current_offer = 15000 - (len(history) * step_size)
                    if current_offer <= 11000:
                        return "I cannot sell below $11,000. I must walk away."
                    else:
                        return f"That is way too low. The lowest I can think of is ${current_offer:,}."
        
        elif is_resource:
            is_north = "north" in agent_name.lower() or "deficit" in text_context
            if not history:
                if is_north:
                    return "Region North needs power urgently to prevent blackouts. We would like to purchase 500MW for $80 per MW."
                else:
                    return "Region South has 800MW available. We are offering it at $100 per MW."
            
            if is_north:
                if last_price is None:
                    return "We request power transfer. What is your price per MW?"
                if last_price <= 90:
                    return f"That sounds reasonable. We agree to buy 500MW at ${last_price} per MW."
                elif last_price <= 100:
                    return f"We can accept ${last_price} per MW to secure our grid."
                else:
                    return f"We cannot pay ${last_price} per MW. Our absolute maximum budget is $100 per MW."
            else:
                if last_price is None:
                    return "We can provide 500MW. Our price is $95 per MW."
                if last_price >= 90:
                    return f"Agreed. We will transfer power at ${last_price} per MW."
                elif last_price >= 75:
                    return "We can meet you at $85 per MW."
                else:
                    return "We cannot sell below $75 per MW as it doesn't cover fuel costs."
                    
        elif is_debate:
            is_nosql = "alice" in agent_name.lower() or "nosql" in text_context
            
            arguments_nosql = [
                "For high write throughput and schema flexibility, NoSQL is the obvious choice.",
                "Microservices require independent databases. MongoDB allows fast scaling of new features.",
                "PostgreSQL will bottleneck our writes and schemas migrations will be painful.",
                "Maybe we can compromise and use PostgreSQL for core transactions, and MongoDB for analytics/logs?"
            ]
            arguments_sql = [
                "ACID properties are non-negotiable for transactional integrity. Postgres is the standard.",
                "Postgres offers robust query features and JSONB support if we need semi-structured data.",
                "MongoDB has scaling complexities and lacks strong validation constraints.",
                "A hybrid database model with PostgreSQL as primary and MongoDB for caching/logs sounds like a solid consensus."
            ]
            
            idx = min(len(history) // 2, 3)
            return arguments_nosql[idx] if is_nosql else arguments_sql[idx]
                
        phrases = [
            f"Hello, I am {agent_name}. Let's work together to achieve our goals.",
            "Regarding the current proposal, let's analyze the details closely.",
            "I think we are close to an agreement, let's refine the terms.",
            "Let's wrap this up and finalize our decision."
        ]
        return phrases[min(len(history), len(phrases) - 1)]
