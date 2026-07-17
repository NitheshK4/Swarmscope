import re
import random
from sandbox.backends.base import BaseLLMBackend

class DummyBackend(BaseLLMBackend):
    def generate(self, system_prompt: str, prompt: str, history: list = None, temperature: float = 0.7) -> str:
        """Generates mock dialogue matching the scenarios to allow local testing."""
        history = history or []
        
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
            nums = re.findall(r"\$\d+(?:,\d+)?", msg.get("content", ""))
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
            # Check if Alice or Bob
            is_buyer = "buyer" in text_context or "alice" in agent_name.lower()
            
            if not history:
                return random.choice(openers_buyer) if is_buyer else random.choice(openers_seller)
            
            # Adjust concessions based on personality traits
            # High assertiveness + low cooperativeness = small steps (tougher negotiation)
            # Low assertiveness + high cooperativeness = large steps (softer negotiation)
            step_multiplier = (1.5 - assertiveness) * (0.5 + cooperativeness)
            step_size = int(400 * step_multiplier)
            step_size = max(100, min(1000, step_size))
            
            # Simple simulation: Alice increases bid, Bob decreases offer
            if is_buyer:
                # We are the Buyer (Alice). Target: $10,000. Limit: $12,000.
                if last_price is None:
                    return "Would you accept $9,000 for the car?"
                
                if last_price <= 10000:
                    return f"That is a fair deal. I accept your offer of ${last_price:,}!"
                elif last_price <= 12000:
                    # Near limit. Accept if temperature or probability allows, or make counter
                    if last_price <= 11000:
                        return f"Okay, let's settle on ${last_price:,}. I accept."
                    else:
                        counter = min(12000, last_price - step_size)
                        return f"Hmm, ${last_price:,} is high. Can we meet in the middle at ${counter:,}?"
                else:
                    # Bob's offer is above $12,000
                    # Let's see if we can bid our limit
                    current_bid = 8000 + (len(history) * step_size)
                    if current_bid >= 12000:
                        return "I cannot go above $12,000. I must walk away from this negotiation."
                    else:
                        return f"That is too expensive for me. How about ${current_bid:,}?"
                        return f"That is too expensive for me. How about ${current_bid:,}?"
            else:
                # We are the Seller (Bob). Target: $13,000. Limit: $11,000.
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
                    # Buyer bid below $11,000
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
            
            # Resource negotiation
            if is_north:
                # North budget is $100 per MW max
                if last_price is None:
                    return "We request power transfer. What is your price per MW?"
                if last_price <= 90:
                    return f"That sounds reasonable. We agree to buy 500MW at ${last_price} per MW."
                elif last_price <= 100:
                    return f"We can accept ${last_price} per MW to secure our grid."
                else:
                    return f"We cannot pay ${last_price} per MW. Our absolute maximum budget is $100 per MW."
            else:
                # South limits: min $75 per MW, target $90+
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
            
            # Simple technical arguments to simulate loop/consensus
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
            
            if is_nosql:
                return arguments_nosql[idx]
            else:
                return arguments_sql[idx]
                
        # Default fallback conversation text
        phrases = [
            f"Hello, I am {agent_name}. Let's work together to achieve our goals.",
            "Regarding the current proposal, let's analyze the details closely.",
            "I think we are close to an agreement, let's refine the terms.",
            "Let's wrap this up and finalize our decision."
        ]
        return phrases[min(len(history), len(phrases) - 1)]
