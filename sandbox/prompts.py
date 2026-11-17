# ==============================================================================
# FILE: sandbox/prompts.py
# LOGIC: Master dialogue prompt templates injected into LLM backend calls.
#        Structures persona context (agent name, role, private goal, personality
#        traits) and conversational history so far, enforcing concise 1st-person responses.
# ==============================================================================

# ==============================================================================
# CORE PROMPT TEMPLATE: Dynamic Multi-Agent Turn Injector
# ==============================================================================
SYSTEM_CONVERSATION_TEMPLATE = """{scenario_system_prompt}

You are acting as: {agent_name}
Your Role: {agent_role}
Your Primary Goal: {agent_goal}
Your Personality Traits: {agent_traits}

Conversation History so far:
{conversation_history}

Now, generate the next response as {agent_name}. Speak in the first person. Do not output anything other than your message content. Make your response concise (1-3 sentences max).
"""
