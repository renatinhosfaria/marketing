"""
Prompts do agente de tráfego.
"""

from app.agent.prompts.system import SYSTEM_PROMPT, get_system_prompt
from app.agent.prompts.templates import ResponseTemplates

__all__ = [
    "SYSTEM_PROMPT",
    "get_system_prompt",
    "ResponseTemplates",
]
