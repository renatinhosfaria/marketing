"""
Prompts do agente de tráfego.
"""

from projects.agent.prompts.system import SYSTEM_PROMPT, get_system_prompt
from projects.agent.prompts.templates import ResponseTemplates

__all__ = [
    "SYSTEM_PROMPT",
    "get_system_prompt",
    "ResponseTemplates",
]
