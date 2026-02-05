"""
Módulo de integração com LLMs.
"""

from projects.agent.llm.provider import get_llm, get_llm_with_tools, LLMProvider

__all__ = [
    "get_llm",
    "get_llm_with_tools",
    "LLMProvider",
]
