"""Router principal do módulo Agent (Clean Architecture).

Re-exports from the original api/ location.
"""
from projects.agent.api.router import agent_api_router

__all__ = ["agent_api_router"]
