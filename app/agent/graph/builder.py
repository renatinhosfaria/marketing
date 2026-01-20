"""
Construção do grafo LangGraph para o agente de tráfego pago.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from app.agent.graph.state import AgentState
from app.agent.graph.nodes import (
    classify_intent,
    gather_data,
    call_model,
    call_tools,
    generate_response,
    handle_error,
)
from app.agent.graph.edges import (
    should_continue,
    route_by_intent,
    after_tools,
    check_data_quality,
)
from app.agent.config import get_agent_settings
from app.config import settings as app_settings


def build_agent_graph() -> StateGraph:
    """
    Constrói o grafo do agente de tráfego pago.

    Fluxo do grafo:
    1. START -> classify_intent: Classifica a intenção do usuário
    2. classify_intent -> route_by_intent: Decide se precisa de dados
    3. gather_data (opcional) -> call_model: Coleta dados do ML
    4. call_model -> should_continue: Chama o LLM
    5. call_tools (se necessário) -> call_model: Executa ferramentas
    6. generate_response -> END: Finaliza resposta

    Returns:
        StateGraph configurado e compilado
    """
    # Criar o grafo com o estado do agente
    graph = StateGraph(AgentState)

    # ==========================================
    # Adicionar nós
    # ==========================================
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("gather_data", gather_data)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", call_tools)
    graph.add_node("generate_response", generate_response)
    graph.add_node("handle_error", handle_error)

    # ==========================================
    # Adicionar arestas
    # ==========================================

    # Início -> Classificar intenção
    graph.add_edge(START, "classify_intent")

    # Classificar -> Decidir se precisa de dados
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "gather_data": "gather_data",
            "call_model": "call_model",
        }
    )

    # Coletar dados -> Verificar qualidade -> Modelo
    graph.add_conditional_edges(
        "gather_data",
        check_data_quality,
        {
            "call_model": "call_model",
            "handle_error": "handle_error",
        }
    )

    # Modelo -> Decidir próximo passo
    graph.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "call_tools": "call_tools",
            "generate_response": "generate_response",
            "handle_error": "handle_error",
        }
    )

    # Ferramentas -> Voltar para modelo
    graph.add_conditional_edges(
        "call_tools",
        after_tools,
        {
            "call_model": "call_model",
            "generate_response": "generate_response",
        }
    )

    # Gerar resposta -> Fim
    graph.add_edge("generate_response", END)

    # Tratar erro -> Fim
    graph.add_edge("handle_error", END)

    return graph


async def get_checkpointer() -> AsyncPostgresSaver:
    """
    Obtém o checkpointer PostgreSQL para persistência de estado.

    Usa a mesma conexão do banco de dados da aplicação.

    Returns:
        AsyncPostgresSaver configurado
    """
    # Usar a URL do banco de dados da aplicação
    database_url = app_settings.database_url

    # Converter URL para formato asyncpg se necessário
    if database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+asyncpg://", 1)

    checkpointer = AsyncPostgresSaver.from_conn_string(database_url)

    # Configurar tabelas (se ainda não existirem)
    await checkpointer.setup()

    return checkpointer


async def create_agent():
    """
    Cria e compila o agente com persistência de estado.

    Returns:
        Agente compilado pronto para uso
    """
    graph = build_agent_graph()
    checkpointer = await get_checkpointer()

    # Compilar o grafo com o checkpointer
    agent = graph.compile(checkpointer=checkpointer)

    return agent


def create_agent_sync():
    """
    Cria o agente sem persistência (para testes).

    Returns:
        Agente compilado sem checkpointer
    """
    graph = build_agent_graph()
    return graph.compile()
