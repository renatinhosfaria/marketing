"""
Nós do grafo LangGraph para o agente de tráfego pago.
"""

from typing import Any, Dict, List
import json
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from app.agent.graph.state import AgentState
from app.agent.llm.provider import get_llm
from app.agent.prompts.system import get_system_prompt, INTENT_PROMPTS
from app.agent.config import get_agent_settings
from app.agent.tools import get_all_tools


settings = get_agent_settings()


async def classify_intent(state: AgentState) -> Dict[str, Any]:
    """
    Classifica a intenção do usuário baseado na mensagem.

    Analisa a última mensagem e determina o tipo de análise necessária:
    - analyze: Análise geral de campanhas
    - compare: Comparação entre campanhas
    - recommend: Buscar recomendações
    - forecast: Previsões de métricas
    - troubleshoot: Resolver problemas/anomalias
    - general: Conversa geral
    """
    messages = state.get("messages", [])
    if not messages:
        return {"current_intent": "general"}

    last_message = messages[-1]
    content = ""

    if isinstance(last_message, dict):
        content = last_message.get("content", "")
    elif hasattr(last_message, "content"):
        content = last_message.content

    content_lower = content.lower()

    # Palavras-chave para detecção de intenção
    intent_keywords = {
        "analyze": ["analise", "análise", "analisar", "análisar", "como está", "desempenho", "performance", "métricas", "resumo"],
        "compare": ["compare", "comparar", "comparação", "versus", "vs", "diferença", "melhor", "pior"],
        "recommend": ["recomend", "sugest", "o que fazer", "próximos passos", "ação", "conselho"],
        "forecast": ["previsão", "prever", "futuro", "projeção", "projetar", "estimar", "estimativa"],
        "troubleshoot": ["problema", "erro", "anomalia", "queda", "caiu", "piorou", "ruim", "crítico", "alerta"],
    }

    # Detectar intenção baseado em palavras-chave
    detected_intent = "general"
    max_matches = 0

    for intent, keywords in intent_keywords.items():
        matches = sum(1 for kw in keywords if kw in content_lower)
        if matches > max_matches:
            max_matches = matches
            detected_intent = intent

    return {"current_intent": detected_intent}


async def gather_data(state: AgentState) -> Dict[str, Any]:
    """
    Coleta dados relevantes do ML baseado na intenção.

    Busca classificações, recomendações, anomalias e previsões
    relevantes para a análise solicitada.
    """
    from app.agent.tools.classification_tools import get_classifications
    from app.agent.tools.recommendation_tools import get_recommendations
    from app.agent.tools.anomaly_tools import get_anomalies
    from app.agent.tools.forecast_tools import get_forecasts

    config_id = state.get("config_id")
    intent = state.get("current_intent", "general")

    updates: Dict[str, Any] = {}

    try:
        # Sempre buscar classificações (base para qualquer análise)
        classifications = await get_classifications.ainvoke({
            "config_id": config_id,
            "active_only": True,
        })
        updates["classifications"] = classifications.get("classifications", [])

        # Buscar dados específicos baseado na intenção
        if intent in ["recommend", "analyze", "general"]:
            recommendations = await get_recommendations.ainvoke({
                "config_id": config_id,
                "active_only": True,
            })
            updates["recommendations"] = recommendations.get("recommendations", [])

        if intent in ["troubleshoot", "analyze"]:
            anomalies = await get_anomalies.ainvoke({
                "config_id": config_id,
                "days": 7
            })
            updates["anomalies"] = anomalies.get("anomalies", [])

        if intent in ["forecast", "analyze"]:
            forecasts = await get_forecasts.ainvoke({
                "config_id": config_id,
                "days_ahead": 7
            })
            updates["forecasts"] = forecasts.get("forecasts", [])

    except Exception as e:
        updates["last_error"] = f"Erro ao coletar dados: {str(e)}"

    return updates


async def call_model(state: AgentState) -> Dict[str, Any]:
    """
    Chama o LLM com as mensagens e ferramentas disponíveis.

    O modelo pode:
    1. Responder diretamente
    2. Solicitar chamada de ferramenta
    """
    messages = list(state.get("messages", []))
    config_id = state.get("config_id")

    # Construir system prompt contextualizado
    context = {
        "classifications": state.get("classifications"),
        "recommendations": state.get("recommendations"),
        "anomalies": state.get("anomalies"),
        "forecasts": state.get("forecasts"),
        "intent": state.get("current_intent"),
    }

    context_filtered = {key: value for key, value in context.items() if value}
    context_text = json.dumps(context_filtered, ensure_ascii=False, default=str) if context_filtered else None
    system_prompt = get_system_prompt(config_id, context_text)

    # Obter LLM e tools
    llm = get_llm(
        provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )

    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    # Preparar mensagens com system prompt
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    # Chamar o modelo
    response = await llm_with_tools.ainvoke(full_messages)

    # Incrementar contador de tool calls
    tool_calls_count = state.get("tool_calls_count", 0)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_calls_count += len(response.tool_calls)

    return {
        "messages": [response],
        "tool_calls_count": tool_calls_count
    }


async def call_tools(state: AgentState) -> Dict[str, Any]:
    """
    Executa as ferramentas solicitadas pelo modelo.

    Processa cada tool_call e retorna os resultados
    como ToolMessages.
    """
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}

    last_message = messages[-1]

    # Verificar se há tool calls
    tool_calls = []
    if hasattr(last_message, "tool_calls"):
        tool_calls = last_message.tool_calls
    elif isinstance(last_message, dict):
        tool_calls = last_message.get("tool_calls", [])

    if not tool_calls:
        return {"messages": []}

    # Obter ferramentas disponíveis
    tools = get_all_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    # Executar cada tool call
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call.get("name") if isinstance(tool_call, dict) else tool_call.name
        tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else tool_call.args
        tool_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id

        if tool_name not in tools_by_name:
            result = f"Ferramenta '{tool_name}' não encontrada."
        else:
            try:
                tool = tools_by_name[tool_name]
                result = await tool.ainvoke(tool_args)
            except Exception as e:
                result = f"Erro ao executar ferramenta: {str(e)}"

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_id,
                name=tool_name
            )
        )

    return {"messages": tool_messages}


async def generate_response(state: AgentState) -> Dict[str, Any]:
    """
    Gera a resposta final para o usuário.

    Usado quando não há mais tool calls pendentes
    e o modelo está pronto para responder.
    """
    # A resposta já está nas mensagens após call_model
    # Este nó pode ser usado para pós-processamento

    messages = state.get("messages", [])
    if not messages:
        return {}

    last_message = messages[-1]

    # Verificar se a última mensagem é do assistente
    is_assistant = False
    if hasattr(last_message, "type"):
        is_assistant = last_message.type == "ai"
    elif isinstance(last_message, dict):
        is_assistant = last_message.get("role") == "assistant"

    if is_assistant:
        # Pode adicionar formatação adicional aqui se necessário
        pass

    return {}


async def handle_error(state: AgentState) -> Dict[str, Any]:
    """
    Trata erros ocorridos durante a execução.

    Gera uma mensagem amigável para o usuário
    quando algo dá errado.
    """
    error = state.get("last_error", "Erro desconhecido")

    error_message = AIMessage(
        content=f"Desculpe, ocorreu um erro durante o processamento: {error}. "
                f"Por favor, tente novamente ou reformule sua pergunta."
    )

    return {
        "messages": [error_message],
        "last_error": None  # Limpa o erro após tratamento
    }
