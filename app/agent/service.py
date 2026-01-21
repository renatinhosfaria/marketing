"""
Serviço principal do agente de tráfego pago.

Suporta dois modos de operação:
- Single-agent (padrão): Agente monolítico original
- Multi-agent: Sistema orquestrado com subagentes especializados
"""

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Set
import json

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.graph.state import AgentState, create_initial_state
from app.agent.graph.builder import build_agent_graph
from app.agent.memory.checkpointer import AgentCheckpointer, get_agent_checkpointer
from app.agent.config import get_agent_settings
from app.core.logging import get_logger


logger = get_logger(__name__)
settings = get_agent_settings()

# Nós do grafo que devem emitir eventos de debug
GRAPH_NODE_NAMES: Set[str] = {
    "classify_intent",
    "gather_data",
    "call_model",
    "call_tools",
    "generate_response",
    "handle_error",
}


def should_use_multiagent() -> bool:
    """Verifica se deve usar sistema multi-agente.

    Returns:
        True se multi-agent esta habilitado
    """
    settings = get_agent_settings()
    return settings.multi_agent_enabled


def get_agent():
    """Retorna o agente apropriado baseado na configuracao.

    Returns:
        Orchestrator se multi-agent habilitado, senao agente legado
    """
    if should_use_multiagent():
        from app.agent.orchestrator import get_orchestrator
        return get_orchestrator()
    # Retorna agente legado existente
    return build_agent_graph()


def format_sse_event(event_type: str, data: dict) -> str:
    """Formata evento SSE.

    Args:
        event_type: Tipo do evento
        data: Dados do evento

    Returns:
        String formatada para SSE
    """
    payload = {
        "type": event_type,
        **data
    }
    return f"data: {json.dumps(payload)}\\n\\n"


class TrafficAgentService:
    """
    Serviço do agente de tráfego pago.

    Gerencia conversas, execução do grafo e persistência de estado.

    Suporta dois modos de operação:
    - Single-agent (padrão): Usa o grafo monolítico original
    - Multi-agent: Usa o OrchestratorAgent com subagentes especializados

    Attributes:
        _agent: Grafo compilado do agente single-agent
        _checkpointer: Checkpointer para persistência de estado
        _multi_agent_mode: Se True, usa o sistema multi-agente
        _orchestrator: Instância do OrchestratorAgent (somente em modo multi-agent)
    """

    def __init__(self, multi_agent_mode: bool = False):
        """
        Inicializa o serviço do agente.

        Args:
            multi_agent_mode: Se True, usa o sistema multi-agente ao invés do single-agent
        """
        self._agent = None
        self._checkpointer = None
        self._multi_agent_mode = multi_agent_mode
        self._orchestrator = None  # Para modo multi-agente

    @property
    def multi_agent_mode(self) -> bool:
        """Retorna se o serviço está em modo multi-agente."""
        return self._multi_agent_mode

    @property
    def orchestrator(self):
        """Retorna o orchestrator (somente em modo multi-agente)."""
        return self._orchestrator

    async def initialize(self):
        """
        Inicializa o agente e checkpointer.

        Em modo single-agent: Compila o grafo monolítico.
        Em modo multi-agent: Inicializa o OrchestratorAgent.
        """
        if self._multi_agent_mode:
            # Modo multi-agente: inicializar orchestrator
            if self._orchestrator is None:
                try:
                    from app.agent.orchestrator import OrchestratorAgent
                    self._orchestrator = OrchestratorAgent()
                    self._orchestrator.build_graph()
                    logger.info("TrafficAgentService inicializado em modo multi-agente")
                except ImportError as e:
                    logger.error(f"Falha ao importar OrchestratorAgent: {e}")
                    raise RuntimeError(
                        "Falha ao inicializar modo multi-agente. "
                        "Verifique se o módulo orchestrator está disponível."
                    ) from e
                except Exception as e:
                    logger.error(f"Erro ao inicializar orchestrator: {e}")
                    raise
        else:
            # Modo single-agent: inicializar grafo tradicional
            if self._agent is None:
                graph = build_agent_graph()
                self._checkpointer = await AgentCheckpointer.get_checkpointer()
                self._agent = graph.compile(checkpointer=self._checkpointer)
                logger.info("TrafficAgentService inicializado")

    async def chat(
        self,
        message: str,
        config_id: int,
        user_id: int,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processa uma mensagem do usuário e retorna a resposta.

        Roteia automaticamente para o modo multi-agente se configurado.

        Args:
            message: Mensagem do usuário
            config_id: ID da configuração Facebook Ads
            user_id: ID do usuário
            thread_id: ID da conversa (opcional, cria novo se None)

        Returns:
            Dicionário com resposta e metadados
        """
        await self.initialize()

        # Gerar thread_id se não fornecido
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Rotear para multi-agente se habilitado
        if self._multi_agent_mode:
            return await self.chat_multi_agent(message, config_id, user_id, thread_id)
        if should_use_multiagent():
            return await self._chat_multiagent(
                message=message,
                config_id=config_id,
                user_id=user_id,
                thread_id=thread_id,
                db=None
            )

        # Criar mensagem do usuário
        user_message = HumanMessage(content=message)

        # Configurar estado inicial
        initial_state = create_initial_state(
            config_id=config_id,
            user_id=user_id,
            thread_id=thread_id,
            initial_message={"role": "user", "content": message}
        )

        # Configurar thread para o checkpointer
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        try:
            # Executar o agente
            result = await self._agent.ainvoke(
                {"messages": [user_message], **initial_state},
                config=config
            )

            # Extrair resposta
            messages = result.get("messages", [])
            response_content = ""

            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    response_content = msg.content
                    break
                elif isinstance(msg, dict) and msg.get("role") == "assistant":
                    response_content = msg.get("content", "")
                    break

            return {
                "success": True,
                "thread_id": thread_id,
                "response": response_content,
                "intent": result.get("current_intent"),
                "tool_calls_count": result.get("tool_calls_count", 0),
            }

        except Exception as e:
            logger.error(f"Erro no chat: {e}")
            return {
                "success": False,
                "thread_id": thread_id,
                "error": str(e),
                "response": "Desculpe, ocorreu um erro ao processar sua mensagem."
            }

    async def _chat_multiagent(
        self,
        message: str,
        config_id: int,
        user_id: int,
        thread_id: str,
        db: Any
    ) -> dict:
        """Executa chat usando sistema multi-agente.

        Args:
            message: Mensagem do usuário
            config_id: ID da configuração
            user_id: ID do usuário
            thread_id: ID da thread
            db: Sessão do banco

        Returns:
            Dicionário com resposta e metadados
        """
        from app.agent.orchestrator import (
            get_orchestrator,
            create_initial_orchestrator_state
        )
        from langchain_core.messages import HumanMessage

        logger.info(f"Chat multi-agente: thread={thread_id}")

        # Criar estado inicial
        state = create_initial_orchestrator_state(
            config_id=config_id,
            user_id=user_id,
            thread_id=thread_id,
            messages=[HumanMessage(content=message)]
        )

        # Obter orchestrator
        orchestrator = get_orchestrator()

        # Executar
        result = await orchestrator.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id}}
        )

        # Extrair resposta
        response = result.get("synthesized_response", "")
        intent = result.get("user_intent", "general")
        confidence = result.get("confidence_score", 0.0)

        return {
            "response": response,
            "intent": intent,
            "confidence": confidence,
            "thread_id": thread_id,
            "agent_results": result.get("agent_results", {})
        }

    async def chat_multi_agent(
        self,
        message: str,
        config_id: int,
        user_id: int,
        thread_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processa uma mensagem usando o sistema multi-agente.

        Este método utiliza o OrchestratorAgent para coordenar
        múltiplos subagentes especializados e sintetizar uma resposta.

        Args:
            message: Mensagem do usuário
            config_id: ID da configuração Facebook Ads
            user_id: ID do usuário autenticado
            thread_id: ID da conversa (opcional)

        Returns:
            Dicionário com:
            - success: bool indicando se a operação foi bem-sucedida
            - thread_id: str identificador da conversa
            - response: str resposta sintetizada
            - confidence_score: float score de confiança (0-1)
            - intent: str intenção do usuário detectada
            - agents_used: list[str] lista de subagentes utilizados
            - agent_results: dict resultados de cada subagente
            - error: str mensagem de erro (se success=False)
        """
        # Garantir inicialização
        await self.initialize()

        # Verificar se estamos em modo multi-agente
        if not self._multi_agent_mode or self._orchestrator is None:
            logger.error("chat_multi_agent chamado sem modo multi-agente habilitado")
            return {
                "success": False,
                "thread_id": thread_id or str(uuid.uuid4()),
                "error": "Modo multi-agente não está habilitado",
                "response": "O sistema multi-agente não está configurado.",
            }

        # Gerar thread_id se não fornecido
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        try:
            # Executar o orchestrator
            logger.info(f"Processando mensagem multi-agente: thread={thread_id}")

            result = await self._orchestrator.run(
                message=message,
                config_id=config_id,
                user_id=user_id,
                thread_id=thread_id
            )

            # Verificar se houve erro no orchestrator
            if result.get("error"):
                logger.warning(f"Orchestrator retornou erro: {result.get('error')}")
                return {
                    "success": False,
                    "thread_id": thread_id,
                    "error": result.get("error"),
                    "response": "Desculpe, ocorreu um erro ao processar sua mensagem.",
                    "confidence_score": 0.0,
                    "intent": result.get("user_intent"),
                    "agents_used": result.get("required_agents", []),
                    "agent_results": result.get("agent_results", {}),
                }

            # Extrair informações do resultado
            synthesized_response = result.get("synthesized_response", "")
            confidence_score = result.get("confidence_score", 0.0)
            user_intent = result.get("user_intent")
            required_agents = result.get("required_agents", [])
            agent_results = result.get("agent_results", {})

            # Se não houver resposta sintetizada, gerar uma básica
            if not synthesized_response:
                synthesized_response = "Não foi possível gerar uma resposta. Por favor, tente novamente."
                logger.warning(f"Resposta vazia do orchestrator: thread={thread_id}")

            # Determinar quais agentes realmente executaram com sucesso
            agents_used = [
                agent_name
                for agent_name, agent_result in agent_results.items()
                if isinstance(agent_result, dict) and agent_result.get("success", False)
            ]

            return {
                "success": True,
                "thread_id": thread_id,
                "response": synthesized_response,
                "confidence_score": confidence_score,
                "intent": user_intent,
                "agents_used": agents_used,
                "agent_results": agent_results,
            }

        except Exception as e:
            logger.error(f"Erro no chat_multi_agent: {e}", exc_info=True)
            return {
                "success": False,
                "thread_id": thread_id,
                "error": str(e),
                "response": "Desculpe, ocorreu um erro ao processar sua mensagem.",
                "confidence_score": 0.0,
                "intent": None,
                "agents_used": [],
                "agent_results": {},
            }

    async def stream_chat_multi_agent(
        self,
        message: str,
        config_id: int,
        user_id: int,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processa uma mensagem com streaming usando sistema multi-agente.

        Emite eventos de streaming para acompanhamento em tempo real do
        processo de orquestração multi-agente.

        Eventos emitidos:
        - orchestrator_start: Início do processamento
        - intent_detected: Quando a intenção é identificada
        - plan_created: Quando o plano de execução é criado
        - agent_start: Quando um subagente começa
        - agent_end: Quando um subagente termina
        - synthesis_start: Início da síntese
        - text: Chunks da resposta sintetizada
        - done: Finalização com metadados
        - error: Em caso de erro

        Args:
            message: Mensagem do usuário
            config_id: ID da configuração Facebook Ads
            user_id: ID do usuário
            thread_id: ID da conversa (opcional, cria novo se None)

        Yields:
            Eventos de streaming conforme o processamento avança
        """
        # Garantir inicialização
        await self.initialize()

        # Verificar se estamos em modo multi-agente
        if not self._multi_agent_mode or self._orchestrator is None:
            logger.error("stream_chat_multi_agent chamado sem modo multi-agente habilitado")
            yield {
                "type": "error",
                "error": "Modo multi-agente não está habilitado",
                "thread_id": thread_id or "",
                "timestamp": time.time() * 1000,
            }
            return

        # Gerar thread_id se não fornecido
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        # Timestamp de início do processamento
        start_time = time.time() * 1000

        try:
            # Evento: orchestrator_start
            yield {
                "type": "orchestrator_start",
                "thread_id": thread_id,
                "timestamp": start_time,
            }

            # Importar dependências do orchestrator
            from langchain_core.messages import HumanMessage
            from app.agent.orchestrator.state import (
                create_initial_orchestrator_state,
                get_agents_for_intent,
            )

            # Criar estado inicial
            initial_state = create_initial_orchestrator_state(
                config_id=config_id,
                user_id=user_id,
                thread_id=thread_id,
                messages=[HumanMessage(content=message)] if message else []
            )

            # Obter grafo
            graph = self._orchestrator.build_graph()

            # Rastrear timestamps por agente
            agent_start_times: Dict[str, float] = {}
            agents_completed: List[str] = []
            detected_intent: Optional[str] = None
            plan_agents: List[str] = []
            plan_parallel: bool = True
            confidence_score: float = 0.0

            # Usar astream com stream_mode="updates" para capturar cada nó
            async for updates in graph.astream(
                initial_state,
                stream_mode="updates",
            ):
                current_time = time.time() * 1000

                if not isinstance(updates, dict):
                    continue

                # Processar cada nó que foi atualizado
                for node_name, node_output in updates.items():
                    if not isinstance(node_output, dict):
                        continue

                    # Nó: parse_request - Detecção de intenção
                    if node_name == "parse_request":
                        user_intent = node_output.get("user_intent")
                        if user_intent:
                            detected_intent = user_intent
                            yield {
                                "type": "intent_detected",
                                "intent": user_intent,
                                "thread_id": thread_id,
                                "timestamp": current_time,
                            }

                    # Nó: plan_execution - Criação do plano
                    elif node_name == "plan_execution":
                        required_agents = node_output.get("required_agents", [])
                        execution_plan = node_output.get("execution_plan")

                        if required_agents:
                            plan_agents = required_agents
                            plan_parallel = True
                            if execution_plan and isinstance(execution_plan, dict):
                                plan_parallel = execution_plan.get("parallel", True)

                            yield {
                                "type": "plan_created",
                                "agents": plan_agents,
                                "parallel": plan_parallel,
                                "thread_id": thread_id,
                                "timestamp": current_time,
                            }

                            # Emitir agent_start para cada agente planejado
                            for agent_name in plan_agents:
                                agent_start_times[agent_name] = current_time
                                agent_description = self._get_agent_description(agent_name)
                                yield {
                                    "type": "agent_start",
                                    "agent": agent_name,
                                    "description": agent_description,
                                    "thread_id": thread_id,
                                    "timestamp": current_time,
                                }

                    # Nó: dispatch_agents ou collect_results - Resultados dos subagentes
                    elif node_name in ("dispatch_agents", "collect_results"):
                        agent_results = node_output.get("agent_results", {})
                        for agent_name, agent_result in agent_results.items():
                            if agent_name not in agents_completed:
                                agents_completed.append(agent_name)

                                # Calcular duração
                                start_ts = agent_start_times.get(agent_name, current_time)
                                duration_ms = current_time - start_ts

                                # Determinar sucesso
                                success = True
                                if isinstance(agent_result, dict):
                                    success = agent_result.get("success", True)

                                yield {
                                    "type": "agent_end",
                                    "agent": agent_name,
                                    "success": success,
                                    "duration_ms": duration_ms,
                                    "thread_id": thread_id,
                                    "timestamp": current_time,
                                }

                    # Nó: synthesize - Síntese da resposta
                    elif node_name == "synthesize":
                        # Emitir synthesis_start
                        yield {
                            "type": "synthesis_start",
                            "agents_completed": len(agents_completed),
                            "thread_id": thread_id,
                            "timestamp": current_time,
                        }

                        # Obter resposta sintetizada
                        synthesized_response = node_output.get("synthesized_response", "")
                        confidence_score = node_output.get("confidence_score", 0.0)

                        # Fazer streaming da resposta em chunks para efeito suave
                        if synthesized_response:
                            chunks = self._chunk_text(synthesized_response)
                            for chunk in chunks:
                                yield {
                                    "type": "text",
                                    "content": chunk,
                                    "thread_id": thread_id,
                                    "timestamp": time.time() * 1000,
                                }

            # Calcular duração total
            end_time = time.time() * 1000
            total_duration = end_time - start_time

            # Evento: done
            yield {
                "type": "done",
                "thread_id": thread_id,
                "confidence_score": confidence_score,
                "total_duration_ms": total_duration,
                "agents_used": agents_completed,
                "timestamp": end_time,
            }

        except Exception as e:
            logger.error(f"Erro no stream_chat_multi_agent: {e}", exc_info=True)
            yield {
                "type": "error",
                "error": str(e),
                "thread_id": thread_id,
                "timestamp": time.time() * 1000,
            }

    def _get_agent_description(self, agent_name: str) -> str:
        """Retorna descrição legível para um subagente.

        Args:
            agent_name: Nome do subagente

        Returns:
            Descrição do agente
        """
        descriptions = {
            "classification": "Analisando performance de campanhas",
            "anomaly": "Detectando anomalias e problemas",
            "forecast": "Gerando previsões de CPL e leads",
            "recommendation": "Elaborando recomendações de ações",
            "campaign": "Coletando dados de campanhas",
            "analysis": "Executando análises avançadas",
        }
        return descriptions.get(agent_name, f"Executando {agent_name}")

    def _chunk_text(self, text: str, chunk_size: int = 50) -> List[str]:
        """Divide texto em chunks para streaming suave.

        Args:
            text: Texto para dividir
            chunk_size: Tamanho aproximado de cada chunk em caracteres

        Returns:
            Lista de chunks de texto
        """
        if not text:
            return []

        chunks = []
        words = text.split()
        current_chunk = ""

        for word in words:
            if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(current_chunk + " ")
                current_chunk = word
            else:
                if current_chunk:
                    current_chunk += " " + word
                else:
                    current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def stream_chat(
        self,
        message: str,
        config_id: int,
        user_id: int,
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Processa uma mensagem com streaming de resposta.

        Usa astream com stream_mode="updates" para capturar transições de nós
        do grafo LangGraph, permitindo debug em tempo real.

        Emite eventos de debug para acompanhamento em tempo real:
        - node_start/node_end: Início e fim de cada nó do grafo
        - intent_classified: Quando a intenção é detectada
        - data_gathered: Quando dados são coletados
        - tool_start/tool_end: Execução de ferramentas
        - text: Chunks de texto do modelo
        - done/error: Finalização

        Args:
            message: Mensagem do usuário
            config_id: ID da configuração Facebook Ads
            user_id: ID do usuário
            thread_id: ID da conversa

        Yields:
            Chunks da resposta conforme são gerados
        """
        await self.initialize()

        if thread_id is None:
            thread_id = str(uuid.uuid4())

        user_message = HumanMessage(content=message)

        initial_state = create_initial_state(
            config_id=config_id,
            user_id=user_id,
            thread_id=thread_id,
            initial_message={"role": "user", "content": message}
        )

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        # Rastrear timestamps de início dos nós para calcular duração
        node_start_times: Dict[str, float] = {}
        # Rastrear nós que já finalizaram
        completed_nodes: Set[str] = set()
        # Rastrear timestamps de início das ferramentas
        tool_start_times: Dict[str, float] = {}
        # Timestamp de início do processamento
        start_time = time.time() * 1000
        # Acumular conteúdo de texto
        accumulated_text = ""

        try:
            # Emitir evento de início do processamento
            yield {
                "type": "stream_start",
                "thread_id": thread_id,
                "timestamp": start_time,
            }

            # Usar astream com stream_mode="updates" para capturar cada nó
            # Isso retorna {node_name: output_dict} para cada nó executado
            async for updates in self._agent.astream(
                {"messages": [user_message], **initial_state},
                config=config,
                stream_mode="updates",
            ):
                current_time = time.time() * 1000

                if not isinstance(updates, dict):
                    continue

                # Processar cada nó que foi atualizado
                for node_name, node_output in updates.items():
                    if node_name not in GRAPH_NODE_NAMES:
                        continue

                    # Emitir node_start se ainda não começou
                    if node_name not in node_start_times:
                        node_start_times[node_name] = current_time
                        yield {
                            "type": "node_start",
                            "node": node_name,
                            "timestamp": current_time,
                            "thread_id": thread_id,
                        }

                    # Extrair dados específicos de cada nó
                    if isinstance(node_output, dict):
                        # Evento de intenção classificada
                        if node_name == "classify_intent":
                            intent = node_output.get("current_intent")
                            if intent:
                                yield {
                                    "type": "intent_classified",
                                    "intent": intent,
                                    "timestamp": current_time,
                                    "thread_id": thread_id,
                                }

                        # Evento de dados coletados
                        elif node_name == "gather_data":
                            classifications = node_output.get("classifications", [])
                            recommendations = node_output.get("recommendations", [])
                            anomalies = node_output.get("anomalies", [])
                            forecasts = node_output.get("forecasts", [])

                            yield {
                                "type": "data_gathered",
                                "data_counts": {
                                    "classifications": len(classifications) if isinstance(classifications, list) else 0,
                                    "recommendations": len(recommendations) if isinstance(recommendations, list) else 0,
                                    "anomalies": len(anomalies) if isinstance(anomalies, list) else 0,
                                    "forecasts": len(forecasts) if isinstance(forecasts, list) else 0,
                                },
                                "timestamp": current_time,
                                "thread_id": thread_id,
                            }

                        # Extrair mensagens do modelo para streaming de texto
                        elif node_name in ["call_model", "generate_response"]:
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                # Verificar se é AIMessage com conteúdo
                                if isinstance(msg, AIMessage) and msg.content:
                                    # Enviar o conteúdo completo como texto
                                    # (para streaming real, usaríamos astream_events)
                                    new_text = str(msg.content)
                                    if new_text and new_text != accumulated_text:
                                        # Enviar apenas a parte nova
                                        if accumulated_text and new_text.startswith(accumulated_text):
                                            delta = new_text[len(accumulated_text):]
                                        else:
                                            delta = new_text

                                        if delta:
                                            yield {
                                                "type": "text",
                                                "content": delta,
                                                "timestamp": current_time,
                                                "thread_id": thread_id,
                                            }
                                        accumulated_text = new_text

                                # Detectar tool calls
                                if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
                                    for tool_call in msg.tool_calls:
                                        tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
                                        if tool_name not in tool_start_times:
                                            tool_start_times[tool_name] = current_time
                                            tool_input = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
                                            input_preview = str(tool_input)[:100] if tool_input else None

                                            yield {
                                                "type": "tool_start",
                                                "tool": tool_name,
                                                "input_preview": input_preview,
                                                "timestamp": current_time,
                                                "thread_id": thread_id,
                                            }

                        # Resultados de ferramentas no call_tools
                        elif node_name == "call_tools":
                            messages = node_output.get("messages", [])
                            for msg in messages:
                                # ToolMessage contém resultado
                                if hasattr(msg, "type") and getattr(msg, "type", None) == "tool":
                                    tool_name = getattr(msg, "name", "unknown")
                                    start_ts = tool_start_times.pop(tool_name, current_time)
                                    duration_ms = current_time - start_ts

                                    content = getattr(msg, "content", "")
                                    output_preview = str(content)[:100] if content else None
                                    success = "error" not in str(content).lower() if content else True

                                    yield {
                                        "type": "tool_end",
                                        "tool": tool_name,
                                        "success": success,
                                        "output_preview": output_preview,
                                        "duration_ms": duration_ms,
                                        "timestamp": current_time,
                                        "thread_id": thread_id,
                                    }

                    # Emitir node_end
                    if node_name not in completed_nodes:
                        completed_nodes.add(node_name)
                        duration_ms = current_time - node_start_times.get(node_name, current_time)
                        yield {
                            "type": "node_end",
                            "node": node_name,
                            "timestamp": current_time,
                            "duration_ms": duration_ms,
                            "thread_id": thread_id,
                        }

            # Calcular duração total
            end_time = time.time() * 1000
            total_duration = end_time - start_time

            yield {
                "type": "done",
                "thread_id": thread_id,
                "timestamp": end_time,
                "total_duration_ms": total_duration,
            }

        except Exception as e:
            logger.error(f"Erro no stream: {e}")
            yield {
                "type": "error",
                "error": str(e),
                "thread_id": thread_id,
                "timestamp": time.time() * 1000,
            }

    async def get_conversation_history(
        self,
        thread_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Obtém o histórico de uma conversa.

        Args:
            thread_id: ID da conversa

        Returns:
            Lista de mensagens da conversa
        """
        await self.initialize()

        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = await self._agent.aget_state(config)

            if state and state.values:
                messages = state.values.get("messages", [])
                return [
                    {
                        "role": getattr(msg, "type", "unknown"),
                        "content": getattr(msg, "content", str(msg)),
                    }
                    for msg in messages
                ]

            return []

        except Exception as e:
            logger.error(f"Erro ao obter histórico: {e}")
            return []

    async def clear_conversation(self, thread_id: str) -> bool:
        """
        Limpa o histórico de uma conversa.

        Args:
            thread_id: ID da conversa

        Returns:
            True se limpo com sucesso
        """
        try:
            # Deletar checkpoints da thread
            if self._checkpointer:
                async with self._checkpointer._pool.connection() as conn:
                    async with conn.cursor() as cur:
                        await cur.execute(
                            "DELETE FROM agent_checkpoints WHERE thread_id = %s",
                            (thread_id,)
                        )
                        await cur.execute(
                            "DELETE FROM agent_writes WHERE thread_id = %s",
                            (thread_id,)
                        )
                        await conn.commit()

            logger.info(f"Conversa {thread_id} limpa")
            return True

        except Exception as e:
            logger.error(f"Erro ao limpar conversa: {e}")
            return False


# Instância singleton do serviço (single-agent)
_agent_service: Optional[TrafficAgentService] = None

# Instância singleton do serviço (multi-agent)
_multi_agent_service: Optional[TrafficAgentService] = None


async def get_agent_service() -> TrafficAgentService:
    """
    Obtém a instância do serviço do agente (modo single-agent).

    Esta função retorna o serviço no modo single-agent tradicional.
    Para modo multi-agente, use get_multi_agent_service().

    Returns:
        TrafficAgentService inicializado em modo single-agent
    """
    global _agent_service

    if _agent_service is None:
        _agent_service = TrafficAgentService(multi_agent_mode=False)
        await _agent_service.initialize()

    return _agent_service


async def get_multi_agent_service() -> TrafficAgentService:
    """
    Obtém a instância do serviço do agente em modo multi-agente.

    Esta função retorna o serviço configurado para usar o
    OrchestratorAgent com subagentes especializados.

    Para modo single-agent tradicional, use get_agent_service().

    Returns:
        TrafficAgentService inicializado em modo multi-agente

    Raises:
        RuntimeError: Se o módulo orchestrator não estiver disponível
    """
    global _multi_agent_service

    if _multi_agent_service is None:
        _multi_agent_service = TrafficAgentService(multi_agent_mode=True)
        await _multi_agent_service.initialize()

    return _multi_agent_service


def reset_services() -> None:
    """
    Reseta as instâncias singleton dos serviços.

    Útil para testes que precisam reinicializar os serviços.
    """
    global _agent_service, _multi_agent_service
    _agent_service = None
    _multi_agent_service = None
