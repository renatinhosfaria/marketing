"""
Serviço principal do agente de tráfego pago.
"""

import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional, Set

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


class TrafficAgentService:
    """
    Serviço do agente de tráfego pago.

    Gerencia conversas, execução do grafo e persistência de estado.
    """

    def __init__(self):
        self._agent = None
        self._checkpointer = None

    async def initialize(self):
        """
        Inicializa o agente e checkpointer.
        """
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


# Instância singleton do serviço
_agent_service: Optional[TrafficAgentService] = None


async def get_agent_service() -> TrafficAgentService:
    """
    Obtém a instância do serviço do agente.

    Returns:
        TrafficAgentService inicializado
    """
    global _agent_service

    if _agent_service is None:
        _agent_service = TrafficAgentService()
        await _agent_service.initialize()

    return _agent_service
