"""Base class para subagentes especialistas.

Este modulo define a classe abstrata BaseSubagent que serve como base
para todos os subagentes especialistas do sistema multi-agente.
Cada subagente herda desta classe e implementa seus metodos abstratos
para definir comportamento especializado.
"""
from abc import ABC, abstractmethod
from typing import Any, Sequence, List, Optional
from datetime import datetime, timezone
import asyncio
import json
import time
from shared.core.tracing.decorators import log_span
from shared.core.tracing.events import log_tool_call, log_tool_call_error

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from projects.agent.utils.messages import truncate_messages

from .state import (
    SubagentState,
    AgentResult,
    SubagentTask,
    create_initial_subagent_state
)


class BaseSubagent(ABC):
    """Classe base abstrata para subagentes especialistas.

    Esta classe define a estrutura comum para todos os subagentes do sistema.
    Cada subagente especializado deve herdar desta classe e implementar:
    - AGENT_NAME: Nome unico do agente
    - AGENT_DESCRIPTION: Descricao do proposito do agente
    - get_tools(): Lista de ferramentas disponiveis
    - get_system_prompt(): Prompt de sistema especifico

    Attributes:
        AGENT_NAME: Nome identificador do subagente
        AGENT_DESCRIPTION: Descricao das capacidades do agente
    """

    AGENT_NAME: str
    AGENT_DESCRIPTION: str

    def __init__(self):
        """Inicializa o subagente.

        Raises:
            TypeError: Se AGENT_NAME nao estiver definido
        """
        if not hasattr(self, 'AGENT_NAME') or not self.AGENT_NAME:
            raise TypeError(
                f"{self.__class__.__name__} deve definir AGENT_NAME"
            )
        self._graph = None
        self._compiled_graph = None

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Retorna lista de ferramentas disponiveis para este subagente.

        Returns:
            Lista de ferramentas LangChain que o subagente pode usar
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Retorna o prompt de sistema especifico para este subagente.

        Returns:
            String com o prompt de sistema
        """
        pass

    def get_timeout(self) -> int:
        """Retorna o timeout em segundos para este subagente.

        Busca o timeout especifico nas configuracoes do agente.
        Se nao encontrar, retorna o timeout padrao de 30 segundos.

        Returns:
            Timeout em segundos
        """
        try:
            from projects.agent.config import get_agent_settings
            settings = get_agent_settings()
            timeout_attr = f"timeout_{self.AGENT_NAME}"
            return getattr(settings, timeout_attr, 30)
        except Exception:
            return 30

    def build_graph(self) -> StateGraph:
        """Constroi o grafo LangGraph para este subagente.

        O grafo padrao tem a seguinte estrutura:
        - receive_task: Prepara mensagens iniciais com tarefa
        - call_model: Chama o LLM para processar
        - call_tools: Executa ferramentas se necessario
        - respond: Gera resultado final

        Returns:
            Grafo compilado pronto para execucao
        """
        if self._compiled_graph is not None:
            return self._compiled_graph

        tools = self.get_tools()
        # Wrappear tools com logging
        wrapped_tools = [self._wrap_tool_with_logging(t) for t in tools] if tools else []
        tool_node = ToolNode(wrapped_tools) if wrapped_tools else None

        # Criar grafo de estado
        graph = StateGraph(SubagentState)

        # Adicionar nos
        graph.add_node("receive_task", self._receive_task_node)
        graph.add_node("call_model", self._call_model_node)
        if tool_node:
            graph.add_node("call_tools", tool_node)
        graph.add_node("respond", self._respond_node)

        # Definir fluxo
        graph.add_edge(START, "receive_task")
        graph.add_edge("receive_task", "call_model")

        if tool_node:
            # Se temos tools, adicionar logica condicional
            graph.add_conditional_edges(
                "call_model",
                self._should_call_tools,
                {
                    "call_tools": "call_tools",
                    "respond": "respond"
                }
            )
            graph.add_edge("call_tools", "call_model")
        else:
            # Sem tools, vai direto para respond
            graph.add_edge("call_model", "respond")

        graph.add_edge("respond", END)

        self._compiled_graph = graph.compile()
        return self._compiled_graph

    def _receive_task_node(self, state: SubagentState) -> dict:
        """No que prepara as mensagens iniciais com a tarefa.

        Args:
            state: Estado atual do subagente

        Returns:
            Atualizacoes do estado com mensagens iniciais
        """
        system_prompt = self.get_system_prompt()
        task_message = self._format_task_message(state)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=task_message)
        ]

        return {
            "messages": messages,
            "started_at": datetime.now(timezone.utc)
        }

    async def _call_model_node(self, state: SubagentState) -> dict:
        """No que chama o modelo LLM (async).

        Args:
            state: Estado atual do subagente

        Returns:
            Atualizacoes do estado com resposta do modelo
        """
        from projects.agent.llm.provider import get_llm_with_tools
        from shared.core.logging import get_logger

        logger = get_logger(__name__)

        # Obter tools e criar LLM com tools vinculadas
        tools = self.get_tools()

        try:
            if tools:
                # LLM com tools para poder fazer tool calls
                llm = get_llm_with_tools(tools)
            else:
                # LLM sem tools para subagentes que não usam
                from projects.agent.llm.provider import get_llm
                llm = get_llm()

            # Invocar LLM com mensagens do estado (async)
            messages = state.get("messages", [])
            response = await llm.ainvoke(messages)

            logger.debug(
                "LLM invocado",
                agent=self.AGENT_NAME,
                has_tool_calls=hasattr(response, 'tool_calls') and bool(response.tool_calls),
                tool_calls_count=len(response.tool_calls) if hasattr(response, 'tool_calls') and response.tool_calls else 0
            )

            current_tool_calls = int(state.get("tool_calls_count", 0) or 0)
            new_tool_calls = 0
            if hasattr(response, "tool_calls") and response.tool_calls:
                new_tool_calls = len(response.tool_calls)

            return {
                "messages": [response],
                "tool_calls_count": current_tool_calls + new_tool_calls,
            }

        except Exception as e:
            logger.error(
                "Erro ao invocar LLM",
                agent=self.AGENT_NAME,
                error=str(e)
            )
            # Em caso de erro, retornar mensagem de erro
            response = AIMessage(
                content=f"Erro ao processar requisição: {str(e)}"
            )
            return {
                "messages": [response],
                "error": str(e),
                "tool_calls_count": int(state.get("tool_calls_count", 0) or 0),
            }

    def _respond_node(self, state: SubagentState) -> dict:
        """No que gera o resultado final.

        Args:
            state: Estado atual do subagente

        Returns:
            Atualizacoes do estado com resultado final
        """
        # Extrair resultado das mensagens
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None

        result = None
        if last_message and isinstance(last_message, AIMessage):
            result = {
                "response": last_message.content,
                "tool_calls": self._extract_tool_calls(messages)
            }

        return {
            "result": result,
            "completed_at": datetime.now(timezone.utc)
        }

    def _should_call_tools(self, state: SubagentState) -> str:
        """Decide se deve chamar ferramentas ou responder.

        Args:
            state: Estado atual do subagente

        Returns:
            "call_tools" se houver tool_calls, "respond" caso contrario
        """
        from projects.agent.config import get_agent_settings

        max_tool_calls = get_agent_settings().max_tool_calls
        tool_calls_count = int(state.get("tool_calls_count", 0) or 0)
        if tool_calls_count >= max_tool_calls:
            return "respond"

        messages = state.get("messages", [])
        if not messages:
            return "respond"

        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls'):
            if last_message.tool_calls:
                return "call_tools"

        return "respond"

    def _format_task_message(self, state: SubagentState) -> str:
        """Formata a mensagem da tarefa para o LLM.

        Args:
            state: Estado atual com a tarefa

        Returns:
            String formatada com descricao e contexto da tarefa
        """
        task = state.get("task", {})
        description = task.get("description", "")
        context = task.get("context", {})
        user_question = task.get("user_question", "")

        context_str = json.dumps(context, indent=2, ensure_ascii=False) if context else "{}"

        parts = []
        if user_question:
            parts.append(f"## Pergunta do usuario\n{user_question}")
        parts.append(f"## Tarefa\n{description}")
        parts.append(f"## Dados disponiveis\n```json\n{context_str}\n```")
        parts.append("Responda focado na pergunta do usuario. Use as tools disponiveis se precisar de mais dados.")

        return "\n\n".join(parts)

    def _extract_tool_calls(self, messages: Sequence[BaseMessage]) -> list[str]:
        """Extrai nomes das ferramentas chamadas das mensagens.

        Args:
            messages: Sequencia de mensagens para analisar

        Returns:
            Lista com nomes das ferramentas chamadas
        """
        tool_names = []

        for message in messages:
            if isinstance(message, AIMessage) and hasattr(message, 'tool_calls'):
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if isinstance(tool_call, dict):
                            name = tool_call.get("name")
                        else:
                            name = getattr(tool_call, "name", None)
                        if name:
                            tool_names.append(name)

        return tool_names

    def _wrap_tool_with_logging(self, tool: BaseTool) -> BaseTool:
        """Wrappeia uma tool para adicionar logging automático.

        IMPORTANTE: Cria uma CÓPIA da tool original para evitar mutação.

        Args:
            tool: Tool original do LangChain

        Returns:
            Nova instância de tool com logging wrapper (não modifica a original)
        """
        import copy

        # Criar cópia profunda da tool para não modificar a original
        try:
            wrapped_tool = copy.deepcopy(tool)
        except TypeError:
            wrapped_tool = copy.copy(tool)
        # LangChain tools: .coroutine para async, .func para sync
        original_coroutine = getattr(tool, 'coroutine', None)
        original_func = tool.func

        # Verificar se a tool e async (tem coroutine ou func e async)
        is_async = (
            original_coroutine is not None
            or asyncio.iscoroutinefunction(original_func)
        )
        async_callable = original_coroutine or original_func

        if is_async:
            async def logged_tool_async(*args, **kwargs):
                start = time.time()
                try:
                    result = await async_callable(*args, **kwargs)
                    duration = (time.time() - start) * 1000

                    log_tool_call(
                        tool_name=wrapped_tool.name,
                        params=kwargs,
                        result=result,
                        duration_ms=duration,
                        status="success"
                    )
                    return result
                except Exception as e:
                    duration = (time.time() - start) * 1000
                    log_tool_call_error(
                        tool_name=wrapped_tool.name,
                        params=kwargs,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        duration_ms=duration
                    )
                    raise

            wrapped_tool.coroutine = logged_tool_async
        else:
            def logged_tool_sync(*args, **kwargs):
                start = time.time()
                try:
                    result = original_func(*args, **kwargs)
                    duration = (time.time() - start) * 1000

                    log_tool_call(
                        tool_name=wrapped_tool.name,
                        params=kwargs,
                        result=result,
                        duration_ms=duration,
                        status="success"
                    )
                    return result
                except Exception as e:
                    duration = (time.time() - start) * 1000
                    log_tool_call_error(
                        tool_name=wrapped_tool.name,
                        params=kwargs,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        duration_ms=duration
                    )
                    raise

            wrapped_tool.func = logged_tool_sync

        return wrapped_tool

    @log_span("subagent_execution", log_args=True, log_result=True)
    async def run(
        self,
        task: SubagentTask,
        config_id: int,
        user_id: int,
        thread_id: str,
        messages: Optional[Sequence[BaseMessage]] = None
    ) -> AgentResult:
        """Executa o subagente com a tarefa especificada.

        Args:
            task: Tarefa a ser executada
            config_id: ID da configuracao Facebook Ads
            user_id: ID do usuario
            thread_id: ID da thread para persistencia
            messages: Mensagens iniciais opcionais

        Returns:
            AgentResult com resultado da execucao
        """
        start_time = datetime.now(timezone.utc)

        # Truncar mensagens para respeitar limite de contexto
        if messages is not None:
            from projects.agent.config import get_agent_settings
            messages = truncate_messages(
                messages,
                max_messages=get_agent_settings().max_conversation_messages,
            )

        # Criar estado inicial
        initial_state = create_initial_subagent_state(
            task=task,
            config_id=config_id,
            user_id=user_id,
            thread_id=thread_id,
            messages=messages
        )

        try:
            # Executar grafo com timeout
            graph = self.build_graph()
            timeout = self.get_timeout()

            final_state = await asyncio.wait_for(
                graph.ainvoke(initial_state),
                timeout=timeout
            )

            # Calcular duracao
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            # Extrair resultado
            result_data = final_state.get("result")
            error = final_state.get("error")
            tool_calls = self._extract_tool_calls(final_state.get("messages", []))

            return AgentResult(
                agent_name=self.AGENT_NAME,
                success=error is None,
                data=result_data,
                error=error,
                duration_ms=duration_ms,
                tool_calls=tool_calls
            )

        except asyncio.TimeoutError:
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            return AgentResult(
                agent_name=self.AGENT_NAME,
                success=False,
                data=None,
                error=f"Timeout after {self.get_timeout()} seconds",
                duration_ms=duration_ms,
                tool_calls=[]
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            return AgentResult(
                agent_name=self.AGENT_NAME,
                success=False,
                data=None,
                error=f"Error: {str(e)}",
                duration_ms=duration_ms,
                tool_calls=[]
            )
