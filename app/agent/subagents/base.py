"""Base class para subagentes especialistas.

Este modulo define a classe abstrata BaseSubagent que serve como base
para todos os subagentes especialistas do sistema multi-agente.
Cada subagente herda desta classe e implementa seus metodos abstratos
para definir comportamento especializado.
"""
from abc import ABC, abstractmethod
from typing import Any, Sequence, List, Optional
from datetime import datetime
import asyncio
import json
import time
from app.core.tracing.decorators import log_span
from app.core.tracing.events import log_tool_call, log_tool_call_error

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

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
            from app.agent.config import get_agent_settings
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
            "started_at": datetime.utcnow()
        }

    def _call_model_node(self, state: SubagentState) -> dict:
        """No que chama o modelo LLM.

        Este no e um placeholder que sera substituido quando
        o LLM estiver configurado. Para testes, retorna uma
        resposta mock.

        Args:
            state: Estado atual do subagente

        Returns:
            Atualizacoes do estado com resposta do modelo
        """
        # Placeholder - em producao, isso chamaria o LLM real
        # Por enquanto, retorna uma resposta mock para testes
        response = AIMessage(
            content="Analysis complete. This is a placeholder response."
        )

        return {
            "messages": [response]
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
            "completed_at": datetime.utcnow()
        }

    def _should_call_tools(self, state: SubagentState) -> str:
        """Decide se deve chamar ferramentas ou responder.

        Args:
            state: Estado atual do subagente

        Returns:
            "call_tools" se houver tool_calls, "respond" caso contrario
        """
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
        description = task.get("description", "No description provided")
        context = task.get("context", {})
        priority = task.get("priority", 3)

        context_str = json.dumps(context, indent=2, ensure_ascii=False) if context else "{}"

        return f"""## Task
{description}

## Context
```json
{context_str}
```

## Priority
{priority} (1 = highest, 5 = lowest)

## Instructions
Analyze the task and context above. Use available tools if needed.
Provide a clear and actionable response."""

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

        Args:
            tool: Tool original do LangChain

        Returns:
            Tool com logging wrapper
        """
        original_func = tool.func

        # Verificar se é async
        if asyncio.iscoroutinefunction(original_func):
            async def logged_tool_async(*args, **kwargs):
                start = time.time()
                try:
                    result = await original_func(*args, **kwargs)
                    duration = (time.time() - start) * 1000

                    log_tool_call(
                        tool_name=tool.name,
                        params=kwargs,
                        result=result,
                        duration_ms=duration,
                        status="success"
                    )
                    return result
                except Exception as e:
                    duration = (time.time() - start) * 1000
                    log_tool_call_error(
                        tool_name=tool.name,
                        params=kwargs,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        duration_ms=duration
                    )
                    raise

            tool.func = logged_tool_async
        else:
            def logged_tool_sync(*args, **kwargs):
                start = time.time()
                try:
                    result = original_func(*args, **kwargs)
                    duration = (time.time() - start) * 1000

                    log_tool_call(
                        tool_name=tool.name,
                        params=kwargs,
                        result=result,
                        duration_ms=duration,
                        status="success"
                    )
                    return result
                except Exception as e:
                    duration = (time.time() - start) * 1000
                    log_tool_call_error(
                        tool_name=tool.name,
                        params=kwargs,
                        error_type=type(e).__name__,
                        error_message=str(e),
                        duration_ms=duration
                    )
                    raise

            tool.func = logged_tool_sync

        return tool

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
        start_time = datetime.utcnow()

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
            end_time = datetime.utcnow()
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
            end_time = datetime.utcnow()
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
            end_time = datetime.utcnow()
            duration_ms = int((end_time - start_time).total_seconds() * 1000)

            return AgentResult(
                agent_name=self.AGENT_NAME,
                success=False,
                data=None,
                error=f"Error: {str(e)}",
                duration_ms=duration_ms,
                tool_calls=[]
            )
