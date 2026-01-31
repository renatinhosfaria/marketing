"use client";

/**
 * Hook para gerenciar o estado do painel de debug do agente
 */

import { useCallback, useState } from "react";
import type {
  AgentStreamChunk,
  DebugContext,
  DebugEvent,
  DebugState,
  GraphNode,
  ToolExecution,
} from "@/types/ai-agent";

// Definição dos nós do grafo LangGraph
const INITIAL_GRAPH_NODES: GraphNode[] = [
  { id: "classify_intent", name: "Classificar Intenção", status: "pending" },
  { id: "gather_data", name: "Coletar Dados", status: "pending" },
  { id: "call_model", name: "Chamar Modelo", status: "pending" },
  { id: "call_tools", name: "Executar Ferramentas", status: "pending" },
  { id: "generate_response", name: "Gerar Resposta", status: "pending" },
];

// Contexto inicial vazio
const INITIAL_CONTEXT: DebugContext = {
  intent: undefined,
  classifications: 0,
  recommendations: 0,
  anomalies: 0,
  forecasts: 0,
  toolCallsCount: 0,
};

// Estado inicial do debug
const INITIAL_STATE: DebugState = {
  isEnabled: false,
  isActive: false,
  nodes: INITIAL_GRAPH_NODES.map((n) => ({ ...n })),
  activeNode: null,
  tools: [],
  context: { ...INITIAL_CONTEXT },
  events: [],
  startTime: null,
  totalDuration: null,
};

/**
 * Hook para gerenciar o estado do painel de debug
 */
export function useDebugState() {
  const [state, setState] = useState<DebugState>({ ...INITIAL_STATE });

  /**
   * Ativa/desativa o painel de debug
   */
  const setEnabled = useCallback((enabled: boolean) => {
    setState((prev) => ({ ...prev, isEnabled: enabled }));
  }, []);

  /**
   * Processa um evento SSE e atualiza o estado do debug
   */
  const processEvent = useCallback((event: AgentStreamChunk) => {
    const timestamp = event.timestamp || Date.now();
    const eventId = `${event.type}-${timestamp}-${Math.random().toString(36).slice(2, 7)}`;

    setState((prev) => {
      // Criar evento para o log
      const debugEvent: DebugEvent = {
        id: eventId,
        type: event.type,
        timestamp,
        data: { ...event },
      };

      // Processar baseado no tipo de evento
      switch (event.type) {
        case "stream_start": {
          // Iniciar novo debug - resetar estado
          return {
            ...prev,
            isActive: true,
            nodes: INITIAL_GRAPH_NODES.map((n) => ({ ...n })),
            activeNode: null,
            tools: [],
            context: { ...INITIAL_CONTEXT },
            events: [debugEvent],
            startTime: timestamp,
            totalDuration: null,
          };
        }

        case "node_start": {
          const nodeId = event.node;
          if (!nodeId) return prev;

          return {
            ...prev,
            activeNode: nodeId,
            nodes: prev.nodes.map((n) =>
              n.id === nodeId
                ? { ...n, status: "active", startTime: timestamp }
                : n,
            ),
            events: [...prev.events, debugEvent],
          };
        }

        case "node_end": {
          const nodeId = event.node;
          if (!nodeId) return prev;

          return {
            ...prev,
            activeNode: prev.activeNode === nodeId ? null : prev.activeNode,
            nodes: prev.nodes.map((n) =>
              n.id === nodeId
                ? {
                    ...n,
                    status: "completed",
                    endTime: timestamp,
                    duration: event.duration_ms,
                  }
                : n,
            ),
            events: [...prev.events, debugEvent],
          };
        }

        case "intent_classified": {
          return {
            ...prev,
            context: {
              ...prev.context,
              intent: event.intent,
            },
            events: [...prev.events, debugEvent],
          };
        }

        case "data_gathered": {
          const counts = event.data_counts;
          if (!counts) return prev;

          return {
            ...prev,
            context: {
              ...prev.context,
              classifications: counts.classifications,
              recommendations: counts.recommendations,
              anomalies: counts.anomalies,
              forecasts: counts.forecasts,
            },
            events: [...prev.events, debugEvent],
          };
        }

        case "tool_start": {
          const toolName = event.tool;
          if (!toolName) return prev;

          const toolId = `${toolName}-${timestamp}`;
          const newTool: ToolExecution = {
            id: toolId,
            name: toolName,
            status: "running",
            startTime: timestamp,
            inputPreview: event.input_preview,
          };

          return {
            ...prev,
            tools: [...prev.tools, newTool],
            context: {
              ...prev.context,
              toolCallsCount: prev.context.toolCallsCount + 1,
            },
            events: [...prev.events, debugEvent],
          };
        }

        case "tool_end": {
          const toolName = event.tool;
          if (!toolName) return prev;

          // Encontrar a ferramenta mais recente com esse nome que está running
          const toolIndex = [...prev.tools]
            .reverse()
            .findIndex((t) => t.name === toolName && t.status === "running");

          if (toolIndex === -1) return prev;

          const actualIndex = prev.tools.length - 1 - toolIndex;

          return {
            ...prev,
            tools: prev.tools.map((t, i) =>
              i === actualIndex
                ? {
                    ...t,
                    status: event.success ? "success" : "error",
                    endTime: timestamp,
                    duration: event.duration_ms,
                    outputPreview: event.output_preview,
                  }
                : t,
            ),
            events: [...prev.events, debugEvent],
          };
        }

        case "text": {
          // Apenas adiciona ao log, não altera outros estados
          return {
            ...prev,
            events: [...prev.events, debugEvent],
          };
        }

        case "done": {
          return {
            ...prev,
            isActive: false,
            activeNode: null,
            totalDuration: event.total_duration_ms || null,
            events: [...prev.events, debugEvent],
          };
        }

        case "error": {
          // Marcar o nó ativo como erro
          return {
            ...prev,
            isActive: false,
            nodes: prev.nodes.map((n) =>
              n.id === prev.activeNode ? { ...n, status: "error" } : n,
            ),
            activeNode: null,
            events: [...prev.events, debugEvent],
          };
        }

        default:
          return {
            ...prev,
            events: [...prev.events, debugEvent],
          };
      }
    });
  }, []);

  /**
   * Reseta o estado do debug para inicial
   */
  const reset = useCallback(() => {
    setState((prev) => ({
      ...INITIAL_STATE,
      isEnabled: prev.isEnabled, // Mantém a preferência de ativação
    }));
  }, []);

  /**
   * Limpa apenas os eventos do log
   */
  const clearEvents = useCallback(() => {
    setState((prev) => ({
      ...prev,
      events: [],
    }));
  }, []);

  return {
    state,
    setEnabled,
    processEvent,
    reset,
    clearEvents,
  };
}

export type UseDebugStateReturn = ReturnType<typeof useDebugState>;
