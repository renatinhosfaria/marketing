"use client";

/**
 * Hooks do módulo AI Agent
 *
 * Combina todos os hooks de chat, análise, status, conversas, sugestões e exclusão
 * em um único arquivo. Comunica-se com o FastAPI via /api/v1/agent.
 */

import { useState, useCallback, useEffect, useRef } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import type {
  AgentMessage,
  AgentChatResponse,
  AgentStreamChunk,
  AgentConversationHistoryResponse,
  AgentConversationListResponse,
  AgentAnalyzeResponse,
  AgentStatus,
  AgentSuggestionsResponse,
  ChatState,
} from "@/types/ai-agent";

// ---------------------------------------------------------------------------
// Constantes
// ---------------------------------------------------------------------------

const AGENT_BASE = "/api/v1/agent";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Fetch com autenticação para streaming SSE.
 * Não usa apiFetch porque precisamos do Response raw para ReadableStream.
 */
async function streamFetch(
  path: string,
  options: RequestInit = {},
): Promise<Response> {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) || {}),
  };

  return fetch(`${API_BASE}${AGENT_BASE}${path}`, {
    ...options,
    headers,
  });
}

// ---------------------------------------------------------------------------
// useAgentChat — chat com streaming SSE
// ---------------------------------------------------------------------------

interface UseAgentChatOptions {
  configId: number;
  threadId?: string;
  onError?: (error: string) => void;
  onDebugEvent?: (event: AgentStreamChunk) => void;
}

export function useAgentChat({
  configId,
  threadId: initialThreadId,
  onError,
  onDebugEvent,
}: UseAgentChatOptions) {
  const queryClient = useQueryClient();
  const abortControllerRef = useRef<AbortController | null>(null);

  const [state, setState] = useState<ChatState>({
    messages: [],
    threadId: initialThreadId || null,
    isLoading: false,
    isStreaming: false,
    error: null,
  });

  // Buscar histórico da conversa se tiver threadId
  const { data: history } = useQuery<AgentConversationHistoryResponse | null>({
    queryKey: ["agent-history", state.threadId],
    queryFn: async () => {
      if (!state.threadId) return null;
      const res = await apiFetch(
        `${AGENT_BASE}/conversations/${state.threadId}`,
      );
      if (!res.ok) throw new Error("Falha ao carregar histórico");
      return res.json();
    },
    enabled: !!state.threadId,
  });

  useEffect(() => {
    if (!history?.messages) return;
    const threadId = history.thread_id ?? state.threadId;

    setState((prev) => ({
      ...prev,
      messages: history.messages.map((m, i) => ({
        ...m,
        id: `${threadId}-${i}`,
      })),
    }));
  }, [history, state.threadId]);

  // Enviar mensagem (sem streaming)
  const sendMessageMutation = useMutation({
    mutationFn: async (message: string): Promise<AgentChatResponse> => {
      const res = await apiFetch(`${AGENT_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message,
          config_id: configId,
          thread_id: state.threadId,
        }),
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || "Erro ao enviar mensagem");
      }
      return res.json();
    },
    onMutate: async (message) => {
      // Adicionar mensagem do usuário otimisticamente
      const userMessage: AgentMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: message,
        created_at: new Date().toISOString(),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        isLoading: true,
        error: null,
      }));
    },
    onSuccess: (data) => {
      const assistantMessage: AgentMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: data.response,
        created_at: new Date().toISOString(),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
        threadId: data.thread_id,
        isLoading: false,
      }));

      // Invalidar lista de conversas
      queryClient.invalidateQueries({ queryKey: ["agent-conversations"] });
    },
    onError: (error: Error) => {
      setState((prev) => ({
        ...prev,
        isLoading: false,
        error: error.message,
      }));
      onError?.(error.message);
    },
  });

  // Enviar mensagem com streaming
  const sendMessageStream = useCallback(
    async (message: string) => {
      // Adicionar mensagem do usuário
      const userMessage: AgentMessage = {
        id: `user-${Date.now()}`,
        role: "user",
        content: message,
        created_at: new Date().toISOString(),
      };

      // Mensagem do assistente (vai ser preenchida com streaming)
      const assistantMessageId = `assistant-${Date.now()}`;
      const assistantMessage: AgentMessage = {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        isStreaming: true,
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage, assistantMessage],
        isStreaming: true,
        error: null,
      }));

      try {
        abortControllerRef.current = new AbortController();

        const res = await streamFetch("/chat/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message,
            config_id: configId,
            thread_id: state.threadId,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!res.ok) {
          throw new Error("Falha ao iniciar streaming");
        }

        const reader = res.body?.getReader();
        if (!reader) throw new Error("Streaming não suportado");

        const decoder = new TextDecoder();
        let buffer = "";
        let newThreadId = state.threadId;

        let done = false;
        while (!done) {
          const result = await reader.read();
          done = result.done;
          if (done) break;

          buffer += decoder.decode(result.value, { stream: true });

          // Processar eventos SSE
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const chunk: AgentStreamChunk = JSON.parse(line.slice(6));

                // Emitir evento de debug se callback estiver definido
                onDebugEvent?.(chunk);

                if (chunk.type === "text" && chunk.content) {
                  setState((prev) => ({
                    ...prev,
                    messages: prev.messages.map((m) =>
                      m.id === assistantMessageId
                        ? { ...m, content: m.content + chunk.content }
                        : m,
                    ),
                  }));
                }

                if (chunk.thread_id) {
                  newThreadId = chunk.thread_id;
                }

                if (chunk.type === "error") {
                  throw new Error(chunk.error || "Erro no streaming");
                }
              } catch {
                // Ignorar erros de parsing de chunks incompletos
              }
            }
          }
        }

        // Finalizar streaming
        setState((prev) => ({
          ...prev,
          messages: prev.messages.map((m) =>
            m.id === assistantMessageId ? { ...m, isStreaming: false } : m,
          ),
          threadId: newThreadId,
          isStreaming: false,
        }));

        // Invalidar lista de conversas
        queryClient.invalidateQueries({ queryKey: ["agent-conversations"] });
      } catch (error) {
        if (error instanceof Error && error.name === "AbortError") {
          return;
        }

        const errorMessage =
          error instanceof Error ? error.message : "Erro desconhecido";

        setState((prev) => ({
          ...prev,
          messages: prev.messages.filter((m) => m.id !== assistantMessageId),
          isStreaming: false,
          error: errorMessage,
        }));

        onError?.(errorMessage);
      }
    },
    [configId, state.threadId, queryClient, onError, onDebugEvent],
  );

  // Cancelar streaming
  const cancelStream = useCallback(() => {
    abortControllerRef.current?.abort();
    setState((prev) => ({
      ...prev,
      isStreaming: false,
      messages: prev.messages.map((m) =>
        m.isStreaming ? { ...m, isStreaming: false } : m,
      ),
    }));
  }, []);

  // Limpar conversa
  const clearConversation = useCallback(async () => {
    if (!state.threadId) {
      setState((prev) => ({ ...prev, messages: [] }));
      return;
    }

    try {
      await apiFetch(`${AGENT_BASE}/conversations/${state.threadId}`, {
        method: "DELETE",
      });

      setState({
        messages: [],
        threadId: null,
        isLoading: false,
        isStreaming: false,
        error: null,
      });

      queryClient.invalidateQueries({ queryKey: ["agent-conversations"] });
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : "Erro ao limpar conversa";
      onError?.(errorMessage);
    }
  }, [state.threadId, queryClient, onError]);

  // Nova conversa
  const newConversation = useCallback(() => {
    setState({
      messages: [],
      threadId: null,
      isLoading: false,
      isStreaming: false,
      error: null,
    });
  }, []);

  // Carregar conversa existente
  const loadConversation = useCallback((threadId: string) => {
    setState((prev) => ({
      ...prev,
      threadId,
      messages: [],
    }));
  }, []);

  return {
    ...state,
    sendMessage: sendMessageMutation.mutate,
    sendMessageStream,
    cancelStream,
    clearConversation,
    newConversation,
    loadConversation,
  };
}

// ---------------------------------------------------------------------------
// useAgentAnalyze — análise rápida sem chat
// ---------------------------------------------------------------------------

export function useAgentAnalyze(configId: number | null) {
  return useMutation({
    mutationFn: async (query: string): Promise<AgentAnalyzeResponse> => {
      if (!configId) {
        throw new Error("Configuração não definida");
      }

      const res = await apiFetch(`${AGENT_BASE}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          config_id: configId,
        }),
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || "Erro ao analisar");
      }

      return res.json();
    },
  });
}

// ---------------------------------------------------------------------------
// useAgentStatus — verificar status do agente
// ---------------------------------------------------------------------------

export function useAgentStatus() {
  return useQuery<AgentStatus>({
    queryKey: ["agent-status"],
    queryFn: async () => {
      const res = await apiFetch(`${AGENT_BASE}/status`);
      if (!res.ok) {
        return {
          status: "offline" as const,
          llm_provider: "unknown",
          model: "unknown",
          version: "1.0.0",
        };
      }
      return res.json();
    },
    refetchInterval: 30000, // Verificar a cada 30 segundos
    retry: false,
  });
}

// ---------------------------------------------------------------------------
// useConversations — listar conversas do agente
// ---------------------------------------------------------------------------

interface UseConversationsOptions {
  limit?: number;
  offset?: number;
}

export function useConversations(
  configId: number | null,
  options: UseConversationsOptions = {},
) {
  const limit = options.limit ?? 20;
  const offset = options.offset ?? 0;

  return useQuery<AgentConversationListResponse>({
    queryKey: ["agent-conversations", configId, limit, offset],
    queryFn: async () => {
      if (!configId) {
        return { conversations: [], total: 0 };
      }

      const params = new URLSearchParams();
      params.set("config_id", String(configId));
      params.set("limit", String(limit));
      params.set("offset", String(offset));

      const res = await apiFetch(
        `${AGENT_BASE}/conversations?${params.toString()}`,
      );
      if (!res.ok) {
        throw new Error("Falha ao carregar conversas");
      }
      return res.json();
    },
    enabled: !!configId,
  });
}

// ---------------------------------------------------------------------------
// useAgentSuggestions — sugestões proativas do agente
// ---------------------------------------------------------------------------

export function useAgentSuggestions(configId: number | null) {
  return useQuery<AgentSuggestionsResponse>({
    queryKey: ["agent-suggestions", configId],
    queryFn: async () => {
      if (!configId) {
        return { config_id: 0, suggestions: [] };
      }

      const res = await apiFetch(`${AGENT_BASE}/suggestions/${configId}`);
      if (!res.ok) {
        throw new Error("Falha ao carregar sugestões");
      }
      return res.json();
    },
    enabled: !!configId,
  });
}

// ---------------------------------------------------------------------------
// useDeleteConversation — deletar conversa
// ---------------------------------------------------------------------------

interface DeleteConversationResponse {
  success: boolean;
  message: string;
}

export function useDeleteConversation(configId: number | null) {
  const queryClient = useQueryClient();

  return useMutation<DeleteConversationResponse, Error, string>({
    mutationFn: async (threadId: string) => {
      const res = await apiFetch(
        `${AGENT_BASE}/conversations/${threadId}`,
        {
          method: "DELETE",
        },
      );

      if (!res.ok) {
        const error = await res.text();
        throw new Error(error || "Falha ao excluir conversa");
      }

      return res.json();
    },
    onSuccess: () => {
      // Invalidar cache de conversas para recarregar a lista
      queryClient.invalidateQueries({
        queryKey: ["agent-conversations", configId],
      });
    },
  });
}
