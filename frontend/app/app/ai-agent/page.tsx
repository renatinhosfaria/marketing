"use client";

/**
 * Pagina principal do agente de IA
 * Layout com 3 paineis: Sidebar | Chat | Debug (lg+)
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Bot, Bug, Settings } from "lucide-react";
import { useFacebookAdsConfigs } from "@/hooks/use-facebook-ads";
import {
  useAgentChat,
  useAgentStatus,
  useConversations,
  useAgentSuggestions,
  useDeleteConversation,
} from "@/hooks/use-ai-agent";
import { useDebugState } from "@/hooks/use-debug-state";
import { AgentSidebar } from "@/components/ai-agent/agent-sidebar";
import { ChatInterface } from "@/components/ai-agent/chat-interface";
import { DebugPanel } from "@/components/ai-agent/debug-panel";
import { Button } from "@/components/ui/button";
import type { AgentStreamChunk } from "@/types/ai-agent";

export default function AIAgentPage() {
  const { data: configs, isLoading: loadingConfigs } = useFacebookAdsConfigs();
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null);
  const prevConfigId = useRef<number | null>(null);

  // Estado do painel de debug
  const {
    state: debugState,
    setEnabled: setDebugEnabled,
    processEvent: processDebugEvent,
    reset: resetDebug,
    clearEvents: clearDebugEvents,
  } = useDebugState();

  // Callback para processar eventos de debug
  const handleDebugEvent = useCallback(
    (event: AgentStreamChunk) => {
      if (debugState.isEnabled) {
        processDebugEvent(event);
      }
    },
    [debugState.isEnabled, processDebugEvent]
  );

  // Auto-selecionar config ativa ao carregar
  useEffect(() => {
    if (!selectedConfigId && configs?.length) {
      const activeConfig = configs.find((config) => config.isActive);
      setSelectedConfigId(activeConfig?.id ?? configs[0].id);
    }
  }, [configs, selectedConfigId]);

  const {
    messages,
    threadId,
    isLoading,
    isStreaming,
    error,
    sendMessageStream,
    cancelStream,
    clearConversation,
    newConversation,
    loadConversation,
  } = useAgentChat({
    configId: selectedConfigId ?? 0,
    onError: (err) => console.error("Agent error:", err),
    onDebugEvent: handleDebugEvent,
  });

  // Resetar conversa ao trocar de config
  useEffect(() => {
    if (
      prevConfigId.current !== null &&
      prevConfigId.current !== selectedConfigId
    ) {
      newConversation();
      resetDebug();
    }
    prevConfigId.current = selectedConfigId;
  }, [selectedConfigId, newConversation, resetDebug]);

  const { data: status } = useAgentStatus();
  const { data: conversations, isLoading: loadingConversations } =
    useConversations(selectedConfigId);
  const { data: suggestions, isLoading: loadingSuggestions } =
    useAgentSuggestions(selectedConfigId);
  const deleteConversation = useDeleteConversation(selectedConfigId);

  const handleDeleteConversation = useCallback(
    (conversationThreadId: string) => {
      if (confirm("Deseja excluir esta conversa?")) {
        deleteConversation.mutate(conversationThreadId, {
          onSuccess: () => {
            if (conversationThreadId === threadId) {
              newConversation();
              resetDebug();
            }
          },
        });
      }
    },
    [deleteConversation, threadId, newConversation, resetDebug]
  );

  // Loading state
  if (loadingConfigs) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Bot className="h-12 w-12 text-primary mx-auto mb-4 animate-pulse" />
          <p className="text-gray-600">Carregando assistente...</p>
        </div>
      </div>
    );
  }

  // Nenhuma conta configurada
  if (!configs?.length) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md">
          <Bot className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Nenhuma conta configurada
          </h2>
          <p className="text-gray-600 mb-4">
            Configure uma conta do Facebook Ads para usar o agente de IA.
          </p>
          <a
            href="/app/facebook-ads/settings"
            className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:opacity-90"
          >
            <Settings className="h-4 w-4" />
            Configurar Facebook Ads
          </a>
        </div>
      </div>
    );
  }

  const statusLabel =
    status?.status === "online"
      ? "Online"
      : status?.status === "error"
        ? "Erro"
        : "Offline";

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between mb-6">
          <div className="flex items-center gap-3">
            <Bot className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Agente de Trafego Pago
              </h1>
              <p className="text-gray-500">
                Analises e recomendacoes em tempo real
              </p>
            </div>
          </div>

          <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
            {/* Status indicator */}
            <div className="flex items-center gap-2 text-sm text-gray-600">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  status?.status === "online"
                    ? "bg-green-500"
                    : "bg-gray-400"
                }`}
              />
              <span>{statusLabel}</span>
            </div>

            {/* Config selector */}
            <select
              value={selectedConfigId ?? ""}
              onChange={(e) => setSelectedConfigId(Number(e.target.value))}
              className="px-4 py-2 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-primary focus:border-primary"
            >
              {configs.map((config) => (
                <option key={config.id} value={config.id}>
                  {config.accountName} ({config.accountId})
                </option>
              ))}
            </select>

            {/* Botao de Debug */}
            <Button
              variant={debugState.isEnabled ? "default" : "outline"}
              size="sm"
              onClick={() => setDebugEnabled(!debugState.isEnabled)}
              className="gap-2"
              title={debugState.isEnabled ? "Desativar Debug" : "Ativar Debug"}
            >
              <Bug className="h-4 w-4" />
              <span className="hidden sm:inline">Debug</span>
            </Button>
          </div>
        </div>

        {/* Layout com 3 paineis */}
        <div className="bg-white border border-gray-200 rounded-xl shadow-sm min-h-[70vh] flex flex-col md:flex-row">
          {/* Sidebar de conversas */}
          <div className="md:w-72 w-full border-b md:border-b-0 md:border-r shrink-0">
            <AgentSidebar
              conversations={conversations?.conversations ?? []}
              activeThreadId={threadId}
              isLoading={loadingConversations}
              isDeleting={deleteConversation.isPending}
              onSelect={(id) => {
                loadConversation(id);
                resetDebug();
              }}
              onNewConversation={() => {
                newConversation();
                resetDebug();
              }}
              onDelete={handleDeleteConversation}
            />
          </div>

          {/* Area de chat */}
          <div className="flex-1 min-h-[70vh]">
            <ChatInterface
              messages={messages}
              isLoading={isLoading}
              isStreaming={isStreaming}
              error={error}
              suggestions={suggestions?.suggestions ?? []}
              isSuggestionsLoading={loadingSuggestions}
              onSendMessage={sendMessageStream}
              onCancelStream={cancelStream}
              onClearConversation={() => {
                clearConversation();
                resetDebug();
              }}
              onNewConversation={() => {
                newConversation();
                resetDebug();
              }}
            />
          </div>

          {/* Painel de debug (apenas quando ativado, visivel em lg+) */}
          {debugState.isEnabled && (
            <div className="w-80 shrink-0 hidden lg:block">
              <DebugPanel
                state={debugState}
                onClose={() => setDebugEnabled(false)}
                onClearEvents={clearDebugEvents}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
