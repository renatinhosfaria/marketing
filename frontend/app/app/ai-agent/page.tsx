"use client";

import { ChatContainer } from "@/components/ai-agent/chat-container";
import { useFacebookAdsConfigs } from "@/hooks/use-facebook-ads";

/**
 * Pagina principal do Agente de IA.
 * Usa a primeira conta ativa de Facebook Ads para conectar o agente.
 */
export default function AIAgentPage() {
  const { data: configs, isLoading } = useFacebookAdsConfigs();
  const activeConfig = configs?.find((c) => c.isActive);
  const accountId = activeConfig?.accountId || "";

  if (isLoading) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
          <div className="animate-pulse text-muted-foreground">
            Carregando configuracao...
          </div>
        </div>
      </div>
    );
  }

  if (!accountId) {
    return (
      <div className="container mx-auto p-6">
        <div className="flex flex-col items-center justify-center h-[calc(100vh-8rem)] gap-4">
          <p className="text-muted-foreground text-lg">
            Nenhuma conta de Facebook Ads ativa encontrada.
          </p>
          <p className="text-muted-foreground text-sm">
            Configure uma conta em Facebook Ads &gt; Configuracoes para usar o agente.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6">
      <div className="mb-4">
        <h1 className="text-3xl font-bold tracking-tight">Agente IA</h1>
        <p className="text-muted-foreground">
          Assistente inteligente para gestao de campanhas de marketing
          {activeConfig?.accountName && (
            <span className="ml-1">
              — {activeConfig.accountName}
            </span>
          )}
        </p>
      </div>

      <ChatContainer accountId={accountId} />
    </div>
  );
}
