"use client";

/**
 * Painel de contexto coletado
 * Exibe os dados coletados durante a execucao do agente
 */

import {
  AlertTriangle,
  BarChart3,
  Lightbulb,
  Target,
  TrendingUp,
  Wrench,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { DebugContext } from "@/types/ai-agent";

interface ContextPanelProps {
  context: DebugContext;
}

export function ContextPanel({ context }: ContextPanelProps) {
  const hasData =
    context.intent ||
    context.classifications > 0 ||
    context.recommendations > 0 ||
    context.anomalies > 0 ||
    context.forecasts > 0;

  if (!hasData) {
    return (
      <div className="text-xs text-muted-foreground text-center py-2">
        Nenhum dado coletado ainda
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Intencao */}
      {context.intent && (
        <ContextItem
          icon={Target}
          label="Intenção"
          value={formatIntent(context.intent)}
          color="text-primary"
        />
      )}

      {/* Classificacoes */}
      {context.classifications > 0 && (
        <ContextItem
          icon={BarChart3}
          label="Classificações"
          value={`${context.classifications} campanhas`}
          color="text-blue-500"
        />
      )}

      {/* Recomendacoes */}
      {context.recommendations > 0 && (
        <ContextItem
          icon={Lightbulb}
          label="Recomendações"
          value={`${context.recommendations} ativas`}
          color="text-yellow-500"
        />
      )}

      {/* Anomalias */}
      {context.anomalies > 0 && (
        <ContextItem
          icon={AlertTriangle}
          label="Anomalias"
          value={`${context.anomalies} detectadas`}
          color="text-orange-500"
        />
      )}

      {/* Previsoes */}
      {context.forecasts > 0 && (
        <ContextItem
          icon={TrendingUp}
          label="Previsões"
          value={`${context.forecasts} campanhas`}
          color="text-green-500"
        />
      )}

      {/* Tool Calls */}
      {context.toolCallsCount > 0 && (
        <ContextItem
          icon={Wrench}
          label="Ferramentas"
          value={`${context.toolCallsCount} chamadas`}
          color="text-purple-500"
        />
      )}
    </div>
  );
}

interface ContextItemProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  color: string;
}

function ContextItem({ icon: Icon, label, value, color }: ContextItemProps) {
  return (
    <div className="flex items-center gap-2 text-sm">
      <Icon className={cn("h-4 w-4 shrink-0", color)} />
      <span className="text-muted-foreground">{label}:</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

function formatIntent(intent: string): string {
  const intentMap: Record<string, string> = {
    analyze: "Analisar",
    compare: "Comparar",
    recommend: "Recomendar",
    forecast: "Prever",
    troubleshoot: "Resolver Problema",
    general: "Geral",
  };

  return intentMap[intent] || intent;
}
