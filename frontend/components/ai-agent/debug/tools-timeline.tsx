"use client";

/**
 * Timeline de ferramentas executadas
 * Mostra as ferramentas chamadas pelo agente com status e duracao
 */

import { Check, Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ToolExecution } from "@/types/ai-agent";

interface ToolsTimelineProps {
  tools: ToolExecution[];
}

export function ToolsTimeline({ tools }: ToolsTimelineProps) {
  if (tools.length === 0) {
    return (
      <div className="text-xs text-muted-foreground text-center py-2">
        Nenhuma ferramenta executada ainda
      </div>
    );
  }

  return (
    <div className="space-y-1.5">
      {tools.map((tool) => (
        <ToolItem key={tool.id} tool={tool} />
      ))}
    </div>
  );
}

interface ToolItemProps {
  tool: ToolExecution;
}

function ToolItem({ tool }: ToolItemProps) {
  const time = new Date(tool.startTime).toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <div className="flex items-start gap-2 text-sm">
      {/* Timestamp */}
      <span className="font-mono text-xs text-muted-foreground shrink-0 w-16">
        {time}
      </span>

      {/* Status icon */}
      <StatusIcon status={tool.status} />

      {/* Tool name */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between gap-2">
          <span
            className={cn(
              "font-mono text-xs truncate",
              tool.status === "success" && "text-green-600 dark:text-green-400",
              tool.status === "running" && "text-primary",
              tool.status === "error" && "text-destructive"
            )}
          >
            {formatToolName(tool.name)}
          </span>

          {/* Duration */}
          {tool.duration !== undefined && (
            <span className="font-mono text-xs text-muted-foreground shrink-0">
              {tool.duration.toFixed(0)}ms
            </span>
          )}

          {/* Running indicator */}
          {tool.status === "running" && (
            <span className="text-xs text-muted-foreground animate-pulse">
              ...
            </span>
          )}
        </div>

        {/* Preview */}
        {tool.outputPreview && tool.status !== "running" && (
          <div className="mt-0.5 text-xs text-muted-foreground truncate max-w-full">
            {tool.outputPreview}
          </div>
        )}
      </div>
    </div>
  );
}

interface StatusIconProps {
  status: ToolExecution["status"];
}

function StatusIcon({ status }: StatusIconProps) {
  const baseClasses = "h-3.5 w-3.5 shrink-0";

  switch (status) {
    case "success":
      return <Check className={cn(baseClasses, "text-green-500")} />;
    case "running":
      return <Loader2 className={cn(baseClasses, "text-primary animate-spin")} />;
    case "error":
      return <X className={cn(baseClasses, "text-destructive")} />;
  }
}

function formatToolName(name: string): string {
  // Formatar nome da ferramenta para exibicao
  return name
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
