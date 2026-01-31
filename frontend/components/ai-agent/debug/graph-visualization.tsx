"use client";

/**
 * Visualizacao do grafo LangGraph
 * Mostra os nos do grafo com status em tempo real
 */

import { Check, Circle, Loader2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { GraphNode, NodeStatus } from "@/types/ai-agent";

interface GraphVisualizationProps {
  nodes: GraphNode[];
  activeNode: string | null;
}

export function GraphVisualization({
  nodes,
  activeNode,
}: GraphVisualizationProps) {
  return (
    <div className="space-y-1">
      {nodes.map((node, index) => (
        <div key={node.id}>
          <NodeItem
            node={node}
            isActive={node.id === activeNode}
            isLast={index === nodes.length - 1}
          />
        </div>
      ))}
    </div>
  );
}

interface NodeItemProps {
  node: GraphNode;
  isActive: boolean;
  isLast: boolean;
}

function NodeItem({ node, isActive, isLast }: NodeItemProps) {
  return (
    <div className="flex items-start gap-2">
      {/* Indicador de status + linha conectora */}
      <div className="flex flex-col items-center">
        <StatusIcon status={node.status} isActive={isActive} />
        {!isLast && (
          <div
            className={cn(
              "w-0.5 h-4 mt-1",
              node.status === "completed" ? "bg-green-500/50" : "bg-border"
            )}
          />
        )}
      </div>

      {/* Conteudo do no */}
      <div className="flex-1 min-w-0 pb-2">
        <div className="flex items-center justify-between gap-2">
          <span
            className={cn(
              "text-sm truncate",
              node.status === "completed" && "text-green-600 dark:text-green-400",
              node.status === "active" && "text-primary font-medium",
              node.status === "error" && "text-destructive",
              node.status === "pending" && "text-muted-foreground"
            )}
          >
            {node.name}
          </span>

          {/* Duracao */}
          {node.duration !== undefined && (
            <span className="text-xs font-mono text-muted-foreground shrink-0">
              {node.duration.toFixed(0)}ms
            </span>
          )}

          {/* Indicador de ativo */}
          {isActive && node.status === "active" && (
            <span className="text-xs text-muted-foreground animate-pulse">
              ...
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

interface StatusIconProps {
  status: NodeStatus;
  isActive: boolean;
}

function StatusIcon({ status, isActive }: StatusIconProps) {
  const baseClasses = "h-4 w-4 shrink-0";

  switch (status) {
    case "completed":
      return (
        <div className="p-0.5 rounded-full bg-green-500/20">
          <Check className={cn(baseClasses, "text-green-500 h-3 w-3")} />
        </div>
      );

    case "active":
      return (
        <div className="relative">
          <div className="absolute inset-0 animate-ping rounded-full bg-primary/30" />
          <Loader2
            className={cn(baseClasses, "text-primary animate-spin relative")}
          />
        </div>
      );

    case "error":
      return (
        <div className="p-0.5 rounded-full bg-destructive/20">
          <X className={cn(baseClasses, "text-destructive h-3 w-3")} />
        </div>
      );

    case "skipped":
      return (
        <Circle
          className={cn(baseClasses, "text-muted-foreground/50")}
          strokeDasharray="2 2"
        />
      );

    case "pending":
    default:
      return (
        <Circle className={cn(baseClasses, "text-muted-foreground/30")} />
      );
  }
}
