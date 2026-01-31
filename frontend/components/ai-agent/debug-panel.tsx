"use client";

/**
 * Painel de debug do agente de IA
 * Exibe informacoes em tempo real sobre a execucao do agente
 */

import { useState } from "react";
import { Bug, ChevronDown, ChevronUp, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { DebugState } from "@/types/ai-agent";
import { GraphVisualization } from "./debug/graph-visualization";
import { ContextPanel } from "./debug/context-panel";
import { ToolsTimeline } from "./debug/tools-timeline";
import { EventsLog } from "./debug/events-log";

interface DebugPanelProps {
  state: DebugState;
  onClose: () => void;
  onClearEvents: () => void;
  className?: string;
}

type SectionKey = "graph" | "context" | "tools" | "events";

export function DebugPanel({
  state,
  onClose,
  onClearEvents,
  className,
}: DebugPanelProps) {
  const [collapsedSections, setCollapsedSections] = useState<Set<SectionKey>>(
    new Set()
  );

  const toggleSection = (section: SectionKey) => {
    setCollapsedSections((prev) => {
      const next = new Set(prev);
      if (next.has(section)) {
        next.delete(section);
      } else {
        next.add(section);
      }
      return next;
    });
  };

  const isCollapsed = (section: SectionKey) => collapsedSections.has(section);

  return (
    <div
      className={cn(
        "flex flex-col h-full border-l bg-background",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b bg-muted/30">
        <div className="flex items-center gap-2">
          <Bug className="h-4 w-4 text-primary" />
          <span className="font-medium text-sm">Debug do Agente</span>
          {state.isActive && (
            <span className="flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-primary opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-primary" />
            </span>
          )}
        </div>
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
        </Button>
      </div>

      {/* Duracao total */}
      {state.totalDuration !== null && (
        <div className="px-3 py-1.5 text-xs text-muted-foreground border-b bg-muted/20">
          Tempo total: <span className="font-mono">{state.totalDuration.toFixed(0)}ms</span>
        </div>
      )}

      {/* Content */}
      <ScrollArea className="flex-1">
        <div className="p-3 space-y-3">
          {/* Grafo de Execucao */}
          <CollapsibleSection
            title="Grafo de Execução"
            isCollapsed={isCollapsed("graph")}
            onToggle={() => toggleSection("graph")}
          >
            <GraphVisualization
              nodes={state.nodes}
              activeNode={state.activeNode}
            />
          </CollapsibleSection>

          {/* Contexto Coletado */}
          <CollapsibleSection
            title="Contexto Coletado"
            isCollapsed={isCollapsed("context")}
            onToggle={() => toggleSection("context")}
          >
            <ContextPanel context={state.context} />
          </CollapsibleSection>

          {/* Ferramentas Executadas */}
          <CollapsibleSection
            title="Ferramentas Executadas"
            badge={state.tools.length > 0 ? state.tools.length : undefined}
            isCollapsed={isCollapsed("tools")}
            onToggle={() => toggleSection("tools")}
          >
            <ToolsTimeline tools={state.tools} />
          </CollapsibleSection>

          {/* Log de Eventos */}
          <CollapsibleSection
            title="Log de Eventos"
            badge={state.events.length > 0 ? state.events.length : undefined}
            isCollapsed={isCollapsed("events")}
            onToggle={() => toggleSection("events")}
            actions={
              state.events.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-5 text-xs px-1.5"
                  onClick={(e) => {
                    e.stopPropagation();
                    onClearEvents();
                  }}
                >
                  Limpar
                </Button>
              )
            }
          >
            <EventsLog events={state.events} />
          </CollapsibleSection>
        </div>
      </ScrollArea>
    </div>
  );
}

interface CollapsibleSectionProps {
  title: string;
  badge?: number;
  isCollapsed: boolean;
  onToggle: () => void;
  actions?: React.ReactNode;
  children: React.ReactNode;
}

function CollapsibleSection({
  title,
  badge,
  isCollapsed,
  onToggle,
  actions,
  children,
}: CollapsibleSectionProps) {
  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={onToggle}
        className="flex items-center justify-between w-full px-3 py-2 text-left bg-muted/30 hover:bg-muted/50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {title}
          </span>
          {badge !== undefined && (
            <span className="px-1.5 py-0.5 text-xs rounded-full bg-primary/10 text-primary font-mono">
              {badge}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {actions}
          {isCollapsed ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          )}
        </div>
      </button>
      {!isCollapsed && <div className="p-3 pt-2">{children}</div>}
    </div>
  );
}
