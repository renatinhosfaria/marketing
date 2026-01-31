"use client";

/**
 * Log de eventos SSE em tempo real
 * Mostra todos os eventos recebidos do servidor
 */

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import type { AgentEventType, DebugEvent } from "@/types/ai-agent";

interface EventsLogProps {
  events: DebugEvent[];
}

// Cores por tipo de evento
const EVENT_COLORS: Record<AgentEventType, string> = {
  stream_start: "text-blue-500",
  node_start: "text-cyan-500",
  node_end: "text-cyan-600",
  intent_classified: "text-purple-500",
  data_gathered: "text-indigo-500",
  text: "text-foreground",
  tool_start: "text-orange-500",
  tool_end: "text-orange-600",
  done: "text-green-500",
  error: "text-destructive",
};

// Labels por tipo de evento
const EVENT_LABELS: Record<AgentEventType, string> = {
  stream_start: "stream",
  node_start: "node",
  node_end: "node",
  intent_classified: "intent",
  data_gathered: "data",
  text: "text",
  tool_start: "tool",
  tool_end: "tool",
  done: "done",
  error: "error",
};

export function EventsLog({ events }: EventsLogProps) {
  const [filter, setFilter] = useState<AgentEventType | "all">("all");

  const filteredEvents = useMemo(() => {
    if (filter === "all") return events;
    return events.filter((e) => e.type === filter);
  }, [events, filter]);

  // Tipos unicos de eventos para o filtro
  const eventTypes = useMemo(() => {
    const types = new Set(events.map((e) => e.type));
    return Array.from(types);
  }, [events]);

  if (events.length === 0) {
    return (
      <div className="text-xs text-muted-foreground text-center py-2">
        Nenhum evento registrado ainda
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {/* Filtros */}
      {eventTypes.length > 1 && (
        <div className="flex flex-wrap gap-1">
          <FilterButton
            active={filter === "all"}
            onClick={() => setFilter("all")}
          >
            Todos
          </FilterButton>
          {eventTypes.map((type) => (
            <FilterButton
              key={type}
              active={filter === type}
              onClick={() => setFilter(type)}
            >
              {EVENT_LABELS[type]}
            </FilterButton>
          ))}
        </div>
      )}

      {/* Lista de eventos */}
      <div className="space-y-0.5 max-h-48 overflow-y-auto">
        {filteredEvents.map((event) => (
          <EventItem key={event.id} event={event} />
        ))}
      </div>
    </div>
  );
}

interface FilterButtonProps {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}

function FilterButton({ active, onClick, children }: FilterButtonProps) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "px-1.5 py-0.5 text-xs rounded transition-colors",
        active
          ? "bg-primary text-primary-foreground"
          : "bg-muted hover:bg-muted/80 text-muted-foreground"
      )}
    >
      {children}
    </button>
  );
}

interface EventItemProps {
  event: DebugEvent;
}

function EventItem({ event }: EventItemProps) {
  // Formatar timestamp com milissegundos
  const date = new Date(event.timestamp);
  const timeStr = date.toLocaleTimeString("pt-BR", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const ms = String(date.getMilliseconds()).padStart(3, "0");
  const time = `${timeStr}.${ms}`;

  const color = EVENT_COLORS[event.type] || "text-foreground";
  const label = EVENT_LABELS[event.type] || event.type;

  // Extrair informacao resumida do evento
  const summary = getEventSummary(event);

  return (
    <div className="flex items-start gap-2 font-mono text-xs">
      {/* Timestamp */}
      <span className="text-muted-foreground shrink-0 w-20">{time}</span>

      {/* Tipo */}
      <span className={cn("shrink-0 w-12", color)}>{label}</span>

      {/* Resumo */}
      <span className="text-muted-foreground truncate flex-1">{summary}</span>
    </div>
  );
}

function getEventSummary(event: DebugEvent): string {
  const data = event.data;

  switch (event.type) {
    case "stream_start":
      return "Iniciando streaming";

    case "node_start":
      return String(data.node || "");

    case "node_end": {
      const duration = data.duration_ms;
      const node = data.node;
      return duration
        ? `${node} (${Number(duration).toFixed(0)}ms)`
        : String(node || "");
    }

    case "intent_classified":
      return String(data.intent || "");

    case "data_gathered": {
      const counts = data.data_counts as Record<string, number> | undefined;
      if (!counts) return "";
      const parts = [];
      if (counts.classifications) parts.push(`${counts.classifications} class`);
      if (counts.recommendations) parts.push(`${counts.recommendations} recs`);
      if (counts.anomalies) parts.push(`${counts.anomalies} anom`);
      if (counts.forecasts) parts.push(`${counts.forecasts} fore`);
      return parts.join(", ");
    }

    case "text": {
      const content = String(data.content || "");
      return content.length > 40 ? content.slice(0, 40) + "..." : content;
    }

    case "tool_start":
      return String(data.tool || "");

    case "tool_end": {
      const duration = data.duration_ms;
      const tool = data.tool;
      const success = data.success;
      const status = success ? "ok" : "erro";
      return duration
        ? `${tool} (${Number(duration).toFixed(0)}ms, ${status})`
        : `${tool} (${status})`;
    }

    case "done": {
      const total = data.total_duration_ms;
      return total ? `Total: ${Number(total).toFixed(0)}ms` : "Concluído";
    }

    case "error":
      return String(data.error || "Erro desconhecido");

    default:
      return "";
  }
}
