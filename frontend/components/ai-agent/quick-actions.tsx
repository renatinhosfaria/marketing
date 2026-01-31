"use client";

/**
 * Acoes rapidas sugeridas para o agente
 */

import type { AgentSuggestion } from "@/types/ai-agent";

interface QuickActionsProps {
  suggestions: AgentSuggestion[];
  onSelect: (text: string) => void;
  isLoading?: boolean;
}

export function QuickActions({
  suggestions,
  onSelect,
  isLoading = false,
}: QuickActionsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-2">
        {Array.from({ length: 4 }).map((_, index) => (
          <div
            key={`suggestion-skeleton-${index}`}
            className="h-10 rounded-lg bg-muted animate-pulse"
          />
        ))}
      </div>
    );
  }

  if (!suggestions.length) {
    return (
      <p className="text-sm text-muted-foreground">
        Sem sugestões disponíveis no momento.
      </p>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-2">
      {suggestions.map((suggestion) => (
        <button
          key={suggestion.id}
          type="button"
          onClick={() => onSelect(suggestion.text)}
          className="px-3 py-2 rounded-lg border text-muted-foreground hover:bg-muted hover:text-foreground transition-colors text-left text-sm"
        >
          {suggestion.text}
        </button>
      ))}
    </div>
  );
}
