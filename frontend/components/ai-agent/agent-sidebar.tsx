"use client";

/**
 * Sidebar com historico de conversas do agente
 */

import { Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { AgentConversation } from "@/types/ai-agent";

interface AgentSidebarProps {
  conversations: AgentConversation[];
  activeThreadId: string | null;
  isLoading?: boolean;
  isDeleting?: boolean;
  onSelect: (threadId: string) => void;
  onNewConversation: () => void;
  onDelete: (threadId: string) => void;
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleDateString("pt-BR", {
    day: "2-digit",
    month: "2-digit",
  });
}

export function AgentSidebar({
  conversations,
  activeThreadId,
  isLoading = false,
  isDeleting = false,
  onSelect,
  onNewConversation,
  onDelete,
}: AgentSidebarProps) {
  return (
    <aside className="w-full md:w-72 border-r bg-background h-full flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <h3 className="font-semibold">Conversas</h3>
        <Button
          variant="ghost"
          size="icon"
          onClick={onNewConversation}
          title="Nova conversa"
          className="h-8 w-8"
        >
          <Plus className="h-4 w-4" />
        </Button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {isLoading ? (
          <div className="space-y-2">
            {Array.from({ length: 5 }).map((_, index) => (
              <div
                key={`conv-skeleton-${index}`}
                className="h-12 rounded-md bg-muted animate-pulse"
              />
            ))}
          </div>
        ) : conversations.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Sem conversas recentes.
          </p>
        ) : (
          conversations.map((conversation) => (
            <div
              key={conversation.thread_id}
              className={cn(
                "w-full text-left px-3 py-2 rounded-md border transition-colors flex items-start gap-2 group",
                conversation.thread_id === activeThreadId
                  ? "border-primary bg-primary/5 text-foreground"
                  : "border-transparent hover:bg-muted"
              )}
            >
              <button
                type="button"
                onClick={() => onSelect(conversation.thread_id)}
                className="flex-1 text-left min-w-0"
              >
                <div className="text-sm font-medium truncate">
                  {conversation.title || "Conversa sem título"}
                </div>
                <div className="flex items-center justify-between text-xs text-muted-foreground mt-1">
                  <span>{conversation.message_count} mensagens</span>
                  <span>{formatDate(conversation.updated_at)}</span>
                </div>
              </button>
              <Button
                variant="ghost"
                size="icon"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(conversation.thread_id);
                }}
                disabled={isDeleting}
                title="Excluir conversa"
                className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity text-destructive hover:text-destructive hover:bg-destructive/10 shrink-0"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
