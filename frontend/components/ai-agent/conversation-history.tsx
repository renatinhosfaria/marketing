"use client";

import { useMemo } from "react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  MessageSquarePlus,
  MessageSquare,
  Trash2,
  PanelLeftClose,
  Loader2,
} from "lucide-react";
import { isToday, isYesterday, subDays, isAfter, format } from "date-fns";
import { ptBR } from "date-fns/locale";
import type { ConversationPreview } from "@/types/ai-agent";

interface ConversationSidebarProps {
  conversations: ConversationPreview[];
  isLoading: boolean;
  activeThreadId: string | null;
  onSelectConversation: (threadId: string) => void;
  onNewConversation: () => void;
  onDeleteConversation: (threadId: string) => void;
  onClose: () => void;
  className?: string;
}

type TimeGroup = {
  label: string;
  conversations: ConversationPreview[];
};

function groupByTime(conversations: ConversationPreview[]): TimeGroup[] {
  const groups: Record<string, ConversationPreview[]> = {
    today: [],
    yesterday: [],
    week: [],
    month: [],
    older: [],
  };

  const now = new Date();
  const sevenDaysAgo = subDays(now, 7);
  const thirtyDaysAgo = subDays(now, 30);

  for (const conv of conversations) {
    const date = new Date(conv.last_message_at);
    if (isToday(date)) groups.today.push(conv);
    else if (isYesterday(date)) groups.yesterday.push(conv);
    else if (isAfter(date, sevenDaysAgo)) groups.week.push(conv);
    else if (isAfter(date, thirtyDaysAgo)) groups.month.push(conv);
    else groups.older.push(conv);
  }

  const result: TimeGroup[] = [];
  if (groups.today.length) result.push({ label: "Hoje", conversations: groups.today });
  if (groups.yesterday.length) result.push({ label: "Ontem", conversations: groups.yesterday });
  if (groups.week.length) result.push({ label: "Ultimos 7 dias", conversations: groups.week });
  if (groups.month.length) result.push({ label: "Ultimos 30 dias", conversations: groups.month });
  if (groups.older.length) result.push({ label: "Mais antigas", conversations: groups.older });
  return result;
}

export function ConversationSidebar({
  conversations,
  isLoading,
  activeThreadId,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
  onClose,
  className,
}: ConversationSidebarProps) {
  const groups = useMemo(() => groupByTime(conversations), [conversations]);

  return (
    <div className={cn("flex flex-col h-full bg-card border-r", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b">
        <h3 className="text-sm font-semibold">Conversas</h3>
        <button
          onClick={onClose}
          className="p-1 rounded-md hover:bg-accent text-muted-foreground"
          title="Fechar sidebar"
        >
          <PanelLeftClose className="h-4 w-4" />
        </button>
      </div>

      {/* Botao Nova Conversa */}
      <div className="p-3 border-b">
        <button
          onClick={onNewConversation}
          className="flex items-center gap-2 w-full rounded-lg border border-dashed px-3 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
        >
          <MessageSquarePlus className="h-4 w-4" />
          Nova conversa
        </button>
      </div>

      {/* Lista de conversas */}
      <ScrollArea className="flex-1">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        ) : conversations.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-8 text-center px-4">
            <MessageSquare className="h-8 w-8 text-muted-foreground/40 mb-2" />
            <p className="text-sm text-muted-foreground">Nenhuma conversa ainda</p>
            <p className="text-xs text-muted-foreground mt-1">
              Envie uma mensagem para comecar
            </p>
          </div>
        ) : (
          <div className="p-2">
            {groups.map((group) => (
              <div key={group.label} className="mb-3">
                <p className="px-2 py-1 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                  {group.label}
                </p>
                {group.conversations.map((conv) => (
                  <button
                    key={conv.thread_id}
                    onClick={() => onSelectConversation(conv.thread_id)}
                    className={cn(
                      "group flex items-center justify-between w-full rounded-lg px-2 py-2 text-left text-sm transition-colors",
                      conv.thread_id === activeThreadId
                        ? "bg-accent text-accent-foreground"
                        : "hover:bg-accent/50"
                    )}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="truncate font-medium">{conv.title}</p>
                      <p className="text-xs text-muted-foreground">
                        {format(new Date(conv.last_message_at), "HH:mm", { locale: ptBR })}
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDeleteConversation(conv.thread_id);
                      }}
                      className="hidden group-hover:flex p-1 rounded-md hover:bg-destructive/10 text-muted-foreground hover:text-destructive"
                      title="Excluir conversa"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}
      </ScrollArea>
    </div>
  );
}
