"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import { MessageBubble } from "./message-bubble";
import { Bot } from "lucide-react";
import type { Message } from "@/types/ai-agent";

interface MessageListProps {
  messages: Message[];
  isStreaming: boolean;
  className?: string;
  onQuickActionClick?: (message: string) => void;
}

/**
 * Lista scrollavel de mensagens com auto-scroll ao receber novas mensagens.
 */
export function MessageList({
  messages,
  isStreaming,
  className,
  onQuickActionClick,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll ao adicionar mensagens ou durante streaming
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return (
      <div
        className={cn(
          "flex flex-1 flex-col items-center justify-center p-8 text-center",
          className
        )}
      >
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 mb-4">
          <Bot className="h-8 w-8 text-primary" />
        </div>
        <h2 className="text-lg font-semibold mb-2">Agente de Marketing IA</h2>
        <p className="text-sm text-muted-foreground max-w-md">
          Faca perguntas sobre suas campanhas, analises de performance,
          recomendacoes de otimizacao e muito mais.
        </p>
        <div className="flex flex-wrap gap-2 mt-6 max-w-lg justify-center">
          {[
            "Como estao minhas campanhas hoje?",
            "Quais campanhas precisam de atencao?",
            "Previsao de performance para proxima semana",
            "Recomendacoes de otimizacao",
          ].map((suggestion) => (
            <button
              key={suggestion}
              className="rounded-full border bg-card px-3 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
              onClick={() => onQuickActionClick?.(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <ScrollArea className={cn("flex-1", className)}>
      <div className="flex flex-col gap-4 py-4">
        {messages.map((message, index) => (
          <MessageBubble
            key={message.id}
            message={message}
            isStreaming={isStreaming}
            isLastMessage={index === messages.length - 1}
          />
        ))}
        <div ref={bottomRef} />
      </div>
    </ScrollArea>
  );
}
