"use client";

/**
 * Interface principal de chat do agente
 */

import { useEffect, useRef, useState } from "react";
import { Bot, Loader2, Plus, Send, Square, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import type { AgentMessage, AgentSuggestion } from "@/types/ai-agent";
import { MessageBubble } from "./message-bubble";
import { QuickActions } from "./quick-actions";

interface ChatInterfaceProps {
  messages: AgentMessage[];
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
  suggestions: AgentSuggestion[];
  isSuggestionsLoading?: boolean;
  onSendMessage: (message: string) => void;
  onCancelStream: () => void;
  onClearConversation: () => void;
  onNewConversation: () => void;
  className?: string;
}

export function ChatInterface({
  messages,
  isLoading,
  isStreaming,
  error,
  suggestions,
  isSuggestionsLoading = false,
  onSendMessage,
  onCancelStream,
  onClearConversation,
  onNewConversation,
  className,
}: ChatInterfaceProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        200
      )}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading || isStreaming) return;
    onSendMessage(input.trim());
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className={cn("flex flex-col h-full bg-background", className)}>
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-primary" />
          <h2 className="font-semibold">Assistente de Tráfego Pago</h2>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onNewConversation}
            title="Nova conversa"
          >
            <Plus className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={onClearConversation}
            title="Limpar conversa"
          >
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <ScrollArea ref={scrollRef} className="flex-1 p-4">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center p-8">
            <Bot className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="font-semibold text-lg mb-2">
              Assistente de Tráfego Pago
            </h3>
            <p className="text-muted-foreground text-sm max-w-md">
              Faça perguntas sobre campanhas, recomendações e previsões com base
              nos seus dados de Facebook Ads.
            </p>
            <div className="mt-6 w-full max-w-md">
              <QuickActions
                suggestions={suggestions}
                isLoading={isSuggestionsLoading}
                onSelect={(text) => onSendMessage(text)}
              />
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
          </div>
        )}

        {error && (
          <div className="mt-4 p-3 rounded-lg bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}
      </ScrollArea>

      <form onSubmit={handleSubmit} className="p-4 border-t">
        <div className="flex items-end gap-2">
          <Textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Pergunte sobre suas campanhas..."
            className="min-h-[44px] max-h-[200px] resize-none"
            disabled={isLoading || isStreaming}
            rows={1}
          />
          {isStreaming ? (
            <Button
              type="button"
              size="icon"
              variant="destructive"
              onClick={onCancelStream}
              title="Parar"
            >
              <Square className="h-4 w-4" />
            </Button>
          ) : (
            <Button
              type="submit"
              size="icon"
              disabled={!input.trim() || isLoading}
              title="Enviar"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}
