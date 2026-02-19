"use client";

import { cn } from "@/lib/utils";
import { User, Bot } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";
import { StreamingText } from "./streaming-text";
import type { Message } from "@/types/ai-agent";

interface MessageBubbleProps {
  message: Message;
  isStreaming?: boolean;
  isLastMessage?: boolean;
}

/**
 * Bolha de mensagem com estilos diferenciados para user e assistant.
 * User: alinhado a direita, bg-blue-600, texto branco.
 * Assistant: alinhado a esquerda, bg-gray-100, suporta markdown.
 */
export function MessageBubble({
  message,
  isStreaming = false,
  isLastMessage = false,
}: MessageBubbleProps) {
  const isUser = message.role === "user";
  const showStreamingCursor = isStreaming && isLastMessage && !isUser;

  return (
    <div
      className={cn(
        "flex gap-3 px-4",
        isUser ? "justify-end" : "justify-start"
      )}
    >
      {/* Avatar do assistente */}
      {!isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 mt-1">
          <Bot className="h-4 w-4 text-primary" />
        </div>
      )}

      {/* Conteudo da mensagem */}
      <div
        className={cn(
          "max-w-[75%] rounded-2xl px-4 py-2.5",
          isUser
            ? "bg-blue-600 text-white"
            : "bg-muted text-foreground"
        )}
      >
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : showStreamingCursor && !message.content ? (
          <StreamingText content="" isStreaming={true} className="text-sm" />
        ) : showStreamingCursor ? (
          <StreamingText
            content={message.content}
            isStreaming={true}
            className="text-sm"
          />
        ) : (
          <div className="prose prose-sm max-w-none dark:prose-invert prose-p:my-1 prose-ul:my-1 prose-ol:my-1 prose-li:my-0.5 prose-headings:my-2 prose-pre:my-2">
            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
              {message.content}
            </ReactMarkdown>
          </div>
        )}

        {/* Tool results badges */}
        {message.toolResults && message.toolResults.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2 pt-2 border-t border-border/30">
            {message.toolResults.map((tr, idx) => (
              <span
                key={idx}
                className={cn(
                  "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
                  isUser
                    ? "bg-blue-500/30 text-blue-100"
                    : "bg-primary/10 text-primary"
                )}
              >
                {tr.agent}: {tr.tool}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Avatar do usuario */}
      {isUser && (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-600 mt-1">
          <User className="h-4 w-4 text-white" />
        </div>
      )}
    </div>
  );
}
