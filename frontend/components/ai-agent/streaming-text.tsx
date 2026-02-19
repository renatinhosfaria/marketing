"use client";

import { cn } from "@/lib/utils";

interface StreamingTextProps {
  content: string;
  isStreaming: boolean;
  className?: string;
}

/**
 * Exibe texto com cursor piscante durante streaming.
 */
export function StreamingText({
  content,
  isStreaming,
  className,
}: StreamingTextProps) {
  return (
    <span className={cn("whitespace-pre-wrap", className)}>
      {content}
      {isStreaming && (
        <span className="inline-block w-2 h-4 ml-0.5 bg-foreground/70 animate-pulse rounded-sm" />
      )}
    </span>
  );
}
