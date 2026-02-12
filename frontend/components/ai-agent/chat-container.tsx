"use client";

import { Suspense, useState } from "react";
import { cn } from "@/lib/utils";
import { useAgentChat } from "@/hooks/use-agent-chat";
import { useConversations } from "@/hooks/use-conversations";
import { MessageList } from "./message-list";
import { MessageInput } from "./message-input";
import { ConversationSidebar } from "./conversation-history";
import { ApprovalWidget } from "./approval-widget";
import { PanelLeftOpen } from "lucide-react";

interface ChatContainerProps {
  accountId: string;
  className?: string;
}

function ChatContainerInner({ accountId, className }: ChatContainerProps) {
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    if (typeof window === "undefined") return true;
    return window.innerWidth >= 1024;
  });

  const {
    messages,
    interrupt,
    isStreaming,
    isLoadingMessages,
    threadId,
    sendMessage,
    resumeWithApproval,
    switchThread,
    startNewConversation,
  } = useAgentChat(accountId);

  const {
    conversations,
    isLoading: isLoadingConversations,
    deleteConversation,
    invalidate,
  } = useConversations(accountId);

  const handleSend = async (content: string) => {
    await sendMessage(content);
    invalidate();
  };

  const handleNewConversation = () => {
    startNewConversation();
  };

  const handleDeleteConversation = (threadIdToDelete: string) => {
    deleteConversation(threadIdToDelete);
    if (threadIdToDelete === threadId) {
      startNewConversation();
    }
  };

  return (
    <div className={cn("flex h-[calc(100vh-8rem)]", className)}>
      {/* Sidebar de conversas */}
      {sidebarOpen && (
        <>
          {/* Overlay mobile */}
          <div
            className="fixed inset-0 bg-black/50 z-40 lg:hidden"
            onClick={() => setSidebarOpen(false)}
          />
          <div
            className={cn(
              "w-72 flex-shrink-0 z-50",
              "fixed inset-y-0 left-0 lg:relative lg:inset-auto"
            )}
          >
            <ConversationSidebar
              conversations={conversations}
              isLoading={isLoadingConversations}
              activeThreadId={threadId}
              onSelectConversation={(id) => {
                switchThread(id);
                if (window.innerWidth < 1024) setSidebarOpen(false);
              }}
              onNewConversation={() => {
                handleNewConversation();
                if (window.innerWidth < 1024) setSidebarOpen(false);
              }}
              onDeleteConversation={handleDeleteConversation}
              onClose={() => setSidebarOpen(false)}
              className="h-full"
            />
          </div>
        </>
      )}

      {/* Area de chat */}
      <div className="flex-1 flex flex-col rounded-xl border bg-card overflow-hidden">
        {/* Toggle sidebar (quando fechada) */}
        {!sidebarOpen && (
          <div className="flex items-center px-3 py-2 border-b">
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-1 rounded-md hover:bg-accent text-muted-foreground"
              title="Abrir conversas"
            >
              <PanelLeftOpen className="h-4 w-4" />
            </button>
          </div>
        )}

        <MessageList
          messages={messages}
          isStreaming={isStreaming}
          className="flex-1 overflow-hidden"
          onQuickActionClick={handleSend}
        />

        {interrupt && (
          <ApprovalWidget
            interrupt={interrupt}
            onApprove={(overrideValue) => resumeWithApproval(true, overrideValue)}
            onReject={() => resumeWithApproval(false)}
            disabled={isStreaming}
          />
        )}

        <MessageInput onSend={handleSend} disabled={isStreaming || isLoadingMessages} />
      </div>
    </div>
  );
}

export function ChatContainer(props: ChatContainerProps) {
  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-[calc(100vh-8rem)]">
          <div className="animate-pulse text-muted-foreground">
            Carregando chat...
          </div>
        </div>
      }
    >
      <ChatContainerInner {...props} />
    </Suspense>
  );
}
