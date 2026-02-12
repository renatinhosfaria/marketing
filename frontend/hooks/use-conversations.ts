"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchConversations,
  deleteConversation,
} from "@/lib/agent-api";
import type { ConversationPreview } from "@/types/ai-agent";

export function useConversations(accountId: string) {
  const queryClient = useQueryClient();

  const {
    data: conversations = [],
    isLoading,
    error,
  } = useQuery<ConversationPreview[]>({
    queryKey: ["agent-conversations", accountId],
    queryFn: () => fetchConversations(accountId),
    enabled: !!accountId,
    refetchInterval: 30_000,
  });

  const deleteMutation = useMutation({
    mutationFn: (threadId: string) => deleteConversation(threadId, accountId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["agent-conversations", accountId] });
    },
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["agent-conversations", accountId] });
  };

  return {
    conversations,
    isLoading,
    error,
    deleteConversation: deleteMutation.mutate,
    isDeleting: deleteMutation.isPending,
    invalidate,
  };
}
