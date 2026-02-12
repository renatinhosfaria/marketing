"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { parseSSE, sendAgentMessage, fetchConversationMessages } from "@/lib/agent-api";
import type {
  InterruptPayload,
  Message,
  ToolResultData,
} from "@/types/ai-agent";

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

export function useAgentChat(accountId: string) {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Estado principal
  const [messages, setMessages] = useState<Message[]>([]);
  const [toolResults, setToolResults] = useState<ToolResultData[]>([]);
  const [interrupt, setInterrupt] = useState<InterruptPayload | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingMessages, setIsLoadingMessages] = useState(false);

  // Thread ID persistido na URL (lazy init para nao gerar ID novo a cada render)
  const [initialThreadId] = useState(() => searchParams.get("thread") || generateId());
  const threadId = searchParams.get("thread") || initialThreadId;
  const threadIdRef = useRef(threadId);

  // Estabilizar searchParams como string para evitar re-renders em useCallback
  const searchParamsString = searchParams.toString();

  // Persistir threadId na URL na primeira carga
  useEffect(() => {
    if (!searchParams.get("thread")) {
      const params = new URLSearchParams(searchParamsString);
      params.set("thread", threadIdRef.current);
      router.replace(`?${params.toString()}`, { scroll: false });
    }
  }, [router, searchParams, searchParamsString]);

  // Atualizar ref quando threadId muda
  useEffect(() => {
    threadIdRef.current = threadId;
  }, [threadId]);

  const loadMessages = useCallback(
    async (targetThreadId: string) => {
      setIsLoadingMessages(true);
      setMessages([]);
      try {
        const data = await fetchConversationMessages(targetThreadId, accountId);
        const loaded: Message[] = data.messages.map((m, i) => ({
          id: `loaded-${i}-${Date.now()}`,
          role: m.role as 'user' | 'assistant',
          content: m.content,
          timestamp: m.timestamp ? new Date(m.timestamp) : new Date(),
        }));
        setMessages(loaded);
      } catch {
        // Thread nova sem mensagens — nao e erro
      } finally {
        setIsLoadingMessages(false);
      }
    },
    [accountId]
  );

  const switchThread = useCallback(
    async (newThreadId: string) => {
      threadIdRef.current = newThreadId;
      setInterrupt(null);
      const params = new URLSearchParams(searchParamsString);
      params.set("thread", newThreadId);
      router.replace(`?${params.toString()}`, { scroll: false });
      await loadMessages(newThreadId);
    },
    [router, searchParamsString, loadMessages]
  );

  const startNewConversation = useCallback(() => {
    const newId = generateId();
    threadIdRef.current = newId;
    setMessages([]);
    setInterrupt(null);
    const params = new URLSearchParams(searchParamsString);
    params.set("thread", newId);
    router.replace(`?${params.toString()}`, { scroll: false });
  }, [router, searchParamsString]);

  /**
   * Processa o stream SSE de resposta do agente.
   */
  const processStream = useCallback(
    async (response: Response) => {
      if (!response.ok) {
        const errorText = await response.text();
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            role: "assistant",
            content: `Erro ao processar mensagem: ${response.status} - ${errorText}`,
            timestamp: new Date(),
          },
        ]);
        setIsStreaming(false);
        return;
      }

      const reader = response.body?.getReader();
      if (!reader) {
        setIsStreaming(false);
        return;
      }

      const decoder = new TextDecoder();
      let buffer = "";
      let assistantMessageId = generateId();
      let assistantContent = "";
      let currentToolResults: ToolResultData[] = [];

      // Criar mensagem do assistente vazia para streaming
      setMessages((prev) => [
        ...prev,
        {
          id: assistantMessageId,
          role: "assistant",
          content: "",
          timestamp: new Date(),
        },
      ]);

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = parseSSE(buffer);

          // Limpar buffer apos eventos processados
          const lastNewlineIndex = buffer.lastIndexOf("\n\n");
          if (lastNewlineIndex !== -1) {
            buffer = buffer.slice(lastNewlineIndex + 2);
          }

          for (const event of events) {
            switch (event.type) {
              case "message": {
                const content = event.data.content || event.data.text || "";
                assistantContent += content;
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? { ...m, content: assistantContent }
                      : m
                  )
                );
                break;
              }

              case "tool_result": {
                const toolResult: ToolResultData = {
                  tool: event.data.tool || "",
                  data: event.data.result || event.data.data || {},
                  agent: event.data.agent || "",
                };
                currentToolResults.push(toolResult);
                setToolResults((prev) => [...prev, toolResult]);
                // Atualizar mensagem do assistente com tool results
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMessageId
                      ? { ...m, toolResults: [...currentToolResults] }
                      : m
                  )
                );
                break;
              }

              case "interrupt": {
                const interruptPayload: InterruptPayload = {
                  type: event.data.type || "",
                  approval_token: event.data.approval_token || "",
                  details: event.data.details || {},
                  thread_id: event.data.thread_id || threadIdRef.current,
                };
                setInterrupt(interruptPayload);
                break;
              }

              case "done": {
                setIsStreaming(false);
                break;
              }
            }
          }
        }
      } catch (error) {
        console.error("Erro ao processar stream:", error);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantMessageId
              ? {
                  ...m,
                  content:
                    assistantContent ||
                    "Erro na conexao com o servidor. Tente novamente.",
                }
              : m
          )
        );
      } finally {
        setIsStreaming(false);
      }
    },
    []
  );

  /**
   * Envia uma mensagem do usuario para o agente.
   */
  const sendMessage = useCallback(
    async (content: string) => {
      if (!content.trim() || isStreaming) return;

      // Adicionar mensagem do usuario
      const userMessage: Message = {
        id: generateId(),
        role: "user",
        content: content.trim(),
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setIsStreaming(true);
      setInterrupt(null);

      try {
        const response = await sendAgentMessage(
          content.trim(),
          threadIdRef.current,
          accountId
        );
        await processStream(response);
      } catch (error) {
        console.error("Erro ao enviar mensagem:", error);
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            role: "assistant",
            content: "Nao foi possivel conectar ao servidor. Verifique sua conexao.",
            timestamp: new Date(),
          },
        ]);
        setIsStreaming(false);
      }
    },
    [accountId, isStreaming, processStream]
  );

  /**
   * Retoma a execucao com aprovacao ou rejeicao.
   */
  const resumeWithApproval = useCallback(
    async (approved: boolean, overrideValue?: string) => {
      if (!interrupt) return;

      setIsStreaming(true);
      const resumePayload = {
        approval_token: interrupt.approval_token,
        approved,
        override_value: overrideValue,
      };
      setInterrupt(null);

      // Adicionar mensagem do usuario indicando a decisao
      const decisionMessage: Message = {
        id: generateId(),
        role: "user",
        content: approved
          ? `Aprovado${overrideValue ? `: ${overrideValue}` : ""}`
          : "Rejeitado",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, decisionMessage]);

      try {
        const response = await sendAgentMessage(
          "",
          threadIdRef.current,
          accountId,
          resumePayload
        );
        await processStream(response);
      } catch (error) {
        console.error("Erro ao retomar execucao:", error);
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            role: "assistant",
            content: "Erro ao processar sua decisao. Tente novamente.",
            timestamp: new Date(),
          },
        ]);
        setIsStreaming(false);
      }
    },
    [accountId, interrupt, processStream]
  );

  return {
    messages,
    toolResults,
    interrupt,
    isStreaming,
    isLoadingMessages,
    threadId: threadIdRef.current,
    sendMessage,
    resumeWithApproval,
    switchThread,
    startNewConversation,
    loadMessages,
  };
}
