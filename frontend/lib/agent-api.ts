import { SSEEvent } from '@/types/ai-agent';
import type {
  ConversationPreview,
  ConversationMessages,
  ResumePayload,
} from '@/types/ai-agent';

/**
 * Faz parse de eventos SSE (Server-Sent Events) a partir de texto bruto.
 * Retorna um array de SSEEvent com tipo e dados parseados do JSON.
 */
export function parseSSE(raw: string): SSEEvent[] {
  const events: SSEEvent[] = [];
  const lines = raw.split('\n');
  let eventType = '';
  let dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith('event: ')) {
      eventType = line.slice(7).trim();
    } else if (line.startsWith('data: ')) {
      dataLines.push(line.slice(6));
    } else if (line === '' && eventType && dataLines.length > 0) {
      try {
        const data = dataLines.join('\n');
        events.push({ type: eventType, data: JSON.parse(data) });
      } catch {
        // Ignora eventos com JSON invalido
      }
      eventType = '';
      dataLines = [];
    }
  }

  return events;
}

/**
 * Envia uma mensagem para o agente via POST.
 * Retorna a Response do fetch para processamento de streaming SSE.
 */
export async function sendAgentMessage(
  message: string,
  threadId: string,
  accountId: string,
  resumePayload?: ResumePayload
): Promise<Response> {
  return fetch('/api/v1/agent/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      thread_id: threadId,
      account_id: accountId,
      resume_payload: resumePayload,
    }),
  });
}

const AGENT_API_BASE = '/api/v1/agent';

function getAgentHeaders(): Record<string, string> {
  return {
    'Content-Type': 'application/json',
  };
}

export async function fetchConversations(accountId: string): Promise<ConversationPreview[]> {
  const res = await fetch(
    `${AGENT_API_BASE}/conversations?account_id=${encodeURIComponent(accountId)}`,
    { headers: getAgentHeaders() },
  );
  if (!res.ok) throw new Error(`Failed to fetch conversations: ${res.status}`);
  const data = await res.json();
  return data.conversations;
}

export async function fetchConversationMessages(
  threadId: string,
  accountId: string,
): Promise<ConversationMessages> {
  const res = await fetch(
    `${AGENT_API_BASE}/conversations/${encodeURIComponent(threadId)}/messages?account_id=${encodeURIComponent(accountId)}`,
    { headers: getAgentHeaders() },
  );
  if (!res.ok) throw new Error(`Failed to fetch messages: ${res.status}`);
  return res.json();
}

export async function deleteConversation(
  threadId: string,
  accountId: string,
): Promise<void> {
  const res = await fetch(
    `${AGENT_API_BASE}/conversations/${encodeURIComponent(threadId)}?account_id=${encodeURIComponent(accountId)}`,
    { method: 'DELETE', headers: getAgentHeaders() },
  );
  if (!res.ok) throw new Error(`Failed to delete conversation: ${res.status}`);
}
