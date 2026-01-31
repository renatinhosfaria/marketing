/**
 * Tipos do módulo AI Agent
 */

export interface AgentMessage {
  id?: string;
  role: "user" | "assistant" | "tool";
  content: string;
  created_at?: string;
  isStreaming?: boolean;
}

export interface AgentConversation {
  thread_id: string;
  title?: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface AgentConversationListResponse {
  conversations: AgentConversation[];
  total: number;
}

export interface AgentConversationHistoryResponse {
  thread_id: string;
  messages: AgentMessage[];
  message_count: number;
}

export interface AgentChatResponse {
  success: boolean;
  thread_id: string;
  response: string;
  intent?: string;
  tool_calls_count: number;
  error?: string;
}

export interface AgentAnalyzeResponse {
  success: boolean;
  response: string;
  intent?: string;
  tool_calls_count: number;
  error?: string;
}

export interface AgentSuggestion {
  id: string;
  text: string;
  category?: string;
  priority?: number;
}

export interface AgentSuggestionsResponse {
  config_id: number;
  suggestions: AgentSuggestion[];
}

// Tipos de eventos do streaming
export type AgentEventType =
  | "stream_start"
  | "text"
  | "tool_start"
  | "tool_end"
  | "done"
  | "error"
  | "node_start"
  | "node_end"
  | "intent_classified"
  | "data_gathered";

export interface AgentStreamChunk {
  type: AgentEventType;
  thread_id: string;
  timestamp?: number;
  // Campos para eventos de texto
  content?: string;
  // Campos para eventos de ferramenta
  tool?: string;
  input_preview?: string;
  output_preview?: string;
  success?: boolean;
  // Campos para eventos de nó
  node?: string;
  duration_ms?: number;
  // Campos para evento de intenção
  intent?: string;
  // Campos para evento de dados coletados
  data_counts?: DataCounts;
  // Campos para evento de erro
  error?: string;
  // Campos para evento done
  total_duration_ms?: number;
}

// Contadores de dados coletados
export interface DataCounts {
  classifications: number;
  recommendations: number;
  anomalies: number;
  forecasts: number;
}

// Status de um nó do grafo
export type NodeStatus = "pending" | "active" | "completed" | "error" | "skipped";

// Representa um nó do grafo LangGraph
export interface GraphNode {
  id: string;
  name: string;
  status: NodeStatus;
  startTime?: number;
  endTime?: number;
  duration?: number;
}

// Representa uma execução de ferramenta
export interface ToolExecution {
  id: string;
  name: string;
  status: "running" | "success" | "error";
  startTime: number;
  endTime?: number;
  duration?: number;
  inputPreview?: string;
  outputPreview?: string;
}

// Contexto coletado durante a execução
export interface DebugContext {
  intent?: string;
  classifications: number;
  recommendations: number;
  anomalies: number;
  forecasts: number;
  toolCallsCount: number;
}

// Evento de debug para o log
export interface DebugEvent {
  id: string;
  type: AgentEventType;
  timestamp: number;
  data: Record<string, unknown>;
}

// Estado completo do debug
export interface DebugState {
  isEnabled: boolean;
  isActive: boolean;
  nodes: GraphNode[];
  activeNode: string | null;
  tools: ToolExecution[];
  context: DebugContext;
  events: DebugEvent[];
  startTime: number | null;
  totalDuration: number | null;
}

export interface AgentStatus {
  status: "online" | "offline" | "error";
  llm_provider: string;
  model: string;
  version: string;
}

export interface ChatState {
  messages: AgentMessage[];
  threadId: string | null;
  isLoading: boolean;
  isStreaming: boolean;
  error: string | null;
}
