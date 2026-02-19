export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  toolResults?: ToolResultData[];
}

export interface AgentStatus {
  id: string;
  name: string;
  displayName: string;
  status: 'idle' | 'running' | 'completed' | 'error';
  progress?: { message: string; percent: number };
}

export interface ToolResultData {
  tool: string;
  data: Record<string, any>;
  agent: string;
}

export interface InterruptPayload {
  type: string;
  approval_token: string;
  details: Record<string, any>;
  thread_id: string;
}

export interface SSEEvent {
  type: string;
  data: Record<string, any>;
}

export interface ResumePayload {
  approved: boolean;
  approval_token: string;
  new_budget_override?: number;
  override_value?: number;
}

export type AgentId =
  | 'health_monitor'
  | 'performance_analyst'
  | 'creative_specialist'
  | 'audience_specialist'
  | 'forecast_scientist'
  | 'operations_manager';

export const AGENT_DISPLAY_NAMES: Record<AgentId, string> = {
  health_monitor: 'Monitor de Saude',
  performance_analyst: 'Analista de Performance',
  creative_specialist: 'Especialista em Criativos',
  audience_specialist: 'Especialista em Audiencias',
  forecast_scientist: 'Cientista de Previsao',
  operations_manager: 'Gerente de Operacoes',
};

export interface ConversationPreview {
  thread_id: string;
  title: string;
  created_at: string;
  last_message_at: string;
}

export interface ConversationMessages {
  thread_id: string;
  messages: { role: 'user' | 'assistant'; content: string; timestamp: string }[];
}
