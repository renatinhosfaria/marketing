/**
 * Tipos para o módulo ML Ads
 */

// Enums
export type CampaignTier = 'HIGH_PERFORMER' | 'MODERATE' | 'LOW' | 'UNDERPERFORMER';

export type RecommendationType =
  | 'BUDGET_INCREASE'
  | 'BUDGET_DECREASE'
  | 'PAUSE_CAMPAIGN'
  | 'SCALE_UP'
  | 'CREATIVE_REFRESH'
  | 'AUDIENCE_REVIEW'
  | 'REACTIVATE'
  | 'OPTIMIZE_SCHEDULE';

export type AnomalySeverity = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';

// Classificação de Campanha
export interface CampaignClassification {
  id: number;
  config_id: number;
  campaign_id: string;
  campaign_name?: string;
  tier: CampaignTier;
  confidence_score: number;
  metrics_snapshot: {
    cpl_7d: number;
    ctr_7d: number;
    leads_7d: number;
    spend_7d: number;
    cpl_trend: number;
    avg_cpl_reference: number;
  };
  feature_importances?: Record<string, number>;
  previous_tier?: CampaignTier;
  tier_change_direction?: string;
  classified_at: string;
}

export interface ClassificationSummary {
  config_id: number;
  total_campaigns: number;
  by_tier: Record<CampaignTier, number>;
  average_confidence: number;
  recent_changes: number;
  last_classification_at: string | null;
}

// Recomendação
export interface Recommendation {
  id: number;
  config_id: number;
  entity_type: string;
  entity_id: string;
  recommendation_type: RecommendationType;
  priority: number;
  title: string;
  description: string;
  suggested_action?: {
    field?: string;
    change_type?: string;
    change_value?: number;
    expected_impact?: Record<string, unknown>;
    action?: string;
    recommendations?: string[];
  };
  confidence_score: number;
  reasoning?: Record<string, unknown>;
  is_active: boolean;
  was_applied: boolean;
  dismissed: boolean;
  created_at: string;
  expires_at?: string;
}

export interface RecommendationSummary {
  config_id: number;
  total: number;
  by_type: Record<RecommendationType, number>;
  by_priority: Record<string, number>;
  high_priority_count: number;
}

// Anomalia
export interface Anomaly {
  id: number;
  config_id: number;
  entity_type: string;
  entity_id: string;
  anomaly_type: string;
  metric_name: string;
  observed_value: number;
  expected_value: number;
  deviation_score: number;
  severity: AnomalySeverity;
  is_acknowledged: boolean;
  acknowledged_by?: number;
  resolution_notes?: string;
  anomaly_date: string;
  detected_at: string;
}

export interface AnomalySummary {
  config_id: number;
  total: number;
  by_severity: Record<AnomalySeverity, number>;
  unacknowledged: number;
  last_detected_at: string | null;
}

// Health
export interface MLHealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  environment?: string;
  checks?: {
    database: { status: string; type: string };
    redis: { status: string; url?: string };
  };
}

// Cores para os tiers
export const TIER_COLORS: Record<CampaignTier, { bg: string; text: string; border: string }> = {
  HIGH_PERFORMER: { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-500' },
  MODERATE: { bg: 'bg-blue-100', text: 'text-blue-800', border: 'border-blue-500' },
  LOW: { bg: 'bg-yellow-100', text: 'text-yellow-800', border: 'border-yellow-500' },
  UNDERPERFORMER: { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-500' },
};

// Labels para os tiers
export const TIER_LABELS: Record<CampaignTier, string> = {
  HIGH_PERFORMER: 'Alta Performance',
  MODERATE: 'Moderado',
  LOW: 'Baixo',
  UNDERPERFORMER: 'Muito Baixo',
};

// Labels para tipos de recomendação
export const RECOMMENDATION_TYPE_LABELS: Record<RecommendationType, string> = {
  BUDGET_INCREASE: 'Aumentar Budget',
  BUDGET_DECREASE: 'Reduzir Budget',
  PAUSE_CAMPAIGN: 'Pausar Campanha',
  SCALE_UP: 'Escalar',
  CREATIVE_REFRESH: 'Renovar Criativos',
  AUDIENCE_REVIEW: 'Revisar Audiência',
  REACTIVATE: 'Reativar',
  OPTIMIZE_SCHEDULE: 'Otimizar Horários',
};

// Cores para severidade
export const SEVERITY_COLORS: Record<AnomalySeverity, { bg: string; text: string }> = {
  LOW: { bg: 'bg-gray-100', text: 'text-gray-800' },
  MEDIUM: { bg: 'bg-yellow-100', text: 'text-yellow-800' },
  HIGH: { bg: 'bg-orange-100', text: 'text-orange-800' },
  CRITICAL: { bg: 'bg-red-100', text: 'text-red-800' },
};
