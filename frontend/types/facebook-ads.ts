/**
 * Tipos TypeScript para o módulo Facebook Ads no Frontend
 */

// ==============================
// TIPOS DE STATUS
// ==============================

export type FacebookEntityStatus = "ACTIVE" | "PAUSED" | "DELETED" | "ARCHIVED";

export type FacebookEffectiveStatus =
  | "ACTIVE"
  | "PAUSED"
  | "DELETED"
  | "ARCHIVED"
  | "IN_PROCESS"
  | "WITH_ISSUES"
  | "PENDING_REVIEW"
  | "DISAPPROVED"
  | "CAMPAIGN_PAUSED"
  | "ADSET_PAUSED";

export type SyncStatus = "pending" | "running" | "completed" | "failed";

export type EntityType = "account" | "campaign" | "adset" | "ad";

// ==============================
// INTERFACES DE CONFIGURAÇÃO
// ==============================

export interface FacebookAdsConfig {
  id: number;
  accountId: string;
  accountName: string;
  appId?: string;
  isActive: boolean;
  syncEnabled: boolean;
  syncFrequencyMinutes: number;
  lastSyncAt: string | null;
  tokenExpiresAt: string | null;
  tokenExpiringSoon?: boolean | null;
  createdBy?: number;
  createdAt: string;
  updatedAt: string;
}

// Alias para compatibilidade
export type FacebookConfig = FacebookAdsConfig;

export interface AdAccount {
  id: string;
  name: string;
  currency: string;
  accountStatus: number;
}

// ==============================
// INTERFACES DE CAMPANHA
// ==============================

export interface Campaign {
  id: number;
  configId: number;
  campaignId: string;
  name: string;
  objective: string | null;
  status: FacebookEntityStatus;
  effectiveStatus: FacebookEffectiveStatus | null;
  dailyBudget: string | null;
  lifetimeBudget: string | null;
  budgetRemaining: string | null;
  startTime: string | null;
  stopTime: string | null;
  syncedAt: string;
}

export interface CampaignWithMetrics extends Campaign {
  spend: number;
  impressions: number;
  clicks: number;
  leads: number;
  ctr: number;
  cpc: number;
  cpl: number | null;
  cpp: number | null;
  uniqueCtr: number | null;
  costPerUniqueClick: number | null;
  purchaseRoas: number | null;
}

// ==============================
// INTERFACES DE AD SET
// ==============================

export interface AdSet {
  id: number;
  configId: number;
  campaignId: string;
  adsetId: string;
  name: string;
  status: FacebookEntityStatus;
  effectiveStatus: FacebookEffectiveStatus | null;
  dailyBudget: string | null;
  lifetimeBudget: string | null;
  optimizationGoal: string | null;
  syncedAt: string;
}

export interface AdSetWithMetrics extends AdSet {
  spend: number;
  impressions: number;
  clicks: number;
  leads: number;
  ctr: number;
  cpc: number;
  cpl: number | null;
}

// ==============================
// INTERFACES DE AD
// ==============================

export interface Ad {
  id: number;
  configId: number;
  campaignId: string;
  adsetId: string;
  adId: string;
  name: string;
  status: FacebookEntityStatus;
  effectiveStatus: FacebookEffectiveStatus | null;
  previewShareableLink: string | null;
  syncedAt: string;
}

export interface AdWithMetrics extends Ad {
  spend: number;
  impressions: number;
  clicks: number;
  leads: number;
  ctr: number;
  cpc: number;
  cpl: number | null;
}

// ==============================
// INTERFACES DE INSIGHTS
// ==============================

export interface InsightsSummary {
  totalSpend: number;
  totalImpressions: number;
  totalReach: number;
  totalClicks: number;
  totalLeads: number;
  avgCtr: number;
  avgCpc: number;
  avgCpl: number;
  avgCpm: number;
  totalUniqueClicks: number;
  avgCpp: number;
  avgUniqueCtr: number;
  avgPurchaseRoas: number | null;
  comparison?: {
    spendChange: number;
    impressionsChange: number;
    clicksChange: number;
    leadsChange: number;
    ctrChange: number;
    cpcChange: number;
    cplChange: number;
    cpmChange: number;
  };
}

export interface TimeSeriesDataPoint {
  date: string;
  spend: number;
  impressions: number;
  clicks: number;
  leads: number;
  ctr: number;
  cpc: number;
  cpl: number | null;
  uniqueClicks: number;
  uniqueCtr: number | null;
  videoPlays: number | null;
  videoThruplay: number | null;
}

// ==============================
// INTERFACES DE MÉTRICAS EXPANDIDAS
// ==============================

export interface VideoFunnel {
  videoPlays: number;
  video15sWatched: number;
  videoP25Watched: number;
  videoP50Watched: number;
  videoP75Watched: number;
  videoP95Watched: number;
  video30sWatched: number;
  videoP100Watched: number;
  videoThruplay: number;
  videoAvgTime: number;
}

export interface QualityDiagnostics {
  adId: string;
  adName: string | null;
  qualityRanking: string | null;
  engagementRateRanking: string | null;
  conversionRateRanking: string | null;
  impressions: number;
  spend: number;
}

export type QualityRanking =
  | "BELOW_AVERAGE_10"
  | "BELOW_AVERAGE_20"
  | "BELOW_AVERAGE_35"
  | "AVERAGE"
  | "ABOVE_AVERAGE";

export interface BreakdownInsight {
  breakdownType: string;
  breakdownValue: string;
  spend: number;
  impressions: number;
  reach: number;
  clicks: number;
  leads: number;
  conversions: number;
  ctr: number;
  cpc: number;
  cpl: number | null;
}

export type BreakdownType =
  | "platform_position"
  | "region"
  | "impression_device"
  | "frequency_value"
  | "hourly"
  | "age"
  | "gender"
  | "country";

// ==============================
// INTERFACES DE SYNC
// ==============================

export interface SyncProgress {
  stage: string;
  campaignsSynced: number;
  adsetsSynced: number;
  adsSynced: number;
  insightsSynced: number;
}

export interface SyncHistoryItem {
  id: number;
  configId: number;
  status: SyncStatus;
  syncType: string;
  startedAt: string;
  completedAt: string | null;
  durationMs: number | null;
  entitiesSynced: number;
  campaignsSynced: number;
  adsetsSynced: number;
  adsSynced: number;
  insightsSynced: number;
  errorMessage: string | null;
}

export interface SyncStatusResponse {
  configId: number;
  accountName: string | null;
  isRunning: boolean;
  progress: SyncProgress | null;
  lastSync: SyncHistoryItem | null;
  lastSyncAt: string | null;
  nextSyncAt: string | null;
  syncEnabled: boolean;
  syncFrequencyMinutes: number;
}

// ==============================
// TIPOS DE FILTRO
// ==============================

export type DatePreset =
  | "today"
  | "yesterday"
  | "last_7d"
  | "last_14d"
  | "last_30d"
  | "this_month"
  | "last_month"
  | "this_year"
  | "last_year";

export type InsightsTableSource = "today" | "history" | "both";

export interface DateRange {
  start: Date;
  end: Date;
}

// ==============================
// INTERFACES DE RESPOSTA DA API
// ==============================

export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  message?: string;
}

export interface PaginatedResponse<T> {
  success: boolean;
  data: T[];
  pagination: {
    total: number;
    limit: number;
    offset: number;
    hasMore: boolean;
  };
}
