"use client";

/**
 * Hooks para o módulo Facebook Ads
 * Story 2.6 / 3.1: Hooks para sincronização e dados
 *
 * As chamadas são feitas diretamente ao microserviço famachat-ml (Python/FastAPI).
 * A URL base é /api/v1/facebook-ads.
 */

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch, apiFetchJson } from "@/lib/api";
import type {
  Ad,
  AdSet,
  AdSetWithMetrics,
  AdWithMetrics,
  Campaign,
  CampaignWithMetrics,
  DatePreset,
  FacebookAdsConfig,
  InsightsTableSource,
  InsightsSummary,
  PaginatedResponse,
  SyncHistoryItem,
  SyncStatusResponse,
  TimeSeriesDataPoint,
} from "@/types/facebook-ads";

// ==============================
// BASE URL DO MÓDULO FACEBOOK ADS
// ==============================

const FB_ADS_API = "/api/v1/facebook-ads";

// ==============================
// FETCH HELPERS USANDO apiFetch
// ==============================

/**
 * GET autenticado que retorna Response (para hooks que fazem parse manual)
 */
async function fbAdsFetch(path: string): Promise<Response> {
  const url = `${FB_ADS_API}${path}`;
  return apiFetch(url, { method: "GET" });
}

/**
 * Requisições com body (POST, PUT, DELETE, PATCH)
 */
async function fbAdsRequest(
  method: string,
  path: string,
  body?: unknown,
): Promise<unknown> {
  const url = `${FB_ADS_API}${path}`;

  const res = await apiFetch(url, {
    method,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const errorText = await res.text().catch(() => "Erro desconhecido");
    throw new Error(`${res.status}: ${errorText}`);
  }

  const contentType = res.headers.get("content-type");
  if (contentType && contentType.includes("application/json")) {
    return res.json();
  }
  return res;
}

// ==============================
// CONFIG HOOKS
// ==============================

/**
 * Hook para listar configurações de contas
 */
export function useFacebookAdsConfigs() {
  return useQuery<FacebookAdsConfig[]>({
    queryKey: ["fb-ads-config"],
    queryFn: async () => {
      const response = await fbAdsFetch("/config");
      if (!response.ok) throw new Error("Erro ao carregar configurações");
      const data = await response.json();
      return data.data || [];
    },
  });
}

/**
 * Hook para obter uma configuração específica
 */
export function useFacebookAdsConfig(configId: number | null) {
  return useQuery<FacebookAdsConfig>({
    queryKey: ["fb-ads-config", configId],
    queryFn: async () => {
      const response = await fbAdsFetch(`/config/${configId}`);
      if (!response.ok) throw new Error("Erro ao carregar configuração");
      const data = await response.json();
      return data.data;
    },
    enabled: !!configId,
  });
}

// ==============================
// SYNC HOOKS
// ==============================

/**
 * Hook para status de sincronização
 */
export function useSyncStatus(configId: number | null) {
  return useQuery<SyncStatusResponse>({
    queryKey: ["fb-ads-sync-status", configId],
    queryFn: async () => {
      const response = await fbAdsFetch(`/sync/${configId}/status`);
      if (!response.ok) throw new Error("Erro ao carregar status de sync");
      const data = await response.json();
      return data.data;
    },
    enabled: !!configId,
    refetchInterval: (query) => {
      // Refetch a cada 5 segundos se estiver rodando
      return query.state.data?.isRunning ? 5000 : false;
    },
  });
}

/**
 * Hook para histórico de sincronizações
 */
export function useSyncHistory(configId?: number, limit = 10) {
  return useQuery<PaginatedResponse<SyncHistoryItem>>({
    queryKey: ["fb-ads-sync-history", configId, limit],
    queryFn: async () => {
      const params = new URLSearchParams();
      if (configId) params.set("configId", String(configId));
      params.set("limit", String(limit));

      const response = await fbAdsFetch(
        `/sync/${configId}/history?${params}`,
      );
      if (!response.ok) throw new Error("Erro ao carregar histórico de sync");
      return response.json();
    },
  });
}

/**
 * Hook para disparar sincronização manual
 */
export function useTriggerSync() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: {
      configId: number;
      syncType?: "full" | "incremental";
      dateRangeStart?: string;
      dateRangeEnd?: string;
    }) => {
      const { configId, ...body } = params;
      return fbAdsRequest("POST", `/sync/${configId}`, body);
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-sync-status", variables.configId],
      });
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-sync-history"],
      });
    },
  });
}

/**
 * Hook para sincronizar APENAS dados de hoje
 */
export function useTriggerTodaySync() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (configId: number) => {
      return fbAdsRequest("POST", `/sync/${configId}/today`, {});
    },
    onSuccess: (_, configId) => {
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-sync-status", configId],
      });
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-insights"],
      });
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-campaigns"],
      });
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-adsets"],
      });
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-ads"],
      });
    },
  });
}

/**
 * Hook para cancelar sincronização
 */
export function useCancelSync() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (configId: number) => {
      return fbAdsRequest("POST", `/sync/${configId}/cancel`, {});
    },
    onSuccess: (_, configId) => {
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-sync-status", configId],
      });
    },
  });
}

// ==============================
// INSIGHTS HOOKS
// ==============================

/**
 * Hook para resumo de métricas (KPIs)
 */
export function useInsightsSummary(params: {
  configId?: number;
  datePreset?: DatePreset;
  dateStart?: string;
  dateEnd?: string;
  insightsTable?: InsightsTableSource;
}) {
  const queryParams = new URLSearchParams();
  if (params.configId) queryParams.set("configId", String(params.configId));
  if (params.datePreset) queryParams.set("datePreset", params.datePreset);
  if (params.dateStart) queryParams.set("dateFrom", params.dateStart);
  if (params.dateEnd) queryParams.set("dateTo", params.dateEnd);
  // insightsTable não é suportado pelo Python — sempre usa history

  return useQuery<InsightsSummary>({
    queryKey: ["fb-ads-insights-summary", params],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/insights/summary?${queryParams}`,
      );
      if (!response.ok) throw new Error("Erro ao carregar resumo de insights");
      const data = await response.json();

      // Mapear resposta da API para o formato esperado pelo frontend
      const apiData = data.data;
      if (!apiData?.metrics) {
        return {
          totalSpend: 0,
          totalImpressions: 0,
          totalReach: 0,
          totalClicks: 0,
          totalLeads: 0,
          avgCtr: 0,
          avgCpc: 0,
          avgCpl: 0,
        };
      }

      return {
        totalSpend: apiData.metrics.spend || 0,
        totalImpressions: apiData.metrics.impressions || 0,
        totalReach: apiData.metrics.reach || 0,
        totalClicks: apiData.metrics.clicks || 0,
        totalLeads: apiData.metrics.leads || 0,
        avgCtr: apiData.metrics.ctr || 0,
        avgCpc: apiData.metrics.cpc || 0,
        avgCpl: apiData.metrics.cpl || 0,
        comparison: apiData.comparison
          ? {
              spendChange: apiData.comparison.spend || 0,
              impressionsChange: apiData.comparison.impressions || 0,
              clicksChange: apiData.comparison.clicks || 0,
              leadsChange: apiData.comparison.leads || 0,
              ctrChange: apiData.comparison.ctr || 0,
              cpcChange: apiData.comparison.cpc || 0,
              cplChange: apiData.comparison.cpl || 0,
            }
          : undefined,
      };
    },
    enabled: !!params.configId,
  });
}

/**
 * Hook para insights detalhados (série temporal)
 */
export function useInsightsTimeSeries(params: {
  configId?: number;
  datePreset?: DatePreset;
  dateStart?: string;
  dateEnd?: string;
  entityType?: string;
  entityId?: string;
}) {
  const queryParams = new URLSearchParams();
  if (params.configId) queryParams.set("configId", String(params.configId));
  if (params.datePreset) queryParams.set("datePreset", params.datePreset);
  if (params.dateStart) queryParams.set("dateFrom", params.dateStart);
  if (params.dateEnd) queryParams.set("dateTo", params.dateEnd);

  return useQuery<TimeSeriesDataPoint[]>({
    queryKey: ["fb-ads-insights", params],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/insights/daily?${queryParams}`,
      );
      if (!response.ok) throw new Error("Erro ao carregar insights");
      const data = await response.json();
      return data.data || [];
    },
    enabled: !!params.configId,
  });
}

// ==============================
// CAMPAIGN HOOKS
// ==============================

/**
 * Hook para listar campanhas com métricas
 */
export function useCampaigns(params: {
  configId?: number;
  status?: string;
  limit?: number;
  offset?: number;
  datePreset?: DatePreset;
  insightsTable?: InsightsTableSource;
}) {
  const queryParams = new URLSearchParams();
  if (params.configId) queryParams.set("configId", String(params.configId));
  if (params.status) queryParams.set("status", params.status);
  if (params.limit) queryParams.set("limit", String(params.limit));
  if (params.offset) queryParams.set("offset", String(params.offset));
  if (params.datePreset) queryParams.set("datePreset", params.datePreset);

  return useQuery<PaginatedResponse<CampaignWithMetrics>>({
    queryKey: ["fb-ads-campaigns", params],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/campaigns?${queryParams}`,
      );
      if (!response.ok) throw new Error("Erro ao carregar campanhas");
      return response.json();
    },
    enabled: !!params.configId,
  });
}

/**
 * Hook para detalhes de uma campanha
 */
export function useCampaignDetails(campaignId: string | null) {
  return useQuery<{
    campaign: Campaign;
    insights: InsightsSummary;
    adsets: AdSet[];
  }>({
    queryKey: ["fb-ads-campaigns", campaignId],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/campaigns/${campaignId}`,
      );
      if (!response.ok)
        throw new Error("Erro ao carregar detalhes da campanha");
      const data = await response.json();
      return data.data;
    },
    enabled: !!campaignId,
  });
}

/**
 * Hook para ad sets de uma campanha
 */
export function useCampaignAdSets(campaignId: string | null) {
  return useQuery<AdSet[]>({
    queryKey: ["fb-ads-campaigns", campaignId, "adsets"],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/campaigns/${campaignId}/adsets`,
      );
      if (!response.ok) throw new Error("Erro ao carregar ad sets");
      const data = await response.json();
      return data.data || [];
    },
    enabled: !!campaignId,
  });
}

/**
 * Hook para ads de uma campanha
 */
export function useCampaignAds(campaignId: string | null) {
  return useQuery<Ad[]>({
    queryKey: ["fb-ads-campaigns", campaignId, "ads"],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/campaigns/${campaignId}/ads`,
      );
      if (!response.ok) throw new Error("Erro ao carregar ads");
      const data = await response.json();
      return data.data || [];
    },
    enabled: !!campaignId,
  });
}

// ==============================
// AD SETS HOOKS
// ==============================

/**
 * Hook para listar ad sets globalmente com métricas
 */
export function useAdSets(params: {
  configId?: number;
  campaignId?: string;
  status?: string;
  limit?: number;
  offset?: number;
  datePreset?: DatePreset;
  insightsTable?: InsightsTableSource;
}) {
  const queryParams = new URLSearchParams();
  if (params.configId) queryParams.set("configId", String(params.configId));
  if (params.campaignId) queryParams.set("campaignId", params.campaignId);
  if (params.status) queryParams.set("status", params.status);
  if (params.limit) queryParams.set("limit", String(params.limit));
  if (params.offset) queryParams.set("offset", String(params.offset));
  if (params.datePreset) queryParams.set("datePreset", params.datePreset);

  return useQuery<PaginatedResponse<AdSetWithMetrics>>({
    queryKey: ["fb-ads-adsets", params],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/adsets?${queryParams}`,
      );
      if (!response.ok)
        throw new Error("Erro ao carregar conjuntos de anúncios");
      return response.json();
    },
    enabled: !!params.configId,
  });
}

// ==============================
// ADS HOOKS
// ==============================

/**
 * Hook para listar ads globalmente com métricas
 */
export function useAds(params: {
  configId?: number;
  campaignId?: string;
  adsetId?: string;
  status?: string;
  limit?: number;
  offset?: number;
  datePreset?: DatePreset;
  insightsTable?: InsightsTableSource;
}) {
  const queryParams = new URLSearchParams();
  if (params.configId) queryParams.set("configId", String(params.configId));
  if (params.campaignId) queryParams.set("campaignId", params.campaignId);
  if (params.adsetId) queryParams.set("adsetId", params.adsetId);
  if (params.status) queryParams.set("status", params.status);
  if (params.limit) queryParams.set("limit", String(params.limit));
  if (params.offset) queryParams.set("offset", String(params.offset));
  if (params.datePreset) queryParams.set("datePreset", params.datePreset);

  return useQuery<PaginatedResponse<AdWithMetrics>>({
    queryKey: ["fb-ads-ads", params],
    queryFn: async () => {
      const response = await fbAdsFetch(
        `/ads?${queryParams}`,
      );
      if (!response.ok) throw new Error("Erro ao carregar anúncios");
      return response.json();
    },
    enabled: !!params.configId,
  });
}

// ==============================
// UTILITY HOOKS
// ==============================

/**
 * Hook para formatar valores de moeda
 */
export function useFormatCurrency() {
  return (value: number | null | undefined): string => {
    if (value === null || value === undefined) return "—";
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL",
    }).format(value);
  };
}

/**
 * Hook para formatar números grandes
 */
export function useFormatNumber() {
  return (value: number | null | undefined): string => {
    if (value === null || value === undefined) return "—";
    return new Intl.NumberFormat("pt-BR").format(value);
  };
}

/**
 * Hook para formatar porcentagem
 */
export function useFormatPercent() {
  return (value: number | null | undefined): string => {
    if (value === null || value === undefined) return "—";
    return `${value.toFixed(2)}%`;
  };
}

// ==============================
// OAUTH / CONFIG MANAGEMENT HOOKS
// ==============================

/**
 * Hook para listar contas de anúncios disponíveis
 */
export function useAdAccounts(enabled = false) {
  return useQuery<{ id: string; name: string; currency: string }[]>({
    queryKey: ["fb-ads-oauth-ad-accounts"],
    queryFn: async () => {
      const response = await fbAdsFetch("/oauth/ad-accounts");
      if (!response.ok) throw new Error("Erro ao carregar contas de anúncios");
      const data = await response.json();
      return data.data || [];
    },
    enabled,
  });
}

/**
 * Hook para excluir configuração
 */
export function useDeleteConfig() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { configId: number; hardDelete?: boolean }) => {
      const { configId, hardDelete = true } = params;
      const queryStr = hardDelete ? "?hardDelete=true" : "";
      return fbAdsRequest(
        "DELETE",
        `/config/${configId}${queryStr}`,
        {},
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-config"],
      });
    },
  });
}

/**
 * Hook para ativar/desativar configuração
 */
export function useToggleConfigActive() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { configId: number; isActive: boolean }) => {
      return fbAdsRequest(
        "PATCH",
        `/config/${params.configId}`,
        { isActive: params.isActive },
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-config"],
      });
    },
  });
}

/**
 * Hook para completar fluxo OAuth
 */
export function useCompleteOAuth() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: { adAccountId: string }) => {
      return fbAdsRequest("POST", "/oauth/complete", params);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ["fb-ads-config"],
      });
    },
  });
}
