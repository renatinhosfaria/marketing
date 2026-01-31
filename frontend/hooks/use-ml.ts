"use client";

/**
 * React Query hooks para a API ML
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiFetchJson } from '@/lib/api';
import type {
  CampaignClassification,
  ClassificationSummary,
  Recommendation,
  RecommendationSummary,
  Anomaly,
  AnomalySummary,
  MLHealthStatus,
} from '@/types/ml';

const ML_API_BASE = '/api/v1';

// ==================== HEALTH ====================

export function useMLHealth() {
  return useQuery<MLHealthStatus>({
    queryKey: ['ml', 'health'],
    queryFn: () => apiFetchJson<MLHealthStatus>(`${ML_API_BASE}/health`),
    refetchInterval: 30000, // 30 segundos
    staleTime: 10000,
  });
}

export function useMLHealthDetailed() {
  return useQuery<MLHealthStatus>({
    queryKey: ['ml', 'health', 'detailed'],
    queryFn: () => apiFetchJson<MLHealthStatus>(`${ML_API_BASE}/health/detailed`),
    staleTime: 30000,
  });
}

// ==================== CLASSIFICAÇÕES ====================

export function useClassificationSummary(configId: number | undefined) {
  return useQuery<ClassificationSummary>({
    queryKey: ['ml', 'classifications', 'summary', configId],
    queryFn: () =>
      apiFetchJson<ClassificationSummary>(
        `${ML_API_BASE}/classifications/summary?config_id=${configId}`,
      ),
    enabled: !!configId,
    staleTime: 60000,
  });
}

export function useClassifications(
  configId: number | undefined,
  options?: {
    tier?: string;
    limit?: number;
  },
) {
  const params = new URLSearchParams();
  if (configId) params.set('config_id', String(configId));
  if (options?.tier) params.set('tier', options.tier);
  if (options?.limit) params.set('limit', String(options.limit));

  return useQuery<{ classifications: CampaignClassification[]; total: number }>({
    queryKey: ['ml', 'classifications', configId, options],
    queryFn: () =>
      apiFetchJson<{ classifications: CampaignClassification[]; total: number }>(
        `${ML_API_BASE}/classifications?${params.toString()}`,
      ),
    enabled: !!configId,
    staleTime: 60000,
  });
}

export function useCampaignClassification(
  configId: number | undefined,
  campaignId: string | undefined,
) {
  return useQuery<CampaignClassification>({
    queryKey: ['ml', 'classifications', 'campaign', configId, campaignId],
    queryFn: () =>
      apiFetchJson<CampaignClassification>(
        `${ML_API_BASE}/classifications/campaign/${campaignId}?config_id=${configId}`,
      ),
    enabled: !!configId && !!campaignId,
    staleTime: 60000,
  });
}

export function useClassifyCampaigns(configId: number) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (campaignIds?: string[]) =>
      apiFetchJson(`${ML_API_BASE}/classifications/classify`, {
        method: 'POST',
        body: JSON.stringify({ config_id: configId, campaign_ids: campaignIds }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml', 'classifications'] });
    },
  });
}

// ==================== RECOMENDAÇÕES ====================

export function useRecommendationSummary(configId: number | undefined) {
  return useQuery<RecommendationSummary>({
    queryKey: ['ml', 'recommendations', 'summary', configId],
    queryFn: () =>
      apiFetchJson<RecommendationSummary>(
        `${ML_API_BASE}/recommendations/summary?config_id=${configId}`,
      ),
    enabled: !!configId,
    staleTime: 60000,
  });
}

export function useRecommendations(
  configId: number | undefined,
  options?: {
    type?: string;
    priority_min?: number;
    is_active?: boolean;
    limit?: number;
  },
) {
  const params = new URLSearchParams();
  if (configId) params.set('config_id', String(configId));
  if (options?.type) params.set('type', options.type);
  if (options?.priority_min) params.set('priority_min', String(options.priority_min));
  if (options?.is_active !== undefined) params.set('is_active', String(options.is_active));
  if (options?.limit) params.set('limit', String(options.limit));

  return useQuery<{ recommendations: Recommendation[]; total: number; by_type: Record<string, number> }>({
    queryKey: ['ml', 'recommendations', configId, options],
    queryFn: () =>
      apiFetchJson<{ recommendations: Recommendation[]; total: number; by_type: Record<string, number> }>(
        `${ML_API_BASE}/recommendations?${params.toString()}`,
      ),
    enabled: !!configId,
    staleTime: 60000,
  });
}

export function useGenerateRecommendations(configId: number) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (forceRefresh?: boolean) =>
      apiFetchJson(`${ML_API_BASE}/recommendations/generate`, {
        method: 'POST',
        body: JSON.stringify({ config_id: configId, force_refresh: forceRefresh ?? false }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml', 'recommendations'] });
    },
  });
}

export function useApplyRecommendation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, notes }: { id: number; notes?: string }) =>
      apiFetchJson(`${ML_API_BASE}/recommendations/${id}/apply`, {
        method: 'POST',
        body: JSON.stringify({ notes }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml', 'recommendations'] });
    },
  });
}

export function useDismissRecommendation() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, reason }: { id: number; reason: string }) =>
      apiFetchJson(`${ML_API_BASE}/recommendations/${id}/dismiss`, {
        method: 'POST',
        body: JSON.stringify({ reason }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml', 'recommendations'] });
    },
  });
}

// ==================== ANOMALIAS ====================

export function useAnomalySummary(configId: number | undefined, days = 7) {
  return useQuery<AnomalySummary>({
    queryKey: ['ml', 'anomalies', 'summary', configId, days],
    queryFn: () =>
      apiFetchJson<AnomalySummary>(
        `${ML_API_BASE}/anomalies/summary?config_id=${configId}&days=${days}`,
      ),
    enabled: !!configId,
    staleTime: 60000,
  });
}

export function useAnomalies(
  configId: number | undefined,
  options?: {
    severity?: string;
    is_acknowledged?: boolean;
    limit?: number;
  },
) {
  const params = new URLSearchParams();
  if (configId) params.set('config_id', String(configId));
  if (options?.severity) params.set('severity', options.severity);
  if (options?.is_acknowledged !== undefined) params.set('is_acknowledged', String(options.is_acknowledged));
  if (options?.limit) params.set('limit', String(options.limit));

  return useQuery<{ anomalies: Anomaly[]; total: number }>({
    queryKey: ['ml', 'anomalies', configId, options],
    queryFn: () =>
      apiFetchJson<{ anomalies: Anomaly[]; total: number }>(
        `${ML_API_BASE}/anomalies?${params.toString()}`,
      ),
    enabled: !!configId,
    staleTime: 60000,
  });
}

export function useDetectAnomalies(configId: number) {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (options?: { entity_type?: string; entity_id?: string }) =>
      apiFetchJson(`${ML_API_BASE}/anomalies/detect`, {
        method: 'POST',
        body: JSON.stringify({ config_id: configId, ...options }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml', 'anomalies'] });
    },
  });
}

export function useAcknowledgeAnomaly() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ id, notes }: { id: number; notes?: string }) =>
      apiFetchJson(`${ML_API_BASE}/anomalies/${id}/acknowledge`, {
        method: 'POST',
        body: JSON.stringify({ notes }),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ml', 'anomalies'] });
    },
  });
}
