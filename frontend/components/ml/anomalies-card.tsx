"use client";

/**
 * Card resumo das anomalias detectadas
 */

import { useAnomalySummary, useDetectAnomalies } from '@/hooks/use-ml';
import { SEVERITY_COLORS, type AnomalySeverity } from '@/types/ml';
import { cn } from '@/lib/utils';
import { AlertTriangle, RefreshCw, Bell } from 'lucide-react';

interface AnomaliesCardProps {
  configId: number;
  days?: number;
}

const SEVERITY_LABELS: Record<AnomalySeverity, string> = {
  LOW: 'Baixa',
  MEDIUM: 'Média',
  HIGH: 'Alta',
  CRITICAL: 'Crítica',
};

export function AnomaliesCard({ configId, days = 7 }: AnomaliesCardProps) {
  const { data: summary, isLoading, refetch } = useAnomalySummary(configId, days);
  const detectMutation = useDetectAnomalies(configId);

  const handleDetect = async () => {
    try {
      await detectMutation.mutateAsync(undefined);
      refetch();
    } catch (error) {
      console.error('Erro ao detectar anomalias:', error);
    }
  };

  if (isLoading) {
    return (
      <div className="bg-white rounded-lg border border-gray-200 p-6 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-1/3 mb-4" />
        <div className="space-y-2">
          <div className="h-4 bg-gray-200 rounded w-full" />
          <div className="h-4 bg-gray-200 rounded w-2/3" />
        </div>
      </div>
    );
  }

  const severities: AnomalySeverity[] = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'];
  const hasAnomalies = summary && summary.total > 0;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-orange-500" />
          <h3 className="font-semibold text-gray-900">Anomalias Detectadas</h3>
        </div>
        <button
          onClick={handleDetect}
          disabled={detectMutation.isPending}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-orange-600 hover:bg-orange-50 rounded-md transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn('h-4 w-4', detectMutation.isPending && 'animate-spin')} />
          {detectMutation.isPending ? 'Detectando...' : 'Detectar'}
        </button>
      </div>

      {!hasAnomalies ? (
        <div className="text-center py-8 text-gray-500">
          <AlertTriangle className="h-12 w-12 mx-auto mb-2 opacity-30" />
          <p>Nenhuma anomalia detectada</p>
          <p className="text-sm mt-1">Suas campanhas estão operando normalmente</p>
        </div>
      ) : (
        <>
          <div className="mb-4">
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-gray-900">{summary?.total || 0}</span>
              <span className="text-sm text-gray-500">anomalias nos últimos {days} dias</span>
            </div>
          </div>

          {summary?.unacknowledged && summary.unacknowledged > 0 && (
            <div className="mb-4 p-3 bg-orange-50 border border-orange-200 rounded-lg">
              <div className="flex items-center gap-2">
                <Bell className="h-4 w-4 text-orange-600" />
                <span className="text-sm font-medium text-orange-800">
                  {summary.unacknowledged} não reconhecidas
                </span>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {severities.map((severity) => {
              const count = summary?.by_severity?.[severity] || 0;
              if (count === 0) return null;

              const colors = SEVERITY_COLORS[severity];

              return (
                <div key={severity} className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">{SEVERITY_LABELS[severity]}</span>
                  <span className={cn(
                    'px-2 py-0.5 rounded text-xs font-medium',
                    colors.bg,
                    colors.text
                  )}>
                    {count}
                  </span>
                </div>
              );
            })}
          </div>

          {summary?.last_detected_at && (
            <p className="text-xs text-gray-400 mt-3">
              Última detecção: {new Date(summary.last_detected_at).toLocaleString('pt-BR')}
            </p>
          )}
        </>
      )}
    </div>
  );
}
