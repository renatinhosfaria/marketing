"use client";

/**
 * Card resumo das classificações de campanhas
 */

import { useClassificationSummary, useClassifyCampaigns } from '@/hooks/use-ml';
import { TIER_COLORS, TIER_LABELS, type CampaignTier } from '@/types/ml';
import { cn } from '@/lib/utils';
import { BarChart3, RefreshCw, TrendingUp } from 'lucide-react';

interface ClassificationCardProps {
  configId: number;
}

export function ClassificationCard({ configId }: ClassificationCardProps) {
  const { data: summary, isLoading, refetch } = useClassificationSummary(configId);
  const classifyMutation = useClassifyCampaigns(configId);

  const handleClassify = async () => {
    try {
      await classifyMutation.mutateAsync(undefined);
      refetch();
    } catch (error) {
      console.error('Erro ao classificar campanhas:', error);
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

  const tiers: CampaignTier[] = ['HIGH_PERFORMER', 'MODERATE', 'LOW', 'UNDERPERFORMER'];

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-indigo-600" />
          <h3 className="font-semibold text-gray-900">Classificação de Campanhas</h3>
        </div>
        <button
          onClick={handleClassify}
          disabled={classifyMutation.isPending}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn('h-4 w-4', classifyMutation.isPending && 'animate-spin')} />
          {classifyMutation.isPending ? 'Classificando...' : 'Classificar'}
        </button>
      </div>

      {summary?.total_campaigns === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <BarChart3 className="h-12 w-12 mx-auto mb-2 opacity-30" />
          <p>Nenhuma campanha classificada</p>
          <p className="text-sm mt-1">Clique em &quot;Classificar&quot; para analisar suas campanhas</p>
        </div>
      ) : (
        <>
          <div className="mb-4">
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-gray-900">{summary?.total_campaigns || 0}</span>
              <span className="text-sm text-gray-500">campanhas classificadas</span>
            </div>
            {summary?.average_confidence && (
              <p className="text-sm text-gray-500 mt-1">
                Confiança média: {(summary.average_confidence * 100).toFixed(0)}%
              </p>
            )}
          </div>

          <div className="space-y-2">
            {tiers.map((tier) => {
              const count = summary?.by_tier?.[tier] || 0;
              const percentage = summary?.total_campaigns
                ? ((count / summary.total_campaigns) * 100).toFixed(0)
                : '0';
              const colors = TIER_COLORS[tier];

              return (
                <div key={tier} className="flex items-center gap-3">
                  <div className={cn(
                    'w-3 h-3 rounded-full',
                    colors.bg,
                    colors.border,
                    'border-2'
                  )} />
                  <span className="text-sm text-gray-600 flex-1">{TIER_LABELS[tier]}</span>
                  <span className={cn(
                    'px-2 py-0.5 rounded text-xs font-medium',
                    colors.bg,
                    colors.text
                  )}>
                    {count}
                  </span>
                  <span className="text-xs text-gray-400 w-10 text-right">{percentage}%</span>
                </div>
              );
            })}
          </div>

          {summary?.recent_changes !== undefined && summary.recent_changes > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-100">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <TrendingUp className="h-4 w-4 text-green-500" />
                <span>{summary.recent_changes} mudanças recentes de tier</span>
              </div>
            </div>
          )}

          {summary?.last_classification_at && (
            <p className="text-xs text-gray-400 mt-3">
              Última classificação: {new Date(summary.last_classification_at).toLocaleString('pt-BR')}
            </p>
          )}
        </>
      )}
    </div>
  );
}
