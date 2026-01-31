"use client";

/**
 * Card resumo das recomendações
 */

import { useRecommendationSummary, useGenerateRecommendations } from '@/hooks/use-ml';
import { RECOMMENDATION_TYPE_LABELS, type RecommendationType } from '@/types/ml';
import { cn } from '@/lib/utils';
import { Lightbulb, RefreshCw, AlertTriangle, ChevronRight } from 'lucide-react';

interface RecommendationsCardProps {
  configId: number;
  onViewAll?: () => void;
}

export function RecommendationsCard({ configId, onViewAll }: RecommendationsCardProps) {
  const { data: summary, isLoading, refetch } = useRecommendationSummary(configId);
  const generateMutation = useGenerateRecommendations(configId);

  const handleGenerate = async () => {
    try {
      await generateMutation.mutateAsync(false);
      refetch();
    } catch (error) {
      console.error('Erro ao gerar recomendações:', error);
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

  // Ordenar tipos por quantidade
  const sortedTypes = summary?.by_type
    ? Object.entries(summary.by_type)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 4)
    : [];

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-amber-500" />
          <h3 className="font-semibold text-gray-900">Recomendações</h3>
        </div>
        <button
          onClick={handleGenerate}
          disabled={generateMutation.isPending}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-amber-600 hover:bg-amber-50 rounded-md transition-colors disabled:opacity-50"
        >
          <RefreshCw className={cn('h-4 w-4', generateMutation.isPending && 'animate-spin')} />
          {generateMutation.isPending ? 'Gerando...' : 'Gerar'}
        </button>
      </div>

      {summary?.total === 0 ? (
        <div className="text-center py-8 text-gray-500">
          <Lightbulb className="h-12 w-12 mx-auto mb-2 opacity-30" />
          <p>Nenhuma recomendação ativa</p>
          <p className="text-sm mt-1">Clique em &quot;Gerar&quot; para analisar suas campanhas</p>
        </div>
      ) : (
        <>
          <div className="mb-4">
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-bold text-gray-900">{summary?.total || 0}</span>
              <span className="text-sm text-gray-500">recomendações ativas</span>
            </div>
          </div>

          {summary?.high_priority_count && summary.high_priority_count > 0 && (
            <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
              <div className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 text-amber-600" />
                <span className="text-sm font-medium text-amber-800">
                  {summary.high_priority_count} alta prioridade
                </span>
              </div>
            </div>
          )}

          <div className="space-y-2">
            {sortedTypes.map(([type, count]) => (
              <div key={type} className="flex items-center justify-between text-sm">
                <span className="text-gray-600">
                  {RECOMMENDATION_TYPE_LABELS[type as RecommendationType] || type}
                </span>
                <span className="px-2 py-0.5 bg-gray-100 rounded text-gray-700 font-medium">
                  {count}
                </span>
              </div>
            ))}
          </div>

          {onViewAll && (
            <button
              onClick={onViewAll}
              className="mt-4 w-full flex items-center justify-center gap-1 py-2 text-sm font-medium text-indigo-600 hover:bg-indigo-50 rounded-md transition-colors"
            >
              Ver todas
              <ChevronRight className="h-4 w-4" />
            </button>
          )}
        </>
      )}
    </div>
  );
}
