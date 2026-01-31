"use client";

/**
 * Lista de recomendações com ações
 */

import { useState } from 'react';
import { useRecommendations, useApplyRecommendation, useDismissRecommendation } from '@/hooks/use-ml';
import { RECOMMENDATION_TYPE_LABELS, type Recommendation, type RecommendationType } from '@/types/ml';
import { cn } from '@/lib/utils';
import {
  TrendingUp,
  TrendingDown,
  Pause,
  Rocket,
  Paintbrush,
  Users,
  Play,
  Clock,
  Check,
  X,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

interface RecommendationsListProps {
  configId: number;
  limit?: number;
}

const TYPE_ICONS: Record<RecommendationType, React.ComponentType<{ className?: string }>> = {
  BUDGET_INCREASE: TrendingUp,
  BUDGET_DECREASE: TrendingDown,
  PAUSE_CAMPAIGN: Pause,
  SCALE_UP: Rocket,
  CREATIVE_REFRESH: Paintbrush,
  AUDIENCE_REVIEW: Users,
  REACTIVATE: Play,
  OPTIMIZE_SCHEDULE: Clock,
};

const PRIORITY_COLORS: Record<string, string> = {
  high: 'bg-red-100 text-red-700 border-red-200',
  medium: 'bg-yellow-100 text-yellow-700 border-yellow-200',
  low: 'bg-gray-100 text-gray-700 border-gray-200',
};

function getPriorityLevel(priority: number): 'high' | 'medium' | 'low' {
  if (priority >= 7) return 'high';
  if (priority >= 4) return 'medium';
  return 'low';
}

interface RecommendationItemProps {
  recommendation: Recommendation;
  onApply: (id: number) => void;
  onDismiss: (id: number, reason: string) => void;
  isApplying: boolean;
  isDismissing: boolean;
}

function RecommendationItem({
  recommendation,
  onApply,
  onDismiss,
  isApplying,
  isDismissing,
}: RecommendationItemProps) {
  const [expanded, setExpanded] = useState(false);
  const [dismissReason, setDismissReason] = useState('');
  const [showDismissForm, setShowDismissForm] = useState(false);

  const Icon = TYPE_ICONS[recommendation.recommendation_type] || TrendingUp;
  const priorityLevel = getPriorityLevel(recommendation.priority);

  const handleDismiss = () => {
    if (dismissReason.trim()) {
      onDismiss(recommendation.id, dismissReason);
      setShowDismissForm(false);
      setDismissReason('');
    }
  };

  return (
    <div className={cn(
      'border rounded-lg p-4 transition-all',
      recommendation.was_applied && 'bg-green-50 border-green-200',
      recommendation.dismissed && 'bg-gray-50 border-gray-200 opacity-60'
    )}>
      <div className="flex items-start gap-3">
        <div className={cn(
          'p-2 rounded-lg',
          priorityLevel === 'high' && 'bg-red-100',
          priorityLevel === 'medium' && 'bg-yellow-100',
          priorityLevel === 'low' && 'bg-gray-100'
        )}>
          <Icon className={cn(
            'h-5 w-5',
            priorityLevel === 'high' && 'text-red-600',
            priorityLevel === 'medium' && 'text-yellow-600',
            priorityLevel === 'low' && 'text-gray-600'
          )} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <h4 className="font-medium text-gray-900 truncate">{recommendation.title}</h4>
            <span className={cn(
              'px-2 py-0.5 rounded text-xs font-medium border',
              PRIORITY_COLORS[priorityLevel]
            )}>
              P{recommendation.priority}
            </span>
          </div>

          <p className="text-sm text-gray-500 mb-2">
            {RECOMMENDATION_TYPE_LABELS[recommendation.recommendation_type]}
            {' \u2022 '}
            Confiança: {(recommendation.confidence_score * 100).toFixed(0)}%
          </p>

          <p className={cn(
            'text-sm text-gray-600',
            !expanded && 'line-clamp-2'
          )}>
            {recommendation.description}
          </p>

          {expanded && recommendation.suggested_action && (
            <div className="mt-3 p-3 bg-blue-50 rounded-lg">
              <p className="text-sm font-medium text-blue-800 mb-1">Ação Sugerida:</p>
              {recommendation.suggested_action.change_value !== undefined && (
                <p className="text-sm text-blue-700">
                  {recommendation.suggested_action.change_type === 'percentage'
                    ? `${recommendation.suggested_action.change_value > 0 ? '+' : ''}${recommendation.suggested_action.change_value}%`
                    : recommendation.suggested_action.change_value
                  }
                  {recommendation.suggested_action.field && ` em ${recommendation.suggested_action.field}`}
                </p>
              )}
              {recommendation.suggested_action.recommendations && (
                <ul className="text-sm text-blue-700 list-disc list-inside mt-1">
                  {recommendation.suggested_action.recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <button
            onClick={() => setExpanded(!expanded)}
            className="mt-2 text-sm text-indigo-600 hover:text-indigo-700 flex items-center gap-1"
          >
            {expanded ? (
              <>
                <ChevronUp className="h-4 w-4" />
                Ver menos
              </>
            ) : (
              <>
                <ChevronDown className="h-4 w-4" />
                Ver mais
              </>
            )}
          </button>
        </div>

        {!recommendation.was_applied && !recommendation.dismissed && (
          <div className="flex items-center gap-2">
            <button
              onClick={() => onApply(recommendation.id)}
              disabled={isApplying}
              className="p-2 text-green-600 hover:bg-green-50 rounded-lg transition-colors disabled:opacity-50"
              title="Marcar como aplicada"
            >
              <Check className="h-5 w-5" />
            </button>
            <button
              onClick={() => setShowDismissForm(!showDismissForm)}
              disabled={isDismissing}
              className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
              title="Descartar"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        )}

        {recommendation.was_applied && (
          <span className="px-2 py-1 bg-green-100 text-green-700 text-xs font-medium rounded">
            Aplicada
          </span>
        )}

        {recommendation.dismissed && (
          <span className="px-2 py-1 bg-gray-100 text-gray-600 text-xs font-medium rounded">
            Descartada
          </span>
        )}
      </div>

      {showDismissForm && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Motivo do descarte:
          </label>
          <textarea
            value={dismissReason}
            onChange={(e) => setDismissReason(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
            rows={2}
            placeholder="Ex: Já implementei essa mudança, não se aplica ao meu caso..."
          />
          <div className="flex justify-end gap-2 mt-2">
            <button
              onClick={() => setShowDismissForm(false)}
              className="px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded"
            >
              Cancelar
            </button>
            <button
              onClick={handleDismiss}
              disabled={!dismissReason.trim() || isDismissing}
              className="px-3 py-1.5 text-sm bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
            >
              Descartar
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export function RecommendationsList({ configId, limit = 10 }: RecommendationsListProps) {
  const { data, isLoading, error } = useRecommendations(configId, { limit, is_active: true });
  const applyMutation = useApplyRecommendation();
  const dismissMutation = useDismissRecommendation();

  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <div key={i} className="border rounded-lg p-4 animate-pulse">
            <div className="flex items-start gap-3">
              <div className="h-9 w-9 bg-gray-200 rounded-lg" />
              <div className="flex-1">
                <div className="h-5 bg-gray-200 rounded w-1/3 mb-2" />
                <div className="h-4 bg-gray-200 rounded w-full mb-1" />
                <div className="h-4 bg-gray-200 rounded w-2/3" />
              </div>
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8 text-red-500">
        Erro ao carregar recomendações: {error.message}
      </div>
    );
  }

  if (!data?.recommendations?.length) {
    return (
      <div className="text-center py-8 text-gray-500">
        Nenhuma recomendação ativa no momento.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {data.recommendations.map((rec) => (
        <RecommendationItem
          key={rec.id}
          recommendation={rec}
          onApply={(id) => applyMutation.mutate({ id })}
          onDismiss={(id, reason) => dismissMutation.mutate({ id, reason })}
          isApplying={applyMutation.isPending}
          isDismissing={dismissMutation.isPending}
        />
      ))}
    </div>
  );
}
