"use client";

/**
 * Badge de status do serviço ML
 */

import { useMLHealth } from '@/hooks/use-ml';
import { cn } from '@/lib/utils';

export function MLStatusBadge() {
  const { data: health, isLoading, isError } = useMLHealth();

  if (isLoading) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-600">
        <span className="h-2 w-2 rounded-full bg-gray-400 animate-pulse" />
        Verificando ML...
      </span>
    );
  }

  if (isError || !health) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium bg-red-100 text-red-700">
        <span className="h-2 w-2 rounded-full bg-red-500" />
        ML Offline
      </span>
    );
  }

  const statusConfig = {
    healthy: {
      bg: 'bg-green-100',
      text: 'text-green-700',
      dot: 'bg-green-500',
      label: 'ML Ativo',
    },
    degraded: {
      bg: 'bg-yellow-100',
      text: 'text-yellow-700',
      dot: 'bg-yellow-500',
      label: 'ML Degradado',
    },
    unhealthy: {
      bg: 'bg-red-100',
      text: 'text-red-700',
      dot: 'bg-red-500',
      label: 'ML Indisponível',
    },
  };

  const config = statusConfig[health.status] || statusConfig.unhealthy;

  return (
    <span className={cn(
      'inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium',
      config.bg,
      config.text
    )}>
      <span className={cn('h-2 w-2 rounded-full', config.dot)} />
      {config.label}
      <span className="text-xs opacity-70">v{health.version}</span>
    </span>
  );
}
