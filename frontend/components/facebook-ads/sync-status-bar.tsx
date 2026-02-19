"use client";

/**
 * Componente de Barra de Status de Sincronizacao
 * Story 2.6: Mostra status da ultima sync e botao de refresh
 */

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { toast } from "sonner";
import { formatDistanceToNow } from "date-fns";
import { ptBR } from "date-fns/locale";
import {
  AlertCircle,
  CheckCircle2,
  Clock,
  Loader2,
  RefreshCw,
} from "lucide-react";
import { useCancelSync, useSyncStatus, useTriggerTodaySync } from "@/hooks/use-facebook-ads";

interface SyncStatusBarProps {
  configId: number;
  accountName?: string;
}

export function SyncStatusBar({ configId }: SyncStatusBarProps) {
  const { data: syncStatus, isLoading } = useSyncStatus(configId);
  const triggerTodaySync = useTriggerTodaySync();
  const cancelSync = useCancelSync();

  const handleSync = async () => {
    try {
      await triggerTodaySync.mutateAsync(configId);
      toast.success("Sincronizacao iniciada", {
        description: "Os dados de hoje e dos ultimos dias estao sendo atualizados.",
      });
    } catch (error) {
      toast.error("Erro ao iniciar sincronizacao", {
        description:
          error instanceof Error ? error.message : "Erro desconhecido",
      });
    }
  };

  const handleCancel = async () => {
    try {
      await cancelSync.mutateAsync(configId);
      toast.success("Sincronizacao cancelada", {
        description: "A sincronizacao foi interrompida.",
      });
    } catch (error) {
      toast.error("Erro ao cancelar", {
        description:
          error instanceof Error ? error.message : "Erro desconhecido",
      });
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span>Carregando status...</span>
      </div>
    );
  }

  if (!syncStatus) {
    return null;
  }

  const { isRunning, lastSync, lastSyncAt, nextSyncAt, progress } = syncStatus;

  // Calcular tempo desde ultima sync
  const lastSyncText = lastSyncAt
    ? formatDistanceToNow(new Date(lastSyncAt), {
        addSuffix: true,
        locale: ptBR,
      })
    : "Nunca sincronizado";

  // Calcular proxima sync
  const nextSyncText = nextSyncAt
    ? formatDistanceToNow(new Date(nextSyncAt), {
        addSuffix: false,
        locale: ptBR,
      })
    : null;

  // Verificar se dados estao desatualizados (> 1 hora)
  const isStale = lastSyncAt
    ? Date.now() - new Date(lastSyncAt).getTime() > 60 * 60 * 1000
    : true;

  // Status da ultima sync
  const lastSyncStatus = lastSync?.status;
  const lastSyncFailed = lastSyncStatus === "failed";

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-lg border bg-card px-4 py-2 text-sm">
      {/* Icone de status */}
      <div className="flex items-center gap-2">
        {isRunning ? (
          <Loader2 className="h-4 w-4 animate-spin text-blue-500" />
        ) : lastSyncFailed ? (
          <AlertCircle className="h-4 w-4 text-destructive" />
        ) : (
          <CheckCircle2 className="h-4 w-4 text-green-500" />
        )}
      </div>

      {/* Ultima atualizacao */}
      <div className="flex items-center gap-1.5 text-muted-foreground">
        <Clock className="h-3.5 w-3.5" />
        <span>
          {isRunning ? (
            <>
              Sincronizando
              {progress && (
                <span className="ml-1 text-xs">({progress.stage})</span>
              )}
            </>
          ) : (
            <>Ultima atualizacao: {lastSyncText}</>
          )}
        </span>
      </div>

      {/* Proxima sync */}
      {!isRunning && nextSyncText && (
        <>
          <span className="text-muted-foreground">|</span>
          <span className="text-muted-foreground">
            Proxima: em {nextSyncText}
          </span>
        </>
      )}

      {/* Badge de erro se ultima sync falhou */}
      {lastSyncFailed && !isRunning && (
        <Tooltip>
          <TooltipTrigger>
            <Badge variant="destructive" className="text-xs">
              Erro na ultima sync
            </Badge>
          </TooltipTrigger>
          <TooltipContent>
            <p>{lastSync?.errorMessage || "Erro desconhecido"}</p>
          </TooltipContent>
        </Tooltip>
      )}

      {/* Badge se dados estao desatualizados */}
      {isStale && !isRunning && !lastSyncFailed && (
        <Badge
          variant="outline"
          className="text-xs text-amber-600 border-amber-300"
        >
          Dados desatualizados
        </Badge>
      )}

      {/* Progresso se rodando */}
      {isRunning && progress && (
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span>
            {progress.campaignsSynced} campanhas, {progress.adsetsSynced} ad
            sets, {progress.adsSynced} ads
          </span>
        </div>
      )}

      {/* Spacer */}
      <div className="flex-1" />

      {/* Botao de acao */}
      {isRunning ? (
        <Button
          variant="outline"
          size="sm"
          onClick={handleCancel}
          disabled={cancelSync.isPending}
        >
          {cancelSync.isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            "Cancelar"
          )}
        </Button>
      ) : (
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={isStale || lastSyncFailed ? "default" : "outline"}
              size="sm"
              onClick={handleSync}
              disabled={triggerTodaySync.isPending}
            >
              {triggerTodaySync.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="mr-2 h-4 w-4" />
              )}
              Atualizar Agora
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Sincronizar hoje e ultimos dias</p>
          </TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}
