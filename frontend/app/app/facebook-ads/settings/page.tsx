"use client";

/**
 * Pagina de Configuracoes - Facebook Ads
 * Story 3.7: Gerenciamento de integracoes OAuth
 */

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle,
  Clock,
  ExternalLink,
  Link as LinkIcon,
  Plus,
  RefreshCw,
  Settings,
  Trash2,
} from "lucide-react";
import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { apiFetch } from "@/lib/api";
import {
  useAdAccounts,
  useCompleteOAuth,
  useDeleteConfig,
  useFacebookAdsConfigs,
  useToggleConfigActive,
} from "@/hooks/use-facebook-ads";
import type { FacebookConfig } from "@/types/facebook-ads";

interface AdAccountOption {
  id: string;
  name: string;
  status: number;
}

const FB_ADS_API = "/api/v1/facebook-ads";

export default function FacebookAdsSettings() {
  const router = useRouter();
  const searchParams = useSearchParams();
  // State
  const [showConnectDialog, setShowConnectDialog] = useState(false);
  const [selectedConfig, setSelectedConfig] = useState<FacebookConfig | null>(
    null,
  );
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [selectedAccountId, setSelectedAccountId] = useState<string>("");

  // State para selecao de conta OAuth
  const [showAccountSelectDialog, setShowAccountSelectDialog] = useState(false);
  const [availableAccounts, setAvailableAccounts] = useState<
    AdAccountOption[]
  >([]);
  const [tempData, setTempData] = useState<string>("");
  const [isCompletingSetup, setIsCompletingSetup] = useState(false);

  // Queries
  const {
    data: configs,
    isLoading: configsLoading,
    refetch: refetchConfigs,
  } = useFacebookAdsConfigs();
  const { data: adAccounts, isLoading: accountsLoading } = useAdAccounts(
    showConnectDialog ? true : false,
  );

  // Mutations
  const deleteConfig = useDeleteConfig();
  const toggleActive = useToggleConfigActive();
  const completeOAuth = useCompleteOAuth();

  // Processar parametros da URL apos callback OAuth
  useEffect(() => {
    const step = searchParams.get("step");
    const accounts = searchParams.get("accounts");
    const tempDataParam = searchParams.get("tempData");
    const error = searchParams.get("error");
    const errorMessage = searchParams.get("message");
    const success = searchParams.get("success");

    // Mostrar erro se houver
    if (error) {
      toast.error("Erro na autenticacao", {
        description: errorMessage || error,
      });
      // Limpar URL
      window.history.replaceState({}, "", "/app/facebook-ads/settings");
      return;
    }

    // Mostrar sucesso se houver
    if (success === "true") {
      toast.success("Conexao realizada", {
        description: "Conta de anuncios conectada com sucesso!",
      });
      refetchConfigs();
      // Limpar URL
      window.history.replaceState({}, "", "/app/facebook-ads/settings");
      return;
    }

    // Se for passo de selecao de conta
    if (step === "select_account" && accounts && tempDataParam) {
      try {
        const decodedAccounts = JSON.parse(
          atob(accounts),
        ) as AdAccountOption[];
        setAvailableAccounts(decodedAccounts);
        setTempData(tempDataParam);
        setShowAccountSelectDialog(true);
        // Limpar URL para evitar reprocessamento
        window.history.replaceState({}, "", "/app/facebook-ads/settings");
      } catch {
        toast.error("Erro", {
          description: "Falha ao processar contas de anuncios",
        });
      }
    }
  }, [searchParams, refetchConfigs]);

  // Handler para completar setup com conta selecionada
  const handleCompleteAccountSelection = async () => {
    if (!selectedAccountId || !tempData) {
      toast.error("Selecione uma conta", {
        description: "Escolha uma conta de anuncios para conectar",
      });
      return;
    }

    setIsCompletingSetup(true);
    try {
      const selectedAccount = availableAccounts.find(
        (acc) => acc.id === selectedAccountId,
      );

      const res = await apiFetch(`${FB_ADS_API}/oauth/complete-setup`, {
        method: "POST",
        body: JSON.stringify({
          accountId: selectedAccountId,
          accountName: selectedAccount?.name || selectedAccountId,
          tempData,
        }),
      });

      if (!res.ok) {
        const errorText = await res.text().catch(() => "Erro desconhecido");
        throw new Error(errorText);
      }

      toast.success("Sucesso", {
        description: "Conta de anuncios conectada com sucesso!",
      });
      setShowAccountSelectDialog(false);
      setSelectedAccountId("");
      setAvailableAccounts([]);
      setTempData("");
      refetchConfigs();
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Falha ao conectar conta";
      toast.error("Erro", { description: message });
    } finally {
      setIsCompletingSetup(false);
    }
  };

  // Handlers
  const handleStartOAuth = () => {
    // Redirecionar para fluxo OAuth do Facebook
    const apiBase = FB_ADS_API;
    const redirectUri = `${apiBase.startsWith("http") ? apiBase : window.location.origin + apiBase}/oauth/callback`;
    const clientId = process.env.NEXT_PUBLIC_FACEBOOK_APP_ID;

    if (!clientId) {
      toast.error("Erro de Configuracao", {
        description: "NEXT_PUBLIC_FACEBOOK_APP_ID nao esta configurado",
      });
      return;
    }

    const scope = [
      "ads_read",
      "ads_management",
      "business_management",
      "pages_read_engagement",
    ].join(",");

    const state = btoa(
      JSON.stringify({
        timestamp: Date.now(),
        userId: 1,
      }),
    );
    localStorage.setItem("fb_oauth_state", state);

    const authUrl = new URL("https://www.facebook.com/v18.0/dialog/oauth");
    authUrl.searchParams.set("client_id", clientId);
    authUrl.searchParams.set("redirect_uri", redirectUri);
    authUrl.searchParams.set("scope", scope);
    authUrl.searchParams.set("state", state);
    authUrl.searchParams.set("response_type", "code");

    window.location.href = authUrl.toString();
  };

  const handleCompleteSetup = async () => {
    if (!selectedAccountId) {
      toast.error("Selecione uma conta", {
        description: "Escolha uma conta de anuncios para conectar",
      });
      return;
    }

    try {
      await completeOAuth.mutateAsync({ adAccountId: selectedAccountId });
      toast.success("Sucesso", {
        description: "Conta de anuncios conectada com sucesso",
      });
      setShowConnectDialog(false);
      setSelectedAccountId("");
      refetchConfigs();
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Falha ao conectar conta";
      toast.error("Erro", { description: message });
    }
  };

  const handleToggleActive = async (config: FacebookConfig) => {
    try {
      await toggleActive.mutateAsync({
        configId: config.id,
        isActive: !config.isActive,
      });
      toast.success(config.isActive ? "Desativada" : "Ativada", {
        description: `Configuracao ${
          config.isActive ? "desativada" : "ativada"
        } com sucesso`,
      });
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Falha ao alterar status";
      toast.error("Erro", { description: message });
    }
  };

  const handleDeleteConfig = async () => {
    if (!selectedConfig) return;

    try {
      await deleteConfig.mutateAsync({
        configId: selectedConfig.id,
        hardDelete: true,
      });
      toast.success("Conexao excluida", {
        description:
          "A conexao e todos os dados relacionados foram excluidos permanentemente",
      });
      setShowDeleteDialog(false);
      setSelectedConfig(null);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Falha ao excluir conexao";
      toast.error("Erro", { description: message });
    }
  };

  const formatDate = (date: Date | string | null | undefined) => {
    if (!date) return "\u2014";
    return new Date(date).toLocaleDateString("pt-BR", {
      day: "2-digit",
      month: "2-digit",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const isTokenExpiring = (expiresAt: Date | string | null | undefined) => {
    if (!expiresAt) return false;
    const expDate = new Date(expiresAt);
    const daysUntilExpiry = Math.ceil(
      (expDate.getTime() - Date.now()) / (1000 * 60 * 60 * 24),
    );
    return daysUntilExpiry <= 7;
  };

  return (
    <div className="container mx-auto space-y-6 p-6">
      {/* Header */}
      <div className="flex flex-col gap-4">
        <Button
          variant="ghost"
          className="w-fit"
          onClick={() => router.push("/app/facebook-ads")}
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Voltar ao Dashboard
        </Button>

        <div className="flex items-center justify-between">
          <div className="space-y-1">
            <h1 className="flex items-center gap-2 text-2xl font-bold tracking-tight">
              <Settings className="h-6 w-6" />
              Configuracoes Facebook Ads
            </h1>
            <p className="text-muted-foreground">
              Gerencie suas integracoes com o Facebook Ads
            </p>
          </div>

          <Button onClick={handleStartOAuth}>
            <Plus className="mr-2 h-4 w-4" />
            Nova Conexao
          </Button>
        </div>
      </div>

      {/* Lista de Configuracoes */}
      {configsLoading ? (
        <div className="space-y-4">
          {[...Array(2)].map((_, i) => (
            <Skeleton key={i} className="h-48 w-full" />
          ))}
        </div>
      ) : !configs || configs.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <LinkIcon className="mb-4 h-12 w-12 text-muted-foreground" />
            <h3 className="mb-2 text-lg font-semibold">
              Nenhuma conexao configurada
            </h3>
            <p className="mb-4 text-center text-muted-foreground">
              Conecte sua conta do Facebook Ads para comecar a visualizar dados
            </p>
            <Button onClick={handleStartOAuth}>
              <Plus className="mr-2 h-4 w-4" />
              Conectar Facebook Ads
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {configs.map((config) => (
            <Card key={config.id}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="space-y-1">
                    <CardTitle className="flex items-center gap-2">
                      {config.accountName || `Conta ${config.accountId}`}
                      {config.isActive ? (
                        <Badge variant="default" className="gap-1">
                          <CheckCircle className="h-3 w-3" />
                          Ativa
                        </Badge>
                      ) : (
                        <Badge variant="secondary">Inativa</Badge>
                      )}
                    </CardTitle>
                    <CardDescription>
                      ID da conta: {config.accountId}
                    </CardDescription>
                  </div>

                  <div className="flex items-center gap-2">
                    <Switch
                      checked={config.isActive}
                      onCheckedChange={() => handleToggleActive(config)}
                      disabled={toggleActive.isPending}
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        setSelectedConfig(config);
                        setShowDeleteDialog(true);
                      }}
                    >
                      <Trash2 className="h-4 w-4 text-destructive" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-3">
                  {/* Token Status */}
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      Status do Token
                    </Label>
                    <div className="flex items-center gap-2">
                      {isTokenExpiring(config.tokenExpiresAt) ? (
                        <>
                          <AlertCircle className="h-4 w-4 text-amber-500" />
                          <span className="text-amber-600">
                            Expirando em breve
                          </span>
                        </>
                      ) : (
                        <>
                          <CheckCircle className="h-4 w-4 text-green-500" />
                          <span className="text-green-600">Valido</span>
                        </>
                      )}
                    </div>
                    {config.tokenExpiresAt && (
                      <p className="text-xs text-muted-foreground">
                        Expira em: {formatDate(config.tokenExpiresAt)}
                      </p>
                    )}
                  </div>

                  {/* Ultima Sincronizacao */}
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      Ultima Sincronizacao
                    </Label>
                    <div className="flex items-center gap-2">
                      <Clock className="h-4 w-4 text-muted-foreground" />
                      <span>
                        {config.lastSyncAt
                          ? formatDate(config.lastSyncAt)
                          : "Nunca"}
                      </span>
                    </div>
                  </div>

                  {/* Sync Config */}
                  <div className="space-y-1">
                    <Label className="text-xs text-muted-foreground">
                      Configuracao de Sync
                    </Label>
                    <div className="flex items-center gap-2">
                      <RefreshCw className="h-4 w-4 text-muted-foreground" />
                      <span>
                        A cada {config.syncFrequencyMinutes || 60} minutos
                      </span>
                    </div>
                  </div>
                </div>

                {/* Acoes */}
                <div className="mt-4 flex gap-2">
                  <Button variant="outline" size="sm" asChild>
                    <a
                      href={`https://business.facebook.com/adsmanager/manage/campaigns?act=${(
                        config.accountId || ""
                      ).replace("act_", "")}`}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="mr-2 h-4 w-4" />
                      Abrir no Ads Manager
                    </a>
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Informacoes de Ajuda */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Sobre a Integracao</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>
            &bull; Os dados sao sincronizados automaticamente a cada hora
            (configuravel)
          </p>
          <p>
            &bull; Metricas incluem: gastos, impressoes, cliques, CTR, CPC, CPL,
            leads
          </p>
          <p>&bull; Historico armazenado por 90 dias por padrao</p>
          <p>
            &bull; Token de acesso e renovado automaticamente quando possivel
          </p>
        </CardContent>
      </Card>

      {/* Dialog: Selecionar Conta */}
      <Dialog open={showConnectDialog} onOpenChange={setShowConnectDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Selecionar Conta de Anuncios</DialogTitle>
            <DialogDescription>
              Escolha qual conta de anuncios deseja conectar
            </DialogDescription>
          </DialogHeader>

          {accountsLoading ? (
            <div className="space-y-2 py-4">
              <Skeleton className="h-10 w-full" />
              <Skeleton className="h-10 w-full" />
            </div>
          ) : (
            <div className="py-4">
              <Label>Conta de Anuncios</Label>
              <Select
                value={selectedAccountId}
                onValueChange={setSelectedAccountId}
              >
                <SelectTrigger className="mt-2">
                  <SelectValue placeholder="Selecione uma conta" />
                </SelectTrigger>
                <SelectContent>
                  {adAccounts?.map((account) => (
                    <SelectItem key={account.id} value={account.id}>
                      {account.name} ({account.id})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowConnectDialog(false)}
            >
              Cancelar
            </Button>
            <Button
              onClick={handleCompleteSetup}
              disabled={!selectedAccountId || completeOAuth.isPending}
            >
              {completeOAuth.isPending ? "Conectando..." : "Conectar"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Dialog: Selecionar Conta apos OAuth */}
      <Dialog
        open={showAccountSelectDialog}
        onOpenChange={(open) => {
          if (!open) {
            setShowAccountSelectDialog(false);
            setSelectedAccountId("");
            setAvailableAccounts([]);
            setTempData("");
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Selecionar Conta de Anuncios</DialogTitle>
            <DialogDescription>
              Autenticacao concluida! Escolha qual conta de anuncios deseja
              conectar.
            </DialogDescription>
          </DialogHeader>

          <div className="py-4">
            <Label>Conta de Anuncios</Label>
            <Select
              value={selectedAccountId}
              onValueChange={setSelectedAccountId}
            >
              <SelectTrigger className="mt-2">
                <SelectValue placeholder="Selecione uma conta" />
              </SelectTrigger>
              <SelectContent>
                {availableAccounts.map((account) => (
                  <SelectItem key={account.id} value={account.id}>
                    {account.name} ({account.id})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowAccountSelectDialog(false);
                setSelectedAccountId("");
                setAvailableAccounts([]);
                setTempData("");
              }}
            >
              Cancelar
            </Button>
            <Button
              onClick={handleCompleteAccountSelection}
              disabled={!selectedAccountId || isCompletingSetup}
            >
              {isCompletingSetup ? "Conectando..." : "Conectar"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* AlertDialog: Confirmar Exclusao */}
      <AlertDialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              Excluir Conexao Permanentemente
            </AlertDialogTitle>
            <AlertDialogDescription>
              Tem certeza que deseja excluir esta conexao? Esta acao e
              irreversivel e ira remover permanentemente a conexao e todos os
              dados relacionados (campanhas, conjuntos de anuncios, anuncios e
              metricas).
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancelar</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDeleteConfig}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleteConfig.isPending
                ? "Excluindo..."
                : "Excluir Definitivamente"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
