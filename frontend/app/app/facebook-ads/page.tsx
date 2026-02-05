"use client";

/**
 * Pagina Principal do Dashboard - Facebook Ads
 * Story 3.3: Dashboard com KPIs, grafico e lista de campanhas
 * Atualizado: Sistema de abas para Campanhas, Conjuntos e Anuncios
 */

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  DollarSign,
  Image,
  Layers,
  Link as LinkIcon,
  Megaphone,
  MousePointerClick,
  Settings,
  Target,
  TrendingUp,
  Users,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AdSetTable,
  AdTable,
  CampaignTable,
  DatePresetSelector,
  KPICard,
  KPIGrid,
  SyncStatusBar,
} from "@/components/facebook-ads";
import {
  useAdSets,
  useAds,
  useCampaigns,
  useFacebookAdsConfigs,
  useInsightsSummary,
} from "@/hooks/use-facebook-ads";
import type { DatePreset, InsightsTableSource } from "@/types/facebook-ads";

type TabValue = "campaigns" | "adsets" | "ads";

export default function FacebookAdsDashboard() {
  const router = useRouter();
  // Estado do filtro de periodo
  const [datePreset, setDatePreset] = useState<DatePreset>("last_30d");
  const [activeTab, setActiveTab] = useState<TabValue>("campaigns");

  // Estados de filtro por aba
  const [campaignStatusFilter, setCampaignStatusFilter] = useState<
    string | null
  >(null);
  const [adSetStatusFilter, setAdSetStatusFilter] = useState<string | null>(
    null,
  );
  const [adStatusFilter, setAdStatusFilter] = useState<string | null>(null);

  // Determinar qual tabela de insights usar
  const insightsTable: InsightsTableSource = useMemo(() => {
    if (datePreset === "today") return "today";
    if (
      [
        "last_7d",
        "last_14d",
        "last_30d",
        "last_90d",
        "this_month",
        "this_year",
        "last_year",
      ].includes(datePreset)
    ) {
      return "both";
    }
    return "history";
  }, [datePreset]);

  // Buscar configuracoes
  const { data: configs, isLoading: configsLoading } =
    useFacebookAdsConfigs();

  // Usar a primeira config ativa
  const activeConfig = useMemo(
    () => configs?.find((c) => c.isActive) || null,
    [configs],
  );

  // Buscar dados de resumo
  const {
    data: summary,
    isLoading: summaryLoading,
  } = useInsightsSummary({
    configId: activeConfig?.id,
    datePreset,
    insightsTable,
  });

  // Buscar campanhas
  const { data: campaignsData, isLoading: campaignsLoading } = useCampaigns({
    configId: activeConfig?.id,
    status: campaignStatusFilter || undefined,
    limit: 50,
    datePreset,
    insightsTable,
  });

  // Buscar conjuntos de anuncios
  const { data: adSetsData, isLoading: adSetsLoading } = useAdSets({
    configId: activeConfig?.id,
    status: adSetStatusFilter || undefined,
    limit: 50,
    datePreset,
    insightsTable,
  });

  // Buscar anuncios
  const { data: adsData, isLoading: adsLoading } = useAds({
    configId: activeConfig?.id,
    status: adStatusFilter || undefined,
    limit: 50,
    datePreset,
    insightsTable,
  });

  // Loading geral
  const isLoading = configsLoading || summaryLoading;

  // Contagens para badges das abas
  const campaignsCount = campaignsData?.data?.length || 0;
  const adSetsCount = adSetsData?.data?.length || 0;
  const adsCount = adsData?.data?.length || 0;

  // Sem configuracao
  if (!configsLoading && (!configs || configs.length === 0)) {
    return (
      <TooltipProvider>
        <div className="container mx-auto space-y-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                Facebook Ads
              </h1>
              <p className="text-muted-foreground">
                Analise de desempenho de campanhas
              </p>
            </div>
            <Button
              onClick={() => router.push("/app/facebook-ads/settings")}
            >
              <Settings className="mr-2 h-4 w-4" />
              Configuracoes
            </Button>
          </div>

          <div className="flex flex-col items-center justify-center rounded-lg border border-dashed p-12">
            <LinkIcon className="mb-4 h-12 w-12 text-muted-foreground" />
            <h3 className="mb-2 text-lg font-semibold">
              Nenhuma conta conectada
            </h3>
            <p className="mb-6 max-w-md text-center text-muted-foreground">
              Conecte sua conta do Facebook Ads para visualizar metricas de
              campanhas, gastos, leads e muito mais.
            </p>
            <Button
              onClick={() => router.push("/app/facebook-ads/settings")}
            >
              <LinkIcon className="mr-2 h-4 w-4" />
              Conectar Conta do Facebook
            </Button>
          </div>
        </div>
      </TooltipProvider>
    );
  }

  // Sem conta ativa
  if (!configsLoading && !activeConfig) {
    return (
      <TooltipProvider>
        <div className="container mx-auto space-y-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">
                Facebook Ads
              </h1>
              <p className="text-muted-foreground">
                Analise de desempenho de campanhas
              </p>
            </div>
            <Button
              onClick={() => router.push("/app/facebook-ads/settings")}
            >
              <Settings className="mr-2 h-4 w-4" />
              Configuracoes
            </Button>
          </div>

          <Alert variant="destructive">
            <AlertTitle>Nenhuma conta ativa</AlertTitle>
            <AlertDescription>
              Todas as contas estao desativadas. Ative uma conta nas
              configuracoes para visualizar as metricas.
            </AlertDescription>
          </Alert>
        </div>
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider>
      <div className="container mx-auto space-y-6 p-6">
        {/* Header */}
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Facebook Ads</h1>
            <p className="text-muted-foreground">
              {activeConfig?.accountName ||
                "Analise de desempenho de campanhas"}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <DatePresetSelector value={datePreset} onChange={setDatePreset} />
            <Button
              variant="outline"
              size="icon"
              onClick={() => router.push("/app/facebook-ads/settings")}
              title="Configuracoes"
              aria-label="Abrir configuracoes"
            >
              <Settings className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Barra de sincronizacao */}
        {activeConfig && <SyncStatusBar configId={activeConfig.id} />}

        {/* KPIs */}
        <KPIGrid columns={5}>
          <KPICard
            title="Gasto Total"
            value={summary?.totalSpend || 0}
            change={summary?.comparison?.spendChange}
            icon={DollarSign}
            iconColor="text-blue-500"
            format="currency"
            tooltip="Total investido no periodo selecionado"
            isLoading={isLoading}
          />
          <KPICard
            title="Leads"
            value={summary?.totalLeads || 0}
            change={summary?.comparison?.leadsChange}
            icon={Target}
            iconColor="text-green-500"
            format="number"
            tooltip="Total de leads gerados no periodo"
            isLoading={isLoading}
          />
          <KPICard
            title="CPL"
            value={summary?.avgCpl || 0}
            change={summary?.comparison?.cplChange}
            icon={Users}
            iconColor="text-orange-500"
            format="currency"
            tooltip="Custo por Lead - valor medio gasto para gerar cada lead"
            isLoading={isLoading}
          />
          <KPICard
            title="CPM"
            value={summary?.avgCpm || 0}
            change={summary?.comparison?.cpmChange}
            icon={TrendingUp}
            iconColor="text-purple-500"
            format="currency"
            tooltip="Custo por Mil Impressoes - valor gasto a cada 1000 exibicoes do anuncio"
            isLoading={isLoading}
          />
          <KPICard
            title="CPC"
            value={summary?.avgCpc || 0}
            change={summary?.comparison?.cpcChange}
            icon={MousePointerClick}
            iconColor="text-cyan-500"
            format="currency"
            tooltip="Custo por Clique - valor medio gasto por cada clique"
            isLoading={isLoading}
          />
        </KPIGrid>

        {/* Sistema de Abas */}
        <Tabs
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as TabValue)}
          className="space-y-4"
        >
          <TabsList className="grid w-full grid-cols-3 lg:w-[400px]">
            <TabsTrigger value="campaigns" className="flex items-center gap-2">
              <Megaphone className="h-4 w-4" />
              <span className="hidden sm:inline">Campanhas</span>
              <span className="sm:hidden">Camp.</span>
              {campaignsCount > 0 && (
                <span className="ml-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium">
                  {campaignsCount}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="adsets" className="flex items-center gap-2">
              <Layers className="h-4 w-4" />
              <span className="hidden sm:inline">Conjuntos</span>
              <span className="sm:hidden">Conj.</span>
              {adSetsCount > 0 && (
                <span className="ml-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium">
                  {adSetsCount}
                </span>
              )}
            </TabsTrigger>
            <TabsTrigger value="ads" className="flex items-center gap-2">
              <Image className="h-4 w-4" />
              <span className="hidden sm:inline">Anuncios</span>
              <span className="sm:hidden">Anun.</span>
              {adsCount > 0 && (
                <span className="ml-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium">
                  {adsCount}
                </span>
              )}
            </TabsTrigger>
          </TabsList>

          {/* Conteudo da aba Campanhas */}
          <TabsContent value="campaigns" className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Campanhas</h2>
              <p className="text-sm text-muted-foreground">
                Visualize o desempenho das suas campanhas de marketing
              </p>
            </div>
            <CampaignTable
              campaigns={campaignsData?.data || []}
              isLoading={campaignsLoading}
              statusFilter={campaignStatusFilter}
              onStatusFilter={setCampaignStatusFilter}
            />
          </TabsContent>

          {/* Conteudo da aba Conjuntos */}
          <TabsContent value="adsets" className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Conjuntos de Anuncios</h2>
              <p className="text-sm text-muted-foreground">
                Analise a performance por segmentacao de publico
              </p>
            </div>
            <AdSetTable
              adSets={adSetsData?.data || []}
              isLoading={adSetsLoading}
              statusFilter={adSetStatusFilter}
              onStatusFilter={setAdSetStatusFilter}
            />
          </TabsContent>

          {/* Conteudo da aba Anuncios */}
          <TabsContent value="ads" className="space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold">Anuncios</h2>
              <p className="text-sm text-muted-foreground">
                Acompanhe o desempenho individual de cada criativo
              </p>
            </div>
            <AdTable
              ads={adsData?.data || []}
              isLoading={adsLoading}
              statusFilter={adStatusFilter}
              onStatusFilter={setAdStatusFilter}
            />
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
}
