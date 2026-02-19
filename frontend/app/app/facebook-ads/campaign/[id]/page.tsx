"use client";

/**
 * Pagina de Detalhes da Campanha - Facebook Ads
 * Story 3.6: Drill-down com detalhes de uma campanha especifica
 */

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  Archive,
  ArrowLeft,
  DollarSign,
  ExternalLink,
  Pause,
  Play,
  Target,
  TrendingUp,
  Users,
} from "lucide-react";
import { useMemo, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import {
  DatePresetSelector,
  InsightsChart,
  KPICard,
  KPIGrid,
} from "@/components/facebook-ads";
import {
  useCampaignAdSets,
  useCampaignAds,
  useCampaignDetails,
  useFacebookAdsConfigs,
  useInsightsTimeSeries,
} from "@/hooks/use-facebook-ads";
import type { Ad, AdSet, DatePreset, FacebookEntityStatus } from "@/types/facebook-ads";

export default function FacebookAdsCampaignDetail() {
  const params = useParams<{ id: string }>();
  const router = useRouter();
  const campaignId = params.id || null;

  const [datePreset, setDatePreset] = useState<DatePreset>("today");
  const [activeTab, setActiveTab] = useState("overview");

  // Buscar configs para obter configId
  const { data: configs } = useFacebookAdsConfigs();
  const activeConfig = useMemo(
    () => configs?.find((c) => c.isActive) || null,
    [configs],
  );

  // Buscar dados da campanha
  const { data: campaignData, isLoading: campaignLoading } =
    useCampaignDetails(campaignId);

  // Buscar ad sets
  const { data: adsets, isLoading: adsetsLoading } =
    useCampaignAdSets(campaignId);

  // Buscar ads
  const { data: ads, isLoading: adsLoading } = useCampaignAds(campaignId);

  // Buscar serie temporal
  const { data: timeSeries, isLoading: timeSeriesLoading } =
    useInsightsTimeSeries({
      configId: activeConfig?.id,
      entityType: "campaign",
      entityId: campaignId || undefined,
      datePreset,
    });

  // Formatadores
  const formatCurrency = (value: number | null | undefined) => {
    if (value === null || value === undefined) return "\u2014";
    return new Intl.NumberFormat("pt-BR", {
      style: "currency",
      currency: "BRL",
    }).format(value);
  };

  const formatNumber = (value: number | null | undefined) => {
    if (value === null || value === undefined) return "\u2014";
    return new Intl.NumberFormat("pt-BR").format(value);
  };

  const formatPercent = (value: number | null | undefined) => {
    if (value === null || value === undefined) return "\u2014";
    return `${value.toFixed(2)}%`;
  };

  // Status badge
  const StatusBadge = ({ status }: { status: FacebookEntityStatus }) => {
    const statusConfig: Record<
      FacebookEntityStatus,
      {
        variant: "default" | "secondary" | "destructive" | "outline";
        label: string;
        icon: typeof Play | null;
      }
    > = {
      ACTIVE: { variant: "default", label: "Ativa", icon: Play },
      PAUSED: { variant: "secondary", label: "Pausada", icon: Pause },
      DELETED: { variant: "destructive", label: "Deletada", icon: null },
      ARCHIVED: { variant: "outline", label: "Arquivada", icon: Archive },
    };

    const cfg = statusConfig[status] || {
      variant: "outline" as const,
      label: status,
      icon: null,
    };
    const Icon = cfg.icon;

    return (
      <Badge variant={cfg.variant} className="gap-1">
        {Icon && <Icon className="h-3 w-3" />}
        {cfg.label}
      </Badge>
    );
  };

  if (campaignLoading) {
    return (
      <TooltipProvider>
        <div className="container mx-auto space-y-6 p-6">
          <div className="flex items-center gap-4">
            <Skeleton className="h-10 w-10" />
            <Skeleton className="h-8 w-64" />
          </div>
          <Skeleton className="h-40 w-full" />
          <Skeleton className="h-96 w-full" />
        </div>
      </TooltipProvider>
    );
  }

  if (!campaignData?.campaign) {
    return (
      <TooltipProvider>
        <div className="container mx-auto space-y-6 p-6">
          <Button
            variant="ghost"
            onClick={() => router.push("/app/facebook-ads")}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Voltar
          </Button>
          <Card>
            <CardContent className="py-12 text-center">
              <p className="text-muted-foreground">
                Campanha nao encontrada.
              </p>
            </CardContent>
          </Card>
        </div>
      </TooltipProvider>
    );
  }

  const { campaign, insights } = campaignData;

  return (
    <TooltipProvider>
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

          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <h1 className="text-2xl font-bold tracking-tight">
                  {campaign.name}
                </h1>
                <StatusBadge status={campaign.status} />
              </div>
              <div className="flex flex-wrap gap-4 text-sm text-muted-foreground">
                <span>Objetivo: {campaign.objective || "\u2014"}</span>
                {campaign.dailyBudget && (
                  <span>
                    Orcamento Diario:{" "}
                    {formatCurrency(parseFloat(campaign.dailyBudget) / 100)}
                  </span>
                )}
                {campaign.lifetimeBudget && (
                  <span>
                    Orcamento Total:{" "}
                    {formatCurrency(
                      parseFloat(campaign.lifetimeBudget) / 100,
                    )}
                  </span>
                )}
              </div>
            </div>

            <DatePresetSelector value={datePreset} onChange={setDatePreset} />
          </div>
        </div>

        {/* KPIs */}
        <KPIGrid columns={4}>
          <KPICard
            title="Gasto"
            value={insights?.totalSpend || 0}
            icon={DollarSign}
            iconColor="text-blue-500"
            format="currency"
          />
          <KPICard
            title="Impressoes"
            value={insights?.totalImpressions || 0}
            icon={Users}
            iconColor="text-purple-500"
            format="number"
          />
          <KPICard
            title="Leads"
            value={insights?.totalLeads || 0}
            icon={Target}
            iconColor="text-green-500"
            format="number"
          />
          <KPICard
            title="CPL"
            value={insights?.avgCpl || 0}
            icon={TrendingUp}
            iconColor="text-orange-500"
            format="currency"
          />
        </KPIGrid>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList>
            <TabsTrigger value="overview">Visao Geral</TabsTrigger>
            <TabsTrigger value="adsets">
              Conjuntos ({adsets?.length || 0})
            </TabsTrigger>
            <TabsTrigger value="ads">
              Anuncios ({ads?.length || 0})
            </TabsTrigger>
          </TabsList>

          {/* Tab: Visao Geral */}
          <TabsContent value="overview" className="space-y-6">
            <InsightsChart
              data={timeSeries || []}
              isLoading={timeSeriesLoading}
              title="Performance da Campanha"
            />

            {/* Metricas adicionais */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    CTR
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">
                    {formatPercent(insights?.avgCtr)}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    CPC
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">
                    {formatCurrency(insights?.avgCpc)}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Cliques
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">
                    {formatNumber(insights?.totalClicks)}
                  </p>
                </CardContent>
              </Card>
              <Card>
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm font-medium text-muted-foreground">
                    Alcance
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">
                    {formatNumber(insights?.totalReach)}
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          {/* Tab: Ad Sets */}
          <TabsContent value="adsets">
            <Card>
              <CardContent className="pt-6">
                {adsetsLoading ? (
                  <div className="space-y-2">
                    {[...Array(3)].map((_, i) => (
                      <Skeleton key={i} className="h-12 w-full" />
                    ))}
                  </div>
                ) : !adsets || adsets.length === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    Nenhum conjunto de anuncios encontrado.
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Nome</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Otimizacao</TableHead>
                        <TableHead className="text-right">
                          Orcamento
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {adsets.map((adset: AdSet) => (
                        <TableRow key={adset.id}>
                          <TableCell className="font-medium">
                            {adset.name}
                          </TableCell>
                          <TableCell>
                            <StatusBadge status={adset.status} />
                          </TableCell>
                          <TableCell>
                            {adset.optimizationGoal || "\u2014"}
                          </TableCell>
                          <TableCell className="text-right">
                            {adset.dailyBudget
                              ? formatCurrency(
                                  parseFloat(adset.dailyBudget) / 100,
                                ) + "/dia"
                              : adset.lifetimeBudget
                                ? formatCurrency(
                                    parseFloat(adset.lifetimeBudget) / 100,
                                  ) + " total"
                                : "\u2014"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* Tab: Ads */}
          <TabsContent value="ads">
            <Card>
              <CardContent className="pt-6">
                {adsLoading ? (
                  <div className="space-y-2">
                    {[...Array(3)].map((_, i) => (
                      <Skeleton key={i} className="h-12 w-full" />
                    ))}
                  </div>
                ) : !ads || ads.length === 0 ? (
                  <div className="py-8 text-center text-muted-foreground">
                    Nenhum anuncio encontrado.
                  </div>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Nome</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Preview</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {ads.map((ad: Ad) => (
                        <TableRow key={ad.id}>
                          <TableCell className="font-medium">
                            {ad.name}
                          </TableCell>
                          <TableCell>
                            <StatusBadge status={ad.status} />
                          </TableCell>
                          <TableCell>
                            {ad.previewShareableLink ? (
                              <Button variant="ghost" size="sm" asChild>
                                <a
                                  href={ad.previewShareableLink}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                >
                                  <ExternalLink className="mr-1 h-4 w-4" />
                                  Ver
                                </a>
                              </Button>
                            ) : (
                              "\u2014"
                            )}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </TooltipProvider>
  );
}
