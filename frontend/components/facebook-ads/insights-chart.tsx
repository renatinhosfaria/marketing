"use client";

/**
 * Componente de Grafico de Insights
 * Story 3.5: Grafico de tendencia com Recharts
 */

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import { useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TimeSeriesDataPoint } from "@/types/facebook-ads";

interface InsightsChartProps {
  data: TimeSeriesDataPoint[];
  isLoading?: boolean;
  title?: string;
}

type MetricKey =
  | "spend"
  | "impressions"
  | "clicks"
  | "leads"
  | "ctr"
  | "cpc"
  | "cpl";

interface MetricConfig {
  key: MetricKey;
  label: string;
  color: string;
  format: (value: number) => string;
  yAxisId: "left" | "right";
}

const metrics: MetricConfig[] = [
  {
    key: "spend",
    label: "Gasto",
    color: "#3B82F6", // blue-500
    format: (v) =>
      v != null
        ? new Intl.NumberFormat("pt-BR", {
            style: "currency",
            currency: "BRL",
            minimumFractionDigits: 0,
            maximumFractionDigits: 0,
          }).format(v)
        : "R$ 0",
    yAxisId: "left",
  },
  {
    key: "leads",
    label: "Leads",
    color: "#22C55E", // green-500
    format: (v) => (v != null ? new Intl.NumberFormat("pt-BR").format(v) : "0"),
    yAxisId: "right",
  },
  {
    key: "clicks",
    label: "Cliques",
    color: "#8B5CF6", // violet-500
    format: (v) => (v != null ? new Intl.NumberFormat("pt-BR").format(v) : "0"),
    yAxisId: "right",
  },
  {
    key: "impressions",
    label: "Impressoes",
    color: "#F59E0B", // amber-500
    format: (v) =>
      v != null
        ? new Intl.NumberFormat("pt-BR", { notation: "compact" }).format(v)
        : "0",
    yAxisId: "right",
  },
  {
    key: "ctr",
    label: "CTR",
    color: "#EC4899", // pink-500
    format: (v) => (v != null ? `${v.toFixed(2)}%` : "0%"),
    yAxisId: "right",
  },
  {
    key: "cpc",
    label: "CPC",
    color: "#14B8A6", // teal-500
    format: (v) =>
      v != null
        ? new Intl.NumberFormat("pt-BR", {
            style: "currency",
            currency: "BRL",
          }).format(v)
        : "R$ 0,00",
    yAxisId: "left",
  },
  {
    key: "cpl",
    label: "CPL",
    color: "#F97316", // orange-500
    format: (v) =>
      v != null
        ? new Intl.NumberFormat("pt-BR", {
            style: "currency",
            currency: "BRL",
          }).format(v)
        : "R$ 0,00",
    yAxisId: "left",
  },
];

export function InsightsChart({
  data,
  isLoading = false,
  title = "Tendencia de Metricas",
}: InsightsChartProps) {
  // Metricas visiveis (default: spend e leads)
  const [visibleMetrics, setVisibleMetrics] = useState<Set<MetricKey>>(
    new Set(["spend", "leads"])
  );

  // Toggle metrica
  const toggleMetric = (key: MetricKey) => {
    setVisibleMetrics((prev) => {
      const next = new Set(prev);
      if (next.has(key)) {
        // Nao permitir desmarcar todas
        if (next.size > 1) {
          next.delete(key);
        }
      } else {
        next.add(key);
      }
      return next;
    });
  };

  // Formatar data para exibicao
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString("pt-BR", { day: "2-digit", month: "short" });
  };

  // Custom Tooltip
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || !payload.length) return null;

    const date = new Date(label);
    const formattedDate = date.toLocaleDateString("pt-BR", {
      weekday: "long",
      day: "numeric",
      month: "long",
    });

    return (
      <div className="rounded-lg border bg-popover p-3 shadow-lg">
        <p className="mb-2 font-medium capitalize">{formattedDate}</p>
        <div className="space-y-1">
          {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
          {payload.map((entry: any) => {
            const metric = metrics.find((m) => m.key === entry.dataKey);
            if (!metric) return null;

            return (
              <div
                key={entry.dataKey}
                className="flex items-center justify-between gap-4 text-sm"
              >
                <div className="flex items-center gap-2">
                  <div
                    className="h-2.5 w-2.5 rounded-full"
                    style={{ backgroundColor: entry.color }}
                  />
                  <span className="text-muted-foreground">{metric.label}:</span>
                </div>
                <span className="font-medium">
                  {metric.format(entry.value)}
                </span>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Custom Legend
  const CustomLegend = () => (
    <div className="flex flex-wrap justify-center gap-2 pt-4">
      {metrics.map((metric) => (
        <Button
          key={metric.key}
          variant="ghost"
          size="sm"
          className={cn(
            "h-7 px-2 text-xs",
            !visibleMetrics.has(metric.key) && "opacity-50"
          )}
          onClick={() => toggleMetric(metric.key)}
        >
          <div
            className="mr-1.5 h-2 w-2 rounded-full"
            style={{ backgroundColor: metric.color }}
          />
          {metric.label}
        </Button>
      ))}
    </div>
  );

  // Verificar se ha metricas no eixo esquerdo/direito
  const hasLeftAxis = Array.from(visibleMetrics).some(
    (key) => metrics.find((m) => m.key === key)?.yAxisId === "left"
  );
  const hasRightAxis = Array.from(visibleMetrics).some(
    (key) => metrics.find((m) => m.key === key)?.yAxisId === "right"
  );

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-6 w-40" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-[300px] w-full" />
        </CardContent>
      </Card>
    );
  }

  if (!data || data.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex h-[300px] items-center justify-center text-muted-foreground">
            Nenhum dado disponivel para o periodo selecionado.
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart
              data={data}
              margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
            >
              <defs>
                {metrics.map((metric) => (
                  <linearGradient
                    key={metric.key}
                    id={`gradient-${metric.key}`}
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="5%"
                      stopColor={metric.color}
                      stopOpacity={0.2}
                    />
                    <stop
                      offset="95%"
                      stopColor={metric.color}
                      stopOpacity={0}
                    />
                  </linearGradient>
                ))}
              </defs>

              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />

              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                tick={{ fontSize: 12 }}
                tickLine={false}
                axisLine={false}
              />

              {hasLeftAxis && (
                <YAxis
                  yAxisId="left"
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) =>
                    new Intl.NumberFormat("pt-BR", {
                      notation: "compact",
                      compactDisplay: "short",
                    }).format(v)
                  }
                />
              )}

              {hasRightAxis && (
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  tick={{ fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(v) =>
                    new Intl.NumberFormat("pt-BR", {
                      notation: "compact",
                      compactDisplay: "short",
                    }).format(v)
                  }
                />
              )}

              <Tooltip content={<CustomTooltip />} />

              {metrics.map(
                (metric) =>
                  visibleMetrics.has(metric.key) && (
                    <Area
                      key={metric.key}
                      type="monotone"
                      dataKey={metric.key}
                      yAxisId={metric.yAxisId}
                      stroke={metric.color}
                      strokeWidth={2}
                      fill={`url(#gradient-${metric.key})`}
                      dot={false}
                      activeDot={{ r: 4, strokeWidth: 0 }}
                    />
                  )
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <CustomLegend />
      </CardContent>
    </Card>
  );
}
