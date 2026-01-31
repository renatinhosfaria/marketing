"use client";

/**
 * Componente de KPI Card
 * Story 3.2: Card reutilizavel para metricas principais
 */

import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import type { LucideIcon } from "lucide-react";
import { HelpCircle, Minus, TrendingDown, TrendingUp } from "lucide-react";

interface KPICardProps {
  title: string;
  value: string | number;
  change?: number;
  changeLabel?: string;
  icon?: LucideIcon;
  iconColor?: string;
  tooltip?: string;
  isLoading?: boolean;
  format?: "currency" | "number" | "percent" | "none";
}

export function KPICard({
  title,
  value,
  change,
  changeLabel = "vs. periodo anterior",
  icon: Icon,
  iconColor = "text-primary",
  tooltip,
  isLoading = false,
  format = "none",
}: KPICardProps) {
  // Formatar valor
  const formatValue = (val: string | number | null | undefined): string => {
    if (val == null) {
      return format === "currency" ? "R$ 0,00" : "0";
    }
    if (typeof val === "string") return val;

    switch (format) {
      case "currency":
        return new Intl.NumberFormat("pt-BR", {
          style: "currency",
          currency: "BRL",
          minimumFractionDigits: 2,
          maximumFractionDigits: 2,
        }).format(val);
      case "number":
        return new Intl.NumberFormat("pt-BR").format(val);
      case "percent":
        return `${val.toFixed(2)}%`;
      default:
        return String(val);
    }
  };

  // Determinar cor e icone do change
  const getChangeInfo = () => {
    if (change === undefined || change === null) {
      return { color: "text-muted-foreground", Icon: Minus };
    }
    if (change > 0) {
      return { color: "text-green-600", Icon: TrendingUp };
    }
    if (change < 0) {
      return { color: "text-red-600", Icon: TrendingDown };
    }
    return { color: "text-muted-foreground", Icon: Minus };
  };

  const changeInfo = getChangeInfo();

  if (isLoading) {
    return (
      <Card>
        <CardContent className="p-6">
          <div className="flex items-center justify-between">
            <Skeleton className="h-4 w-24" />
            <Skeleton className="h-8 w-8 rounded-full" />
          </div>
          <Skeleton className="mt-3 h-8 w-32" />
          <Skeleton className="mt-2 h-4 w-20" />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-muted-foreground">
              {title}
            </span>
            {tooltip && (
              <Tooltip>
                <TooltipTrigger>
                  <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50" />
                </TooltipTrigger>
                <TooltipContent>
                  <p className="max-w-xs text-sm">{tooltip}</p>
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          {Icon && (
            <div
              className={cn(
                "flex h-10 w-10 items-center justify-center rounded-full bg-primary/10",
                iconColor
              )}
            >
              <Icon className="h-5 w-5" />
            </div>
          )}
        </div>

        {/* Valor */}
        <div className="mt-3">
          <span className="text-2xl font-bold tracking-tight">
            {formatValue(value)}
          </span>
        </div>

        {/* Change */}
        {change !== undefined && (
          <div className="mt-2 flex items-center gap-1.5">
            <changeInfo.Icon className={cn("h-4 w-4", changeInfo.color)} />
            <span className={cn("text-sm font-medium", changeInfo.color)}>
              {change > 0 ? "+" : ""}
              {change.toFixed(1)}%
            </span>
            <span className="text-xs text-muted-foreground">{changeLabel}</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
