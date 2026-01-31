"use client";

/**
 * Componente de Tabela de Anuncios (Ads)
 * Story 3.4: Lista ads com filtros, ordenacao e navegacao
 */

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Search,
} from "lucide-react";
import { useState } from "react";
import type { AdWithMetrics, FacebookEntityStatus } from "@/types/facebook-ads";

interface AdTableProps {
  ads: AdWithMetrics[];
  isLoading?: boolean;
  onStatusFilter?: (status: string | null) => void;
  statusFilter?: string | null;
}

type SortField = "name" | "status" | "spend" | "leads" | "cpl" | "ctr";
type SortOrder = "asc" | "desc";

export function AdTable({
  ads,
  isLoading = false,
  onStatusFilter,
  statusFilter,
}: AdTableProps) {
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<SortField>("spend");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  // Filtrar por busca
  const filteredAds = ads.filter((ad) =>
    ad.name.toLowerCase().includes(search.toLowerCase())
  );

  // Ordenar
  const sortedAds = [...filteredAds].sort((a, b) => {
    let valueA: string | number;
    let valueB: string | number;

    switch (sortField) {
      case "name":
        valueA = a.name.toLowerCase();
        valueB = b.name.toLowerCase();
        break;
      case "status":
        valueA = a.status;
        valueB = b.status;
        break;
      case "spend":
        valueA = a.spend;
        valueB = b.spend;
        break;
      case "leads":
        valueA = a.leads;
        valueB = b.leads;
        break;
      case "cpl":
        valueA = a.cpl ?? Infinity;
        valueB = b.cpl ?? Infinity;
        break;
      case "ctr":
        valueA = a.ctr;
        valueB = b.ctr;
        break;
      default:
        return 0;
    }

    if (valueA < valueB) return sortOrder === "asc" ? -1 : 1;
    if (valueA > valueB) return sortOrder === "asc" ? 1 : -1;
    return 0;
  });

  // Toggle sort
  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortOrder(sortOrder === "asc" ? "desc" : "asc");
    } else {
      setSortField(field);
      setSortOrder("desc");
    }
  };

  // Renderizar icone de ordenacao
  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) {
      return <ArrowUpDown className="ml-1 h-4 w-4 text-muted-foreground/50" />;
    }
    return sortOrder === "asc" ? (
      <ChevronUp className="ml-1 h-4 w-4" />
    ) : (
      <ChevronDown className="ml-1 h-4 w-4" />
    );
  };

  // Formatar valores
  const formatCurrency = (value: number | null | undefined) =>
    value != null
      ? new Intl.NumberFormat("pt-BR", {
          style: "currency",
          currency: "BRL",
        }).format(value)
      : "R$ 0,00";

  const formatNumber = (value: number | null | undefined) =>
    value != null ? new Intl.NumberFormat("pt-BR").format(value) : "0";

  const formatPercent = (value: number | null | undefined) =>
    value != null ? `${value.toFixed(2)}%` : "0,00%";

  // Badge de status
  const StatusBadge = ({ status }: { status: FacebookEntityStatus }) => {
    const variants: Record<
      FacebookEntityStatus,
      { variant: "default" | "secondary" | "destructive" | "outline"; label: string }
    > = {
      ACTIVE: { variant: "default", label: "Ativo" },
      PAUSED: { variant: "secondary", label: "Pausado" },
      DELETED: { variant: "destructive", label: "Deletado" },
      ARCHIVED: { variant: "outline", label: "Arquivado" },
    };

    const config = variants[status] || { variant: "outline" as const, label: status };

    return <Badge variant={config.variant}>{config.label}</Badge>;
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex gap-4">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-10 w-40" />
        </div>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                {["Nome", "Status", "Gasto", "Leads", "CPL", "CTR", ""].map(
                  (_, i) => (
                    <TableHead key={i}>
                      <Skeleton className="h-4 w-20" />
                    </TableHead>
                  )
                )}
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...Array(5)].map((_, i) => (
                <TableRow key={i}>
                  {[...Array(7)].map((_, j) => (
                    <TableCell key={j}>
                      <Skeleton className="h-4 w-full" />
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Filtros */}
      <div className="flex flex-wrap gap-4">
        <div className="relative flex-1 min-w-[200px] max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Buscar anuncio..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>

        {onStatusFilter && (
          <Select
            value={statusFilter || "all"}
            onValueChange={(v) => onStatusFilter(v === "all" ? null : v)}
          >
            <SelectTrigger className="w-[150px]">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">Todos os status</SelectItem>
              <SelectItem value="ACTIVE">Ativo</SelectItem>
              <SelectItem value="PAUSED">Pausado</SelectItem>
              <SelectItem value="ARCHIVED">Arquivado</SelectItem>
            </SelectContent>
          </Select>
        )}
      </div>

      {/* Tabela */}
      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => handleSort("name")}
              >
                <div className="flex items-center">
                  Nome
                  <SortIcon field="name" />
                </div>
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => handleSort("status")}
              >
                <div className="flex items-center">
                  Status
                  <SortIcon field="status" />
                </div>
              </TableHead>
              <TableHead
                className="cursor-pointer text-right hover:bg-muted/50"
                onClick={() => handleSort("spend")}
              >
                <div className="flex items-center justify-end">
                  Gasto
                  <SortIcon field="spend" />
                </div>
              </TableHead>
              <TableHead
                className="cursor-pointer text-right hover:bg-muted/50"
                onClick={() => handleSort("leads")}
              >
                <div className="flex items-center justify-end">
                  Leads
                  <SortIcon field="leads" />
                </div>
              </TableHead>
              <TableHead
                className="cursor-pointer text-right hover:bg-muted/50"
                onClick={() => handleSort("cpl")}
              >
                <div className="flex items-center justify-end">
                  CPL
                  <SortIcon field="cpl" />
                </div>
              </TableHead>
              <TableHead
                className="cursor-pointer text-right hover:bg-muted/50"
                onClick={() => handleSort("ctr")}
              >
                <div className="flex items-center justify-end">
                  CTR
                  <SortIcon field="ctr" />
                </div>
              </TableHead>
              <TableHead className="w-10"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedAds.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="h-24 text-center">
                  {search
                    ? "Nenhum anuncio encontrado para a busca."
                    : "Nenhum anuncio disponivel."}
                </TableCell>
              </TableRow>
            ) : (
              sortedAds.map((ad) => (
                <TableRow key={ad.id} className="hover:bg-muted/50">
                  <TableCell className="font-medium max-w-[300px] truncate">
                    {ad.name}
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={ad.status} />
                  </TableCell>
                  <TableCell className="text-right font-medium">
                    {formatCurrency(ad.spend)}
                  </TableCell>
                  <TableCell className="text-right">
                    {formatNumber(ad.leads)}
                  </TableCell>
                  <TableCell className="text-right">
                    {ad.cpl ? formatCurrency(ad.cpl) : "\u2014"}
                  </TableCell>
                  <TableCell className="text-right">
                    {formatPercent(ad.ctr)}
                  </TableCell>
                  <TableCell>
                    {ad.previewShareableLink && (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8"
                            onClick={(e) => {
                              e.stopPropagation();
                              window.open(ad.previewShareableLink!, "_blank");
                            }}
                          >
                            <ExternalLink className="h-4 w-4 text-muted-foreground" />
                          </Button>
                        </TooltipTrigger>
                        <TooltipContent>
                          <p>Ver previa do anuncio</p>
                        </TooltipContent>
                      </Tooltip>
                    )}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Resumo */}
      <div className="text-sm text-muted-foreground">
        {sortedAds.length} anuncio(s)
        {search && ` encontrado(s) para "${search}"`}
      </div>
    </div>
  );
}
