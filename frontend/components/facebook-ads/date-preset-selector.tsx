"use client";

/**
 * Componente de Seletor de Periodo
 * Story 3.3: Seletor de periodo para filtrar dados
 */

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CalendarDays } from "lucide-react";
import type { DatePreset } from "@/types/facebook-ads";

interface DatePresetSelectorProps {
  value: DatePreset;
  onChange: (value: DatePreset) => void;
}

const presetOptions: { value: DatePreset; label: string }[] = [
  { value: "today", label: "Hoje" },
  { value: "yesterday", label: "Ontem" },
  { value: "last_7d", label: "Ultimos 7 dias" },
  { value: "last_14d", label: "Ultimos 14 dias" },
  { value: "last_30d", label: "Ultimos 30 dias" },
  { value: "this_month", label: "Este mes" },
  { value: "last_month", label: "Mes passado" },
  { value: "this_year", label: "Este ano" },
  { value: "last_year", label: "Ultimo ano" },
];

export function DatePresetSelector({
  value,
  onChange,
}: DatePresetSelectorProps) {
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger className="w-[180px]">
        <CalendarDays className="mr-2 h-4 w-4" />
        <SelectValue placeholder="Selecione o periodo" />
      </SelectTrigger>
      <SelectContent>
        {presetOptions.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

/**
 * Botoes rapidos de periodo
 */
interface QuickDateButtonsProps {
  value: DatePreset;
  onChange: (value: DatePreset) => void;
}

export function QuickDateButtons({ value, onChange }: QuickDateButtonsProps) {
  const quickOptions: { value: DatePreset; label: string }[] = [
    { value: "today", label: "Hoje" },
    { value: "yesterday", label: "Ontem" },
    { value: "last_7d", label: "7 dias" },
    { value: "last_30d", label: "30 dias" },
    { value: "this_month", label: "Este mes" },
  ];

  return (
    <div className="flex flex-wrap gap-2">
      {quickOptions.map((option) => (
        <Button
          key={option.value}
          variant={value === option.value ? "default" : "outline"}
          size="sm"
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </Button>
      ))}
    </div>
  );
}

/**
 * Calcula as datas de inicio e fim baseado no preset
 */
export function getDateRangeFromPreset(preset: DatePreset): {
  start: Date;
  end: Date;
} {
  const now = new Date();
  now.setHours(23, 59, 59, 999);

  let start: Date;
  const end = new Date(now);

  switch (preset) {
    case "today":
      start = new Date(now);
      start.setHours(0, 0, 0, 0);
      break;

    case "yesterday":
      start = new Date(now);
      start.setDate(start.getDate() - 1);
      start.setHours(0, 0, 0, 0);
      end.setDate(end.getDate() - 1);
      end.setHours(23, 59, 59, 999);
      break;

    case "last_7d":
      start = new Date(now);
      start.setDate(start.getDate() - 7);
      start.setHours(0, 0, 0, 0);
      end.setDate(end.getDate() - 1);
      end.setHours(23, 59, 59, 999);
      break;

    case "last_14d":
      start = new Date(now);
      start.setDate(start.getDate() - 14);
      start.setHours(0, 0, 0, 0);
      end.setDate(end.getDate() - 1);
      end.setHours(23, 59, 59, 999);
      break;

    case "last_30d":
      start = new Date(now);
      start.setDate(start.getDate() - 30);
      start.setHours(0, 0, 0, 0);
      end.setDate(end.getDate() - 1);
      end.setHours(23, 59, 59, 999);
      break;

    case "this_month":
      start = new Date(now.getFullYear(), now.getMonth(), 1);
      break;

    case "last_month":
      start = new Date(now.getFullYear(), now.getMonth() - 1, 1);
      end.setDate(0); // Ultimo dia do mes anterior
      break;

    case "this_year":
      start = new Date(now.getFullYear(), 0, 1);
      break;

    case "last_year":
      start = new Date(now.getFullYear() - 1, 0, 1);
      end.setFullYear(now.getFullYear() - 1, 11, 31);
      end.setHours(23, 59, 59, 999);
      break;

    default:
      start = new Date(now);
      start.setDate(start.getDate() - 29);
      start.setHours(0, 0, 0, 0);
  }

  return { start, end };
}

/**
 * Formata data para string ISO (YYYY-MM-DD)
 */
export function formatDateToISO(date: Date): string {
  return date.toISOString().split("T")[0];
}
