"use client";

/**
 * Grid responsivo para KPI Cards
 * 2 colunas mobile, 3 tablet, 4/6 desktop
 */

import { cn } from "@/lib/utils";

interface KPIGridProps {
  children: React.ReactNode;
  columns?: 2 | 3 | 4 | 5 | 6;
}

export function KPIGrid({ children, columns = 4 }: KPIGridProps) {
  const gridCols = {
    2: "grid-cols-1 sm:grid-cols-2",
    3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
    5: "grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5",
    6: "grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6",
  };

  return <div className={cn("grid gap-4", gridCols[columns])}>{children}</div>;
}
