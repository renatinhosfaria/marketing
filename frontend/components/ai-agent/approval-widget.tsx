"use client";

import { useState } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { AlertTriangle, Check, X } from "lucide-react";
import type { InterruptPayload } from "@/types/ai-agent";

interface ApprovalWidgetProps {
  interrupt: InterruptPayload;
  onApprove: (overrideValue?: string) => void;
  onReject: () => void;
  disabled?: boolean;
  className?: string;
}

/**
 * Card amarelo para aprovacao/rejeicao de acoes do agente.
 * Exibe detalhes da acao, input para override e botoes Aprovar/Rejeitar.
 */
export function ApprovalWidget({
  interrupt,
  onApprove,
  onReject,
  disabled = false,
  className,
}: ApprovalWidgetProps) {
  const [overrideValue, setOverrideValue] = useState("");

  return (
    <div
      className={cn(
        "mx-4 rounded-xl border-2 border-amber-300 bg-amber-50 p-4 space-y-3",
        className
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-2">
        <AlertTriangle className="h-5 w-5 text-amber-600 shrink-0" />
        <h4 className="text-sm font-semibold text-amber-900">
          Aprovacao Necessaria
        </h4>
      </div>

      {/* Tipo da acao */}
      <div className="rounded-lg bg-amber-100/60 p-3">
        <p className="text-xs font-medium text-amber-800 uppercase tracking-wide mb-1">
          Tipo da Acao
        </p>
        <p className="text-sm text-amber-900 font-medium">{interrupt.type}</p>
      </div>

      {/* Detalhes */}
      {Object.keys(interrupt.details).length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-amber-800 uppercase tracking-wide">
            Detalhes
          </p>
          <div className="rounded-lg bg-white/60 p-3 space-y-1.5">
            {Object.entries(interrupt.details).map(([key, value]) => (
              <div key={key} className="flex items-start gap-2 text-sm">
                <span className="font-medium text-amber-800 min-w-[100px]">
                  {key}:
                </span>
                <span className="text-amber-900 break-all">
                  {typeof value === "object"
                    ? JSON.stringify(value, null, 2)
                    : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Input para override */}
      <div className="space-y-1.5">
        <label className="text-xs font-medium text-amber-800">
          Valor alternativo (opcional)
        </label>
        <Input
          value={overrideValue}
          onChange={(e) => setOverrideValue(e.target.value)}
          placeholder="Insira um valor para substituir, se necessario..."
          disabled={disabled}
          className="bg-white/80 border-amber-300 text-sm"
        />
      </div>

      {/* Botoes */}
      <div className="flex gap-2 pt-1">
        <Button
          onClick={() => onApprove(overrideValue || undefined)}
          disabled={disabled}
          className="flex-1 bg-green-600 hover:bg-green-700 text-white"
          size="sm"
        >
          <Check className="h-4 w-4" />
          Aprovar
        </Button>
        <Button
          onClick={onReject}
          disabled={disabled}
          variant="outline"
          className="flex-1 border-red-300 text-red-600 hover:bg-red-50 hover:text-red-700"
          size="sm"
        >
          <X className="h-4 w-4" />
          Rejeitar
        </Button>
      </div>
    </div>
  );
}
