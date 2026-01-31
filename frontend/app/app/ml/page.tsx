"use client";

/**
 * Dashboard principal de ML para Facebook Ads
 */

import { useState, useEffect } from "react";
import { Brain, Settings, ChevronLeft } from "lucide-react";
import { useFacebookAdsConfigs } from "@/hooks/use-facebook-ads";
import { MLStatusBadge } from "@/components/ml/ml-status-badge";
import { ClassificationCard } from "@/components/ml/classification-card";
import { RecommendationsCard } from "@/components/ml/recommendations-card";
import { RecommendationsList } from "@/components/ml/recommendations-list";
import { AnomaliesCard } from "@/components/ml/anomalies-card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import Link from "next/link";

export default function MLDashboardPage() {
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null);
  const [showRecommendations, setShowRecommendations] = useState(false);

  // Buscar configuracoes do Facebook Ads
  const { data: configs, isLoading: loadingConfigs } = useFacebookAdsConfigs();

  // Auto-selecionar primeira config ativa
  useEffect(() => {
    if (!selectedConfigId && configs?.length) {
      const activeConfig = configs.find((c) => c.isActive);
      if (activeConfig) {
        setSelectedConfigId(activeConfig.id);
      }
    }
  }, [selectedConfigId, configs]);

  if (loadingConfigs) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Brain className="h-12 w-12 text-indigo-600 mx-auto mb-4 animate-pulse" />
          <p className="text-gray-600">Carregando dashboard ML...</p>
        </div>
      </div>
    );
  }

  if (!configs?.length) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md">
          <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">
            Nenhuma conta configurada
          </h2>
          <p className="text-gray-600 mb-4">
            Configure uma conta do Facebook Ads para usar as funcionalidades de
            ML.
          </p>
          <Button asChild>
            <Link
              href="/app/facebook-ads/settings"
              className="inline-flex items-center gap-2"
            >
              <Settings className="h-4 w-4" />
              Configurar Facebook Ads
            </Link>
          </Button>
        </div>
      </div>
    );
  }

  // Tela de lista de recomendacoes
  if (showRecommendations && selectedConfigId) {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-4xl mx-auto px-4 py-8">
          <button
            onClick={() => setShowRecommendations(false)}
            className="flex items-center gap-1 text-indigo-600 hover:text-indigo-700 mb-6"
          >
            <ChevronLeft className="h-4 w-4" />
            Voltar ao Dashboard
          </button>

          <div className="flex items-center justify-between mb-6">
            <h1 className="text-2xl font-bold text-gray-900">
              Recomendações de Otimização
            </h1>
            <MLStatusBadge />
          </div>

          <RecommendationsList configId={selectedConfigId} limit={50} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center gap-3">
            <Brain className="h-8 w-8 text-indigo-600" />
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Machine Learning
              </h1>
              <p className="text-gray-500">
                Análise inteligente de campanhas
              </p>
            </div>
          </div>
          <MLStatusBadge />
        </div>

        {/* Seletor de conta */}
        {configs.length > 1 && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Conta de Anúncios
            </label>
            <Select
              value={selectedConfigId ? String(selectedConfigId) : undefined}
              onValueChange={(value) => setSelectedConfigId(Number(value))}
            >
              <SelectTrigger className="w-[320px]">
                <SelectValue placeholder="Selecione uma conta" />
              </SelectTrigger>
              <SelectContent>
                {configs.map((config) => (
                  <SelectItem key={config.id} value={String(config.id)}>
                    {config.accountName} ({config.accountId})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        )}

        {selectedConfigId && (
          <>
            {/* Cards de resumo */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
              <ClassificationCard configId={selectedConfigId} />
              <RecommendationsCard
                configId={selectedConfigId}
                onViewAll={() => setShowRecommendations(true)}
              />
              <AnomaliesCard configId={selectedConfigId} />
            </div>

            {/* Secao de recomendacoes recentes */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-gray-900">
                  Recomendações Recentes
                </h2>
                <button
                  onClick={() => setShowRecommendations(true)}
                  className="text-sm text-indigo-600 hover:text-indigo-700"
                >
                  Ver todas →
                </button>
              </div>
              <RecommendationsList configId={selectedConfigId} limit={5} />
            </div>
          </>
        )}
      </div>
    </div>
  );
}
