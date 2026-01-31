"use client";

import { usePathname } from "next/navigation";

const pageTitles: Record<string, string> = {
  "/app/facebook-ads": "Facebook Ads",
  "/app/facebook-ads/settings": "Configurações Facebook Ads",
  "/app/ml": "ML Analytics",
  "/app/ai-agent": "Agente IA",
};

export function Header() {
  const pathname = usePathname();

  // Encontra o titulo mais especifico que corresponde ao pathname atual
  const title = Object.entries(pageTitles)
    .sort(([a], [b]) => b.length - a.length)
    .find(([path]) => pathname.startsWith(path))?.[1] || "FamaChat ML";

  return (
    <header className="flex h-14 items-center border-b bg-card px-6 md:px-8">
      <h1 className="text-lg font-semibold ml-12 md:ml-0">{title}</h1>
    </header>
  );
}
