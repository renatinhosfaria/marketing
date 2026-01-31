import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import { format, parseISO, formatDistanceToNow } from "date-fns";
import { ptBR } from "date-fns/locale";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDate(dateString: string | Date, formatString: string = "PPP"): string {
  try {
    const date = typeof dateString === "string" ? parseISO(dateString) : dateString;
    return format(date, formatString, { locale: ptBR });
  } catch {
    return "Data inválida";
  }
}

export function formatDateShort(dateString: string | Date): string {
  try {
    if (typeof dateString === "string") {
      const parts = dateString.split('T')[0].split('-');
      if (parts.length === 3) {
        const year = parseInt(parts[0]);
        const month = parseInt(parts[1]) - 1;
        const day = parseInt(parts[2]);
        return format(new Date(year, month, day), "dd/MM/yyyy", { locale: ptBR });
      }
    }
    return formatDate(dateString, "dd/MM/yyyy");
  } catch {
    return formatDate(dateString, "dd/MM/yyyy");
  }
}

export function formatDateTime(dateString: string | Date): string {
  return formatDate(dateString, "dd/MM/yyyy HH:mm");
}

export function formatCurrency(value: number): string {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(value / 100);
}

export function getInitials(name: string): string {
  if (!name) return "";
  const parts = name.split(" ");
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

export function getStatusColor(status: string): { bg: string; text: string } {
  switch (status) {
    case "Novo":
      return { bg: "bg-blue-100", text: "text-blue-800" };
    case "Agendamento concluído":
      return { bg: "bg-green-100", text: "text-green-800" };
    case "Aguardando contato":
      return { bg: "bg-yellow-100", text: "text-yellow-800" };
    case "Visita agendada":
      return { bg: "bg-blue-100", text: "text-blue-800" };
    case "Proposta":
      return { bg: "bg-purple-100", text: "text-purple-800" };
    case "Venda":
      return { bg: "bg-green-100", text: "text-green-800" };
    case "Perdido":
      return { bg: "bg-red-100", text: "text-red-800" };
    default:
      return { bg: "bg-gray-100", text: "text-gray-800" };
  }
}

export function formatPhoneNumber(value: string): string {
  if (!value) return "";
  const phoneNumber = value.replace(/\D/g, "");
  if (phoneNumber.length <= 10) {
    return phoneNumber
      .replace(/(\d{2})(\d)/, "($1) $2")
      .replace(/(\d{4})(\d)/, "$1-$2");
  } else {
    return phoneNumber
      .replace(/(\d{2})(\d)/, "($1) $2")
      .replace(/(\d{5})(\d)/, "$1-$2");
  }
}

export function formatTimeAgo(dateString: string | Date): string {
  if (!dateString) return "";
  try {
    const date = typeof dateString === "string" ? parseISO(dateString) : dateString;
    return formatDistanceToNow(date, { addSuffix: true, locale: ptBR });
  } catch {
    return "Data inválida";
  }
}
