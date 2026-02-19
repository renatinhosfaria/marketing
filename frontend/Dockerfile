# syntax=docker/dockerfile:1.4
# Marketing Frontend - Next.js
# Multi-stage build otimizado com standalone output

# ==================== DEPS STAGE ====================
FROM node:20-alpine AS deps

WORKDIR /app

# Copiar apenas arquivos de dependencias para cache
COPY package.json package-lock.json ./

# Instalar dependencias com cache
RUN --mount=type=cache,target=/root/.npm \
    npm ci --ignore-scripts

# ==================== BUILD STAGE ====================
FROM node:20-alpine AS builder

WORKDIR /app

# Copiar deps instaladas
COPY --from=deps /app/node_modules ./node_modules

# Copiar codigo fonte
COPY . .

# Build args para variaveis de ambiente em build time
ARG NEXT_PUBLIC_API_URL=""
ARG NEXT_PUBLIC_APP_NAME="Marketing"
ARG NEXT_PUBLIC_FACEBOOK_APP_ID=""
ARG API_INTERNAL_URL="http://marketing-api:8000"
ARG FB_ADS_INTERNAL_URL="http://marketing-fb-ads:8002"
ARG AGENT_INTERNAL_URL="http://marketing-agent-api:8001"

ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV NEXT_PUBLIC_APP_NAME=${NEXT_PUBLIC_APP_NAME}
ENV NEXT_PUBLIC_FACEBOOK_APP_ID=${NEXT_PUBLIC_FACEBOOK_APP_ID}
ENV API_INTERNAL_URL=${API_INTERNAL_URL}
ENV FB_ADS_INTERNAL_URL=${FB_ADS_INTERNAL_URL}
ENV AGENT_INTERNAL_URL=${AGENT_INTERNAL_URL}
ENV NEXT_TELEMETRY_DISABLED=1

# Build de producao (standalone)
RUN npm run build

# ==================== PRODUCTION STAGE ====================
FROM node:20-alpine AS production

LABEL maintainer="Marketing Team"
LABEL description="Frontend Next.js para Marketing"

WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV HOSTNAME="0.0.0.0"
ENV PORT=3001

# Criar usuario nao-root
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copiar apenas o necessario do build standalone
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3001

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3001/ || exit 1

CMD ["node", "server.js"]
