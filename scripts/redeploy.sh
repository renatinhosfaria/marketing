#!/bin/bash
# Script de redeploy do Marketing Stack (Docker Swarm + Traefik)
# Uso: ./scripts/redeploy.sh [backend|frontend|all]
#
# IMPORTANTE: Producao roda APENAS via Docker Swarm (marketing-stack.yml).
# O docker-compose.yml e para desenvolvimento local apenas.
# NAO rode "docker compose up" em producao.

set -e
cd /var/www/marketing

TARGET=${1:-all}

echo "=== Marketing Redeploy ($TARGET) ==="

# Load .env so docker stack deploy picks up all variables
if [ -f .env ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
fi

# Pull latest code
echo "[1/4] Pulling latest code..."
git pull origin main

if [ "$TARGET" = "backend" ] || [ "$TARGET" = "all" ]; then
    echo "[2/4] Building backend image..."
    DOCKER_BUILDKIT=1 docker build --target production -t marketing:latest -f Dockerfile .
fi

if [ "$TARGET" = "frontend" ] || [ "$TARGET" = "all" ]; then
    echo "[3/4] Building frontend image..."
    DOCKER_BUILDKIT=1 docker build -t marketing-frontend:latest -f frontend/Dockerfile frontend/
fi

echo "[4/4] Deploying stack..."
docker stack deploy -c marketing-stack.yml marketing

echo ""
echo "=== Deploy complete. Checking services... ==="
sleep 10
docker service ls | grep marketing
