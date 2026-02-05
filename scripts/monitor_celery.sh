#!/bin/bash
# Script de monitoramento do Celery Beat e Workers
# Uso: ./scripts/monitor_celery.sh

set -e

cd "$(dirname "$0")/.."

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║              📊 MONITOR CELERY - Marketing                               ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Status dos containers
echo "📦 STATUS DOS CONTAINERS:"
echo ""
docker-compose ps marketing-beat marketing-worker marketing-redis | tail -n +2
echo ""

# Workers ativos
echo "👷 WORKERS ATIVOS:"
echo ""
docker-compose exec -T marketing-worker celery -A app.tasks.celery_app inspect active 2>/dev/null | grep -A 5 "celery@" || echo "  Nenhuma task em execução no momento"
echo ""

# Stats dos workers
echo "📈 ESTATÍSTICAS DOS WORKERS:"
echo ""
docker-compose exec -T marketing-worker celery -A app.tasks.celery_app inspect stats 2>/dev/null | grep -E "(celery@|total:)" | head -10
echo ""

# Scheduled tasks
echo "📅 SCHEDULES REGISTRADOS:"
echo ""
docker-compose exec -T marketing-worker celery -A app.tasks.celery_app inspect scheduled 2>/dev/null | grep -A 3 "celery@" || echo "  Nenhuma task agendada"
echo ""

# Logs recentes do Beat
echo "📋 ÚLTIMAS EXECUÇÕES (Beat logs):"
echo ""
docker-compose logs marketing-beat --tail=20 2>/dev/null | grep "Sending due task" | tail -10 || echo "  Nenhuma execução recente"
echo ""

# Tasks registradas
echo "📝 TASKS REGISTRADAS:"
echo ""
docker-compose exec -T marketing-worker celery -A app.tasks.celery_app inspect registered 2>/dev/null | grep "app.tasks" | sort | uniq
echo ""

echo "╔════════════════════════════════════════════════════════════════════════════╗"
echo "║  💡 COMANDOS ÚTEIS:                                                        ║"
echo "║                                                                            ║"
echo "║  docker-compose logs -f marketing-beat    # Ver logs do Beat              ║"
echo "║  docker-compose logs -f marketing-worker  # Ver logs do Worker            ║"
echo "║  http://localhost:5555                     # Flower (interface web)        ║"
echo "╚════════════════════════════════════════════════════════════════════════════╝"
