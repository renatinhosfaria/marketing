#\!/bin/bash
# Health check para todos os serviços Marketing

SERVICES=(
    "http://localhost:8000/api/v1/health|API ML"
    "http://localhost:8002/api/v1/facebook-ads/health/simple|Facebook Ads"
)

FAILED=0
for service in "${SERVICES[@]}"; do
    URL="${service%%|*}"
    NAME="${service##*|}"
    if curl -sf "$URL" > /dev/null 2>&1; then
        echo "[OK] $NAME"
    else
        echo "[FAIL] $NAME ($URL)"
        FAILED=1
    fi
done

# Check Docker services
echo ""
echo "=== Docker Services ==="
docker service ls | grep marketing | awk "{print \$2, \$4}" | while read name replicas; do
    if [ "$replicas" = "1/1" ]; then
        echo "[OK] $name"
    else
        echo "[FAIL] $name ($replicas)"
        FAILED=1
    fi
done

exit $FAILED
