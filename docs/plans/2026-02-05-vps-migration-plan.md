# Plano de Migração FAMACHAT-ML para Novo Servidor VPS

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrar o projeto FAMACHAT-ML completo de `/var/www/famachat-ml` (servidor atual) para `/var/www/marketing` no novo servidor VPS `144.126.134.23`.

**Arquitetura:** O projeto roda 9 containers Docker (3 APIs FastAPI, 1 frontend Next.js, 2 Celery workers, 1 Beat scheduler, 1 Redis, 1 Flower). O PostgreSQL já está no servidor destino (144.126.134.23:5432). A migração envolve transferência de código, rebuild de imagens Docker, configuração de Nginx com SSL, e atualização de DNS.

**Tech Stack:** Docker Compose v5, Python 3.11, Node 20, FastAPI, Next.js, Celery, Redis, PostgreSQL, Nginx, Let's Encrypt/Certbot

---

## Inventário Completo do Projeto

### Serviços Docker (9 containers)
| Container             | Porta Host → Container | Função                        | RAM Limit |
|-----------------------|------------------------|-------------------------------|-----------|
| marketing-frontend    | 8000 → 3001           | Frontend Next.js              | 256M      |
| marketing-api         | 8001 → 8000           | API ML (FastAPI)              | 2G        |
| marketing-agent       | 8002 → 8001           | Agente IA (FastAPI)           | 1G        |
| marketing-fb-ads      | 8003 → 8002           | Facebook Ads API (FastAPI)    | 512M      |
| marketing-worker      | 8004                   | Celery Worker ML/Training     | 3G        |
| marketing-fb-ads-worker| 8005                  | Celery Worker Facebook Ads    | 1G        |
| marketing-beat        | 8006                   | Celery Beat Scheduler         | 256M      |
| marketing-redis       | 8007 → 6379           | Redis (broker + cache)        | 512M      |
| marketing-flower      | 5555 → 5555           | Flower (monitoramento)        | 256M      |

### Volumes Docker
- `redis_ml_data` — Persistência Redis
- `celerybeat_schedule` — Estado do scheduler

### Portas Necessárias no Novo Servidor
- **80, 443** — Nginx (HTTP/HTTPS)
- **5432** — PostgreSQL (já existente)
- **5555** — Flower Dashboard
- **8000-8007** — Containers Docker (bind em 127.0.0.1, exceto Redis e Flower)

### Banco de Dados
- **PostgreSQL** já roda em `144.126.134.23:5432` (o próprio servidor destino)
- **Database:** `marketing`
- **Conexão:** `postgresql://postgres:IwOLgVnyOfbN@144.126.134.23:5432/marketing`
- **NOTA:** Após migração, o `DATABASE_URL` pode usar `localhost` em vez do IP externo

### Certificados SSL
- Domínio: `marketing.famachat.com.br`
- Certificados Let's Encrypt via Certbot
- Precisam ser gerados no novo servidor

### Requisitos Mínimos do Novo Servidor
- **CPU:** 4+ cores (total limits: ~7 CPUs)
- **RAM:** 12GB+ (total limits: ~8.5GB + sistema)
- **Disco:** 20GB+ (código ~100MB, imagens Docker ~2GB, deps ~2GB, crescimento)
- **OS:** Ubuntu 22.04+ (recomendado)

---

## Fases da Migração

---

## FASE 1: Preparação do Servidor Atual (Antes da Migração)

### Task 1: Fazer commit de todas as alterações pendentes

**Contexto:** O repositório tem alterações não commitadas que precisam ser salvas antes da transferência.

**Step 1: Verificar status do git**

```bash
cd /var/www/famachat-ml
git status
```

**Step 2: Adicionar e commitar todas as alterações pendentes**

```bash
git add -A
git commit -m "chore: save all pending changes before VPS migration"
```

**Step 3: Push para o repositório remoto**

```bash
git push origin main
```

**Expected:** Todas as alterações salvas no GitHub (`https://github.com/renatinhosfaria/marketing.git`)

---

### Task 2: Backup dos dados persistentes

**Step 1: Backup dos modelos ML**

```bash
tar -czf /tmp/marketing-models-backup.tar.gz -C /var/www/famachat-ml models_storage/
```

**Step 2: Backup dos logs**

```bash
tar -czf /tmp/marketing-logs-backup.tar.gz -C /var/www/famachat-ml logs/
```

**Step 3: Backup do arquivo .env**

```bash
cp /var/www/famachat-ml/.env /tmp/marketing-env-backup
cp /var/www/famachat-ml/frontend/.env.local /tmp/marketing-frontend-env-backup
```

**Step 4: Backup do Redis (dados em memória)**

```bash
docker exec marketing-redis redis-cli BGSAVE
sleep 2
docker cp marketing-redis:/data/dump.rdb /tmp/marketing-redis-backup.rdb
```

**Step 5: Backup do banco PostgreSQL**

```bash
pg_dump -h 144.126.134.23 -U postgres -d marketing -F c -f /tmp/marketing-db-backup.dump
```

> **NOTA:** Este backup é preventivo. O banco já está no servidor destino. Se o banco for compartilhado com o servidor atual, NÃO será necessário restaurar — apenas garantir que o novo servidor pode acessá-lo.

---

### Task 3: Exportar a configuração Nginx atual

**Step 1: Copiar config Nginx ativa**

```bash
cp /etc/nginx/sites-available/famachat-ml /tmp/marketing-nginx-backup.conf
```

---

## FASE 2: Preparação do Novo Servidor VPS

### Task 4: Conectar ao novo servidor e verificar o ambiente

**Step 1: Conectar via SSH**

```bash
ssh root@144.126.134.23
```

**Step 2: Verificar sistema operacional e recursos**

```bash
cat /etc/os-release
nproc
free -h
df -h
```

**Expected:** Ubuntu 22.04+, 4+ CPU cores, 12GB+ RAM, 20GB+ disco livre

---

### Task 5: Instalar dependências no novo servidor

**Step 1: Atualizar sistema**

```bash
apt update && apt upgrade -y
```

**Step 2: Instalar Docker (se não existir)**

```bash
# Verificar se Docker existe
docker --version 2>/dev/null || {
    # Instalar Docker
    curl -fsSL https://get.docker.com | sh
    systemctl enable docker
    systemctl start docker
}
```

**Step 3: Instalar Docker Compose plugin (se não existir)**

```bash
docker compose version 2>/dev/null || {
    apt install -y docker-compose-plugin
}
```

**Step 4: Instalar Nginx (se não existir)**

```bash
nginx -v 2>/dev/null || {
    apt install -y nginx
    systemctl enable nginx
    systemctl start nginx
}
```

**Step 5: Instalar Certbot (se não existir)**

```bash
certbot --version 2>/dev/null || {
    apt install -y certbot python3-certbot-nginx
}
```

**Step 6: Instalar Git (se não existir)**

```bash
git --version 2>/dev/null || {
    apt install -y git
}
```

**Step 7: Verificar PostgreSQL acessível localmente**

```bash
psql -h localhost -U postgres -d marketing -c "SELECT 1;" 2>/dev/null && echo "PostgreSQL OK"
```

> **Se falhar:** Verificar se o PostgreSQL está configurado para aceitar conexões locais em `pg_hba.conf` e `postgresql.conf`.

---

### Task 6: Criar estrutura de diretórios no novo servidor

**Step 1: Criar diretório base**

```bash
mkdir -p /var/www/marketing
```

**Step 2: Criar diretórios de dados**

```bash
mkdir -p /var/www/marketing/models_storage
mkdir -p /var/www/marketing/logs
```

---

## FASE 3: Transferência do Código

### Task 7: Clonar repositório no novo servidor

**Step 1: Clonar o repositório**

```bash
cd /var/www
git clone https://github.com/renatinhosfaria/marketing.git marketing
```

> **NOTA:** Se o repositório for privado, será necessário configurar credenciais Git (token ou SSH key).

**Step 2: Verificar se o clone foi bem-sucedido**

```bash
cd /var/www/marketing
git log --oneline -5
ls -la
```

**Expected:** Código completo com os últimos commits.

---

### Task 8: Transferir arquivos de configuração sensíveis

**Step 1: Copiar .env do servidor atual para o novo** (executar no servidor ATUAL)

```bash
scp /var/www/famachat-ml/.env root@144.126.134.23:/var/www/marketing/.env
scp /var/www/famachat-ml/frontend/.env.local root@144.126.134.23:/var/www/marketing/frontend/.env.local
```

**Step 2: No novo servidor — atualizar DATABASE_URL para usar localhost**

Editar `/var/www/marketing/.env`:

```bash
# ANTES:
# DATABASE_URL=postgresql://postgres:IwOLgVnyOfbN@144.126.134.23:5432/marketing?sslmode=disable

# DEPOIS (usar localhost, já que o banco está no mesmo servidor):
DATABASE_URL=postgresql://postgres:IwOLgVnyOfbN@localhost:5432/marketing?sslmode=disable
```

**Step 3: Atualizar REDIS_URL para usar apenas o Redis do Docker**

```bash
# ANTES:
# REDIS_URL=redis://:IwOLgVnyOfbN@127.0.0.1:6379/1

# DEPOIS (o Redis Docker usa porta 8007 no host, sem senha):
# Mas como os containers usam a rede interna Docker (marketing-redis:6379/0),
# esta variável do .env é usada apenas fora do Docker.
# Manter como está se houver uso local, ou remover se desnecessário.
REDIS_URL=redis://:IwOLgVnyOfbN@127.0.0.1:6379/1
```

> **DECISÃO:** Se o novo servidor NÃO tem Redis standalone instalado (apenas o Docker Redis), considere alterar a porta do Redis Docker de 8007 para 6379 ou ajustar a URL.

---

### Task 9: Transferir dados persistentes

**Step 1: Transferir modelos ML** (executar no servidor ATUAL)

```bash
scp /tmp/marketing-models-backup.tar.gz root@144.126.134.23:/tmp/
```

**Step 2: No novo servidor — extrair modelos**

```bash
tar -xzf /tmp/marketing-models-backup.tar.gz -C /var/www/marketing/
```

**Step 3: Transferir backup Redis** (executar no servidor ATUAL)

```bash
scp /tmp/marketing-redis-backup.rdb root@144.126.134.23:/tmp/
```

> **NOTA:** O Redis Docker será iniciado vazio. Se precisar restaurar dados, copie o `.rdb` para o volume após subir o container.

---

## FASE 4: Build e Deploy dos Containers Docker

### Task 10: Build das imagens Docker no novo servidor

**Step 1: Build da imagem backend (marketing:latest)**

```bash
cd /var/www/marketing
docker compose build marketing-api
```

> **NOTA:** Este build cria a imagem `marketing:latest` que é compartilhada por 6 serviços (api, agent, fb-ads, worker, fb-ads-worker, beat, flower).

**Step 2: Build da imagem frontend (marketing-frontend:latest)**

```bash
docker compose build marketing-frontend
```

**Step 3: Verificar imagens criadas**

```bash
docker images | grep marketing
```

**Expected:**
```
marketing             latest    ...    ~1.45GB
marketing-frontend    latest    ...    ~212MB
```

---

### Task 11: Subir os containers Docker

**Step 1: Subir todos os serviços**

```bash
cd /var/www/marketing
docker compose up -d
```

**Step 2: Verificar status dos containers**

```bash
docker compose ps
```

**Expected:** Todos os 9 containers com status `Up` e `healthy`.

**Step 3: Verificar logs de cada serviço**

```bash
docker compose logs marketing-api --tail 20
docker compose logs marketing-agent --tail 20
docker compose logs marketing-fb-ads --tail 20
docker compose logs marketing-frontend --tail 20
docker compose logs marketing-worker --tail 20
docker compose logs marketing-beat --tail 20
docker compose logs marketing-redis --tail 20
```

**Step 4: Testar health checks**

```bash
# API ML
curl -f http://localhost:8001/api/v1/health && echo " [OK]"

# Agent
curl -f http://localhost:8002/api/v1/health && echo " [OK]"

# Facebook Ads
curl -f http://localhost:8003/api/v1/facebook-ads/health/simple && echo " [OK]"

# Frontend
curl -f http://localhost:8000/ && echo " [OK]"
```

**Expected:** Todos retornam resposta 200 OK.

---

### Task 12: Executar migrations do Alembic (se necessário)

**Step 1: Verificar status das migrations**

```bash
docker exec marketing-api alembic current
```

**Step 2: Se houver migrations pendentes, aplicar**

```bash
docker exec marketing-api alembic upgrade head
```

> **NOTA:** O banco `marketing` já existe no servidor destino. As migrations já devem ter sido aplicadas previamente. Execute apenas se `alembic current` mostrar que está desatualizado.

---

## FASE 5: Configuração do Nginx e SSL

### Task 13: Configurar Nginx no novo servidor

**Step 1: Criar arquivo de configuração Nginx**

Criar `/etc/nginx/sites-available/marketing`:

```nginx
# Marketing - Frontend Next.js + 3 APIs FastAPI
# Containers Docker com portas:
#   - marketing-frontend: 8000 (Next.js)
#   - marketing-api: 8001 (FastAPI ML)
#   - marketing-agent: 8002 (FastAPI Agent)
#   - marketing-fb-ads: 8003 (FastAPI Facebook Ads)

server {
    listen 80;
    server_name marketing.famachat.com.br;

    # Certbot challenge
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # Redirect HTTP to HTTPS (após certificado)
    location / {
        return 301 https://$host$request_uri;
    }
}

server {
    listen 443 ssl http2;
    server_name marketing.famachat.com.br;

    # SSL (será preenchido pelo Certbot)
    # ssl_certificate /etc/letsencrypt/live/marketing.famachat.com.br/fullchain.pem;
    # ssl_certificate_key /etc/letsencrypt/live/marketing.famachat.com.br/privkey.pem;

    # Logs
    access_log /var/log/nginx/marketing-access.log;
    error_log /var/log/nginx/marketing-error.log;

    # Agent IA (porta 8002) - DEVE vir antes de /api/ genérico
    location /api/v1/agent {
        proxy_pass http://127.0.0.1:8002;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support (crítico para streaming do Agent)
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        client_max_body_size 10M;
    }

    # Facebook Ads (porta 8003) - DEVE vir antes de /api/ genérico
    location /api/v1/facebook-ads {
        proxy_pass http://127.0.0.1:8003;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_read_timeout 120s;
        proxy_send_timeout 120s;

        client_max_body_size 10M;
    }

    # API FastAPI - Machine Learning (porta 8001)
    location /api/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;

        # Large uploads
        client_max_body_size 50M;
    }

    # Flower Dashboard (porta 5555) - Monitoramento Celery
    location /flower/ {
        proxy_pass http://127.0.0.1:5555;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Frontend Next.js (porta 8000)
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Next.js static assets (porta 8000)
    location /_next/static/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_cache_bypass $http_upgrade;
        expires 365d;
        add_header Cache-Control "public, immutable";
    }
}
```

**Step 2: Ativar o site**

```bash
ln -s /etc/nginx/sites-available/marketing /etc/nginx/sites-enabled/marketing
```

**Step 3: Testar configuração Nginx**

```bash
nginx -t
```

**Expected:** `nginx: configuration file /etc/nginx/nginx.conf test is successful`

**Step 4: Recarregar Nginx**

```bash
systemctl reload nginx
```

---

### Task 14: Atualizar DNS

**Step 1: No provedor de DNS, atualizar o registro A**

```
Tipo: A
Nome: marketing
Domínio: famachat.com.br
Valor: 144.126.134.23  (IP do novo servidor)
TTL: 300 (5 minutos, temporariamente para propagação rápida)
```

> **IMPORTANTE:** Se o IP `144.126.134.23` é o mesmo do servidor atual (conforme DATABASE_URL), o DNS pode já estar apontando para este servidor. Nesse caso, esta etapa pode ser desnecessária.

**Step 2: Verificar propagação DNS**

```bash
dig +short marketing.famachat.com.br
# ou
nslookup marketing.famachat.com.br
```

**Expected:** Deve retornar `144.126.134.23`

---

### Task 15: Gerar certificado SSL com Certbot

> **PRÉ-REQUISITO:** O DNS deve estar apontando para o novo servidor (Task 14).

**Step 1: Gerar certificado (Nginx plugin)**

```bash
certbot --nginx -d marketing.famachat.com.br --non-interactive --agree-tos --email admin@famachat.com.br
```

> **Se falhar com Nginx plugin:** usar modo standalone:
> ```bash
> systemctl stop nginx
> certbot certonly --standalone -d marketing.famachat.com.br --agree-tos --email admin@famachat.com.br
> systemctl start nginx
> ```

**Step 2: Verificar certificado gerado**

```bash
ls -la /etc/letsencrypt/live/marketing.famachat.com.br/
```

**Expected:** `fullchain.pem`, `privkey.pem`, `chain.pem`, `cert.pem`

**Step 3: Verificar auto-renovação**

```bash
certbot renew --dry-run
```

**Step 4: Recarregar Nginx com SSL**

```bash
nginx -t && systemctl reload nginx
```

---

## FASE 6: Validação Completa

### Task 16: Testar todos os endpoints via HTTPS

**Step 1: Testar Frontend**

```bash
curl -I https://marketing.famachat.com.br/
```

**Expected:** `HTTP/2 200`

**Step 2: Testar API ML**

```bash
curl -s https://marketing.famachat.com.br/api/v1/health | python3 -m json.tool
```

**Expected:** JSON com status healthy

**Step 3: Testar Agent**

```bash
curl -s https://marketing.famachat.com.br/api/v1/agent/health | python3 -m json.tool
```

**Expected:** JSON com status healthy

**Step 4: Testar Facebook Ads**

```bash
curl -s https://marketing.famachat.com.br/api/v1/facebook-ads/health/simple | python3 -m json.tool
```

**Expected:** JSON com status healthy

**Step 5: Testar Flower Dashboard**

```bash
curl -I https://marketing.famachat.com.br/flower/
```

**Expected:** `HTTP/2 200` ou `401` (se autenticação ativa)

---

### Task 17: Verificar Celery Workers e Beat

**Step 1: Verificar workers ativos**

```bash
docker exec marketing-worker celery -A app.celery inspect active
```

**Step 2: Verificar Beat está enviando tarefas**

```bash
docker logs marketing-beat --tail 30
```

**Expected:** Logs mostrando schedule de tarefas

**Step 3: Verificar Flower mostra os workers**

Acessar via browser: `https://marketing.famachat.com.br/flower/`
- Login: admin / famachat123
- Verificar: 2 workers online (ml-worker, fb-ads-worker)

---

### Task 18: Verificar conexão com banco de dados

**Step 1: Testar query via API**

```bash
curl -s https://marketing.famachat.com.br/api/v1/health | python3 -m json.tool
```

**Step 2: Verificar logs do container API por erros de conexão**

```bash
docker logs marketing-api 2>&1 | grep -i "error\|exception\|connection" | tail -20
```

**Expected:** Nenhum erro de conexão com PostgreSQL.

---

## FASE 7: Descomissionamento do Servidor Antigo

> **ATENÇÃO:** Somente executar esta fase após confirmar que TUDO funciona no novo servidor por pelo menos 24-48 horas.

### Task 19: Parar containers no servidor antigo

**Step 1: Conectar ao servidor antigo**

```bash
# No servidor antigo (NÃO no novo)
```

**Step 2: Parar todos os containers**

```bash
cd /var/www/famachat-ml
docker compose down
```

**Step 3: Remover configuração Nginx do servidor antigo (se aplicável)**

```bash
rm /etc/nginx/sites-enabled/famachat-ml
nginx -t && systemctl reload nginx
```

**Step 4: Manter backup por 30 dias**

```bash
# NÃO deletar o diretório ainda. Manter como backup.
# Agendar remoção para 30 dias depois:
echo "rm -rf /var/www/famachat-ml" | at "now + 30 days" 2>/dev/null || echo "Lembrete: deletar /var/www/famachat-ml em 30 dias"
```

---

## FASE 8: Configurações Pós-Migração (Opcionais mas Recomendadas)

### Task 20: Configurar auto-start do Docker Compose

**Step 1: Criar serviço systemd para auto-start**

Criar `/etc/systemd/system/marketing-docker.service`:

```ini
[Unit]
Description=Marketing Docker Compose
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/var/www/marketing
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

**Step 2: Habilitar o serviço**

```bash
systemctl daemon-reload
systemctl enable marketing-docker.service
```

---

### Task 21: Configurar firewall (UFW)

**Step 1: Configurar regras**

```bash
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw allow 5432/tcp  # PostgreSQL (se acesso externo necessário)
# NÃO expor 5555, 8000-8007 externamente (Nginx faz proxy)
ufw enable
```

---

### Task 22: Configurar log rotation

**Step 1: Criar config logrotate**

Criar `/etc/logrotate.d/marketing`:

```
/var/www/marketing/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 root root
}
```

---

### Task 23: Configurar monitoramento básico

**Step 1: Script de health check**

Criar `/var/www/marketing/scripts/healthcheck.sh`:

```bash
#!/bin/bash
# Health check para todos os serviços Marketing

SERVICES=(
    "http://localhost:8001/api/v1/health|API ML"
    "http://localhost:8002/api/v1/health|Agent"
    "http://localhost:8003/api/v1/facebook-ads/health/simple|Facebook Ads"
    "http://localhost:8000/|Frontend"
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

exit $FAILED
```

```bash
chmod +x /var/www/marketing/scripts/healthcheck.sh
```

**Step 2: Agendar health check via cron (opcional)**

```bash
# Verificar a cada 5 minutos
echo "*/5 * * * * /var/www/marketing/scripts/healthcheck.sh >> /var/www/marketing/logs/healthcheck.log 2>&1" | crontab -
```

---

## Resumo da Ordem de Execução

| # | Fase | Tarefas | Tempo Estimado |
|---|------|---------|----------------|
| 1 | Preparar servidor atual | Tasks 1-3 | ~15 min |
| 2 | Preparar novo servidor | Tasks 4-6 | ~20 min |
| 3 | Transferir código | Tasks 7-9 | ~10 min |
| 4 | Build e Deploy Docker | Tasks 10-12 | ~15 min |
| 5 | Nginx e SSL | Tasks 13-15 | ~15 min |
| 6 | Validação | Tasks 16-18 | ~15 min |
| 7 | Descomissionamento | Task 19 | ~5 min (após 24-48h) |
| 8 | Pós-migração | Tasks 20-23 | ~15 min |

**Tempo total estimado: ~2 horas** (excluindo propagação DNS e período de observação)

---

## Checklist de Verificação Final

- [ ] Todos os 9 containers rodando e healthy
- [ ] Frontend acessível via HTTPS
- [ ] API ML respondendo health check
- [ ] Agent respondendo health check
- [ ] Facebook Ads respondendo health check
- [ ] Celery workers registrados no Flower
- [ ] Beat scheduler enviando tarefas
- [ ] Certificado SSL válido
- [ ] Auto-renovação SSL funcionando
- [ ] Firewall configurado
- [ ] Auto-start configurado
- [ ] Logs sendo gerados
- [ ] Backup do servidor antigo mantido

---

## Riscos e Mitigações

| Risco | Impacto | Mitigação |
|-------|---------|-----------|
| DNS não propagado | Usuários acessam servidor antigo | Reduzir TTL para 300s antes da migração |
| Build Docker falha no novo servidor | Deploy bloqueado | Resolver deps faltantes; usar `docker compose build --no-cache` |
| PostgreSQL rejeita conexão localhost | APIs não funcionam | Verificar `pg_hba.conf` e `postgresql.conf` |
| Portas em conflito | Containers não sobem | Verificar `ss -tlnp` antes do deploy |
| Certificado SSL falha | HTTPS indisponível | Usar HTTP temporariamente; gerar cert depois |
| Modelos ML perdidos | Predições falham | Backup em `/tmp/` e re-treinar se necessário |

---

## Rollback

Se algo der errado durante a migração:

1. **Reverter DNS** para o IP do servidor antigo
2. **Reiniciar containers** no servidor antigo: `cd /var/www/famachat-ml && docker compose up -d`
3. **Investigar** o problema no novo servidor sem pressa
4. **Retry** quando o problema estiver resolvido
