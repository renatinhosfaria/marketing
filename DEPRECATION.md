# Deprecation Notice - Sistema Legado Single-Agent

**Data:** 2026-01-21
**Status:** ⚠️ DEPRECADO

---

## Resumo

O sistema de agente monolítico (single-agent) localizado em `app/agent/graph/` está **OFICIALMENTE DEPRECADO** e será removido em versões futuras.

## Migração Obrigatória

**De:** Sistema legado single-agent (`app/agent/graph/`)
**Para:** Sistema multi-agente hierárquico (`app/agent/orchestrator/` + `app/agent/subagents/`)

---

## Módulos Deprecados

Os seguintes módulos estão marcados como deprecados:

| Módulo | Status | Substituto |
|--------|--------|------------|
| `app/agent/graph/builder.py` | ⚠️ DEPRECADO | `app/agent/orchestrator/graph.py` |
| `app/agent/graph/nodes.py` | ⚠️ DEPRECADO | `app/agent/orchestrator/nodes/` |
| `app/agent/graph/edges.py` | ⚠️ DEPRECADO | `app/agent/orchestrator/graph.py` |
| `app/agent/graph/state.py` | ⚠️ DEPRECADO | `app/agent/orchestrator/state.py` + `app/agent/subagents/state.py` |

---

## Por que a Migração?

### Limitações do Sistema Legado

1. **Sem paralelização**: Análises executadas sequencialmente
2. **Escalabilidade limitada**: Um único grafo monolítico
3. **Manutenibilidade**: Difícil adicionar novas funcionalidades
4. **Performance**: Latência P95 > 8s

### Vantagens do Sistema Multi-Agente

1. ✅ **Análises paralelas** via orchestrator com `Send()`
2. ✅ **6 subagentes especializados** (classification, anomaly, forecast, recommendation, campaign, analysis)
3. ✅ **Escalabilidade** - cada subagente é independente
4. ✅ **Melhor performance** - Meta: P95 ≤ 6s
5. ✅ **Síntese inteligente** com priorização de insights
6. ✅ **Arquitetura modular** - fácil manutenção e extensão

---

## Timeline de Deprecação

| Data | Milestone |
|------|-----------|
| **2026-01-21** | ⚠️ Sistema legado marcado como DEPRECADO |
| **2026-01-21** | ✅ Sistema multi-agente habilitado em staging |
| **2026-01-27** | 📊 Validação completa em staging (1 semana) |
| **2026-02-03** | 🚀 Rollout gradual em produção (10% → 50% → 100%) |
| **2026-02-17** | 🔒 Sistema multi-agente como padrão (100%) |
| **2026-03-03** | 🗑️ Remoção completa do código legado |

---

## Como Migrar

### Para Usuários (API)

**Nenhuma mudança necessária!** A API permanece compatível:

```python
# Endpoint existente - funciona com ambos os sistemas
POST /api/v1/agent/chat
POST /api/v1/agent/chat/stream
```

O sistema detecta automaticamente qual implementação usar baseado em `AGENT_MULTI_AGENT_ENABLED`.

### Para Desenvolvedores

**1. Habilitar sistema multi-agente:**

```bash
# Em .env
AGENT_MULTI_AGENT_ENABLED=true
```

**2. Novos endpoints (opcional):**

```python
# Endpoints específicos do multi-agente
POST /api/v1/agent/multi-agent/chat
GET  /api/v1/agent/multi-agent/chat/stream
GET  /api/v1/agent/multi-agent/status
GET  /api/v1/agent/subagents
```

**3. Não use mais:**

```python
# ❌ DEPRECADO - NÃO USE
from app.agent.graph.builder import build_agent_graph

# ✅ USE ISTO
from app.agent.orchestrator import get_orchestrator
from app.agent.service import get_multi_agent_service
```

---

## Configuração Multi-Agent

Adicione ao `.env`:

```env
# Sistema Multi-Agente (STAGING - HABILITADO)
AGENT_MULTI_AGENT_ENABLED=true
AGENT_ORCHESTRATOR_TIMEOUT=120
AGENT_MAX_PARALLEL_SUBAGENTS=4

# Subagent Timeouts
AGENT_TIMEOUT_CLASSIFICATION=30
AGENT_TIMEOUT_ANOMALY=30
AGENT_TIMEOUT_FORECAST=45
AGENT_TIMEOUT_RECOMMENDATION=30
AGENT_TIMEOUT_CAMPAIGN=20
AGENT_TIMEOUT_ANALYSIS=45

# Synthesis
AGENT_SYNTHESIS_MAX_TOKENS=4096
AGENT_SYNTHESIS_TEMPERATURE=0.3

# Retry
AGENT_SUBAGENT_MAX_RETRIES=2
AGENT_SUBAGENT_RETRY_DELAY=1.0
```

---

## Rollback Plan

Se encontrar problemas críticos, você pode reverter:

```bash
# Opção 1: Via .env
AGENT_MULTI_AGENT_ENABLED=false

# Opção 2: Via PM2
pm2 restart famachat-ml --env AGENT_MULTI_AGENT_ENABLED=false
```

---

## Suporte

- **Issues:** [GitHub Issues](https://github.com/famachat/famachat-ml/issues)
- **Documentação:** [docs/plans/2026-01-19-multi-agent-system-design.md](docs/plans/2026-01-19-multi-agent-system-design.md)
- **README Orchestrator:** [app/agent/orchestrator/README.md](app/agent/orchestrator/README.md)

---

## Aviso Legal

⚠️ **IMPORTANTE:** O sistema legado funcionará até 2026-03-03. Após esta data, imports diretos de `app.agent.graph.*` resultarão em erro.

**Planeje sua migração agora!**

---

*Última atualização: 2026-01-21*
