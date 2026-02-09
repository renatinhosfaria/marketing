# Memória Avançada do Agente — Plano de Implementação

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Adicionar 4 camadas de memória avançada ao agente de IA: Summarization Memory, Vector Store/RAG, Entity Memory e Cross-thread Memory.

**Architecture:** Sistema incremental em 4 fases, onde cada fase constrói sobre a anterior. Phase 1 (Summarization) reduz o bloat de contexto. Phase 2 (Vector Store) habilita busca semântica via pgvector. Phase 3 (Entity Memory) extrai conhecimento estruturado. Phase 4 (Cross-thread) unifica tudo para aprendizado entre conversas.

**Tech Stack:** PostgreSQL + pgvector, SQLAlchemy 2.0 async, Alembic, LangChain (embeddings), tiktoken (contagem de tokens)

---

## Visão Geral das Fases

| Fase | Feature | Depende de | Impacto no Grafo |
|------|---------|------------|------------------|
| 1 | Summarization Memory | Nenhuma | Novo nó `maybe_summarize` antes de `parse_request` |
| 2 | Vector Store / RAG | pgvector | Novo nó `retrieve_context` após `parse_request` |
| 3 | Entity Memory | Fase 1 | Extração no `synthesize`, injeção no `receive_task` |
| 4 | Cross-thread Memory | Fases 2+3 | Busca cross-thread no `retrieve_context` |

### Grafo Final (após todas as fases):

```
START
  ↓
[load_memory]         ← NOVO: carrega sumário + entidades + contexto cross-thread
  ↓
[parse_request]       ← existente
  ↓
[plan_execution]      ← existente
  ↓
[dispatch_agents]     ← existente (Send paralelo)
  ↓
[subagent_*]          ← existente (6 subagentes)
  ↓
[collect_results]     ← existente
  ↓
[synthesize]          ← existente
  ↓
[persist_memory]      ← NOVO: salva sumário + entidades + embeddings
  ↓
END
```

---

## Phase 1: Summarization Memory

### Objetivo
Quando uma conversa acumula muitas mensagens, resumir automaticamente as mais antigas em um sumário compacto. Isso mantém o contexto importante sem estourar o limite de tokens do LLM.

### Estratégia
- **Trigger:** Quando `len(messages) > summarization_threshold` (padrão: 20 mensagens)
- **Ação:** Pedir ao LLM que resuma as mensagens mais antigas em ~500 tokens
- **Armazenamento:** Salvar o sumário na tabela `agent_conversation_summaries`
- **Injeção:** Na próxima requisição, injetar o sumário como `SystemMessage` antes do histórico truncado

### Fluxo:

```
Requisição com thread_id existente:
  1. Carregar sumário existente do banco (se houver)
  2. Carregar mensagens do checkpointer
  3. Se total de mensagens > threshold:
     a. Separar mensagens antigas (a serem resumidas)
     b. Chamar LLM com prompt de sumarização
     c. Salvar novo sumário no banco
     d. Manter apenas sumário + mensagens recentes no contexto
  4. Injetar sumário como SystemMessage adicional
```

---

### Task 1.1: Migração — Tabela `agent_conversation_summaries`

**Files:**
- Create: `alembic/versions/013_create_conversation_summaries.py`

**Step 1: Escrever a migração**

```python
"""Cria tabela agent_conversation_summaries para memória de sumarização."""

from alembic import op
import sqlalchemy as sa

revision = "013"
down_revision = "012_create_agent_api_keys"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_conversation_summaries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("thread_id", sa.String(255), nullable=False, index=True),
        sa.Column("user_id", sa.Integer, nullable=False),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("summary_text", sa.Text, nullable=False),
        sa.Column("token_count", sa.Integer, nullable=False, default=0),
        sa.Column("messages_summarized", sa.Integer, nullable=False, default=0),
        sa.Column("last_message_index", sa.Integer, nullable=False, default=0),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_conv_summaries_thread",
        "agent_conversation_summaries",
        ["thread_id"],
        unique=True,
    )
    op.create_index(
        "idx_conv_summaries_user",
        "agent_conversation_summaries",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_table("agent_conversation_summaries")
```

**Step 2: Rodar a migração**

Run: `docker compose exec marketing-agent alembic upgrade head`
Expected: Tabela criada com sucesso

**Step 3: Commit**

```bash
git add alembic/versions/013_create_conversation_summaries.py
git commit -m "feat(agent): add migration for conversation summaries table"
```

---

### Task 1.2: Modelo SQLAlchemy — `ConversationSummary`

**Files:**
- Modify: `projects/agent/db/models.py`
- Test: `tests/unit/agent/test_summary_model.py`

**Step 1: Escrever o teste**

```python
"""Testes para o modelo ConversationSummary."""
import pytest
from projects.agent.db.models import ConversationSummary


class TestConversationSummaryModel:
    def test_create_summary(self):
        summary = ConversationSummary(
            thread_id="test-thread-123",
            user_id=1,
            config_id=1,
            summary_text="Resumo da conversa sobre performance de campanhas.",
            token_count=50,
            messages_summarized=15,
            last_message_index=15,
        )
        assert summary.thread_id == "test-thread-123"
        assert summary.summary_text.startswith("Resumo")
        assert summary.token_count == 50
        assert summary.messages_summarized == 15

    def test_tablename(self):
        assert ConversationSummary.__tablename__ == "agent_conversation_summaries"
```

**Step 2: Rodar o teste e verificar que falha**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/test_summary_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'ConversationSummary'`

**Step 3: Implementar o modelo**

Adicionar ao final de `projects/agent/db/models.py`:

```python
class ConversationSummary(Base):
    """
    Sumários de conversas para memória de longo prazo.
    Armazena resumos compactos de conversas longas para preservar contexto
    sem estourar o limite de tokens do LLM.
    """
    __tablename__ = "agent_conversation_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    messages_summarized: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_message_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
```

**Step 4: Rodar o teste e verificar que passa**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/test_summary_model.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add projects/agent/db/models.py tests/unit/agent/test_summary_model.py
git commit -m "feat(agent): add ConversationSummary model"
```

---

### Task 1.3: Serviço de Sumarização — `SummarizationService`

**Files:**
- Create: `projects/agent/memory/summarization.py`
- Test: `tests/unit/agent/memory/test_summarization.py`

**Step 1: Escrever os testes**

```python
"""Testes para SummarizationService."""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from projects.agent.memory.summarization import (
    SummarizationService,
    should_summarize,
    build_summarization_prompt,
)


class TestShouldSummarize:
    def test_below_threshold_returns_false(self):
        messages = [HumanMessage(content="oi")] * 10
        assert should_summarize(messages, threshold=20) is False

    def test_above_threshold_returns_true(self):
        messages = [HumanMessage(content="oi")] * 25
        assert should_summarize(messages, threshold=20) is True

    def test_exactly_at_threshold_returns_false(self):
        messages = [HumanMessage(content="oi")] * 20
        assert should_summarize(messages, threshold=20) is False

    def test_system_messages_not_counted(self):
        messages = [SystemMessage(content="system")] * 15 + [HumanMessage(content="oi")] * 10
        assert should_summarize(messages, threshold=20) is False


class TestBuildSummarizationPrompt:
    def test_includes_messages(self):
        messages = [
            HumanMessage(content="Como estão minhas campanhas?"),
            AIMessage(content="Suas campanhas estão indo bem."),
        ]
        prompt = build_summarization_prompt(messages)
        assert "campanhas" in prompt
        assert "Usuario:" in prompt
        assert "Assistente:" in prompt


@pytest.mark.asyncio
class TestSummarizationService:
    async def test_summarize_messages_calls_llm(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="Resumo: usuario perguntou sobre campanhas."
        )

        service = SummarizationService()
        messages = [
            HumanMessage(content="Como estão?"),
            AIMessage(content="Suas campanhas estão bem."),
        ]

        with patch("projects.agent.memory.summarization.get_llm", return_value=mock_llm):
            result = await service.summarize_messages(messages)

        assert "Resumo" in result
        mock_llm.ainvoke.assert_called_once()

    async def test_get_or_create_summary_creates_new(self):
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=MagicMock(return_value=None)))
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(content="Resumo da conversa.")

        service = SummarizationService()

        messages = [HumanMessage(content=f"msg {i}") for i in range(25)]

        with patch("projects.agent.memory.summarization.get_llm", return_value=mock_llm):
            with patch("projects.agent.memory.summarization.async_session_maker", return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())):
                summary, trimmed = await service.get_or_create_summary(
                    thread_id="t-1",
                    user_id=1,
                    config_id=1,
                    messages=messages,
                    threshold=20,
                    keep_recent=10,
                )

        assert summary is not None
        assert len(trimmed) == 10  # Apenas mensagens recentes mantidas
```

**Step 2: Rodar os testes e verificar que falham**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/memory/test_summarization.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implementar o serviço**

```python
"""
Serviço de sumarização para memória de longo prazo.

Quando uma conversa excede um threshold de mensagens, as mais antigas
são resumidas em um sumário compacto pelo LLM. O sumário é persistido
no banco e injetado como contexto nas próximas requisições.
"""
from typing import Optional, Tuple, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from sqlalchemy import select

from projects.agent.db.models import ConversationSummary
from shared.db.session import async_session_maker
from shared.core.logging import get_logger

logger = get_logger(__name__)


def should_summarize(messages: list[BaseMessage], threshold: int = 20) -> bool:
    """Verifica se as mensagens devem ser resumidas.

    Conta apenas mensagens não-system (HumanMessage, AIMessage).

    Args:
        messages: Lista de mensagens da conversa.
        threshold: Número máximo de mensagens antes de sumarizar.

    Returns:
        True se o número de mensagens não-system excede o threshold.
    """
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    return len(non_system) > threshold


def build_summarization_prompt(messages: list[BaseMessage]) -> str:
    """Constrói prompt para o LLM resumir mensagens.

    Args:
        messages: Mensagens a serem resumidas.

    Returns:
        Prompt formatado para sumarização.
    """
    lines = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            lines.append(f"Usuario: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistente: {msg.content}")

    conversation = "\n".join(lines)

    return (
        "Voce e um assistente de sumarizacao. Resuma a conversa abaixo de forma "
        "concisa, preservando:\n"
        "- Perguntas e topicos discutidos pelo usuario\n"
        "- Campanhas, metricas e valores mencionados\n"
        "- Decisoes tomadas ou recomendacoes aceitas\n"
        "- Qualquer preferencia expressa pelo usuario\n\n"
        "Mantenha o resumo em no maximo 500 tokens.\n\n"
        f"--- CONVERSA ---\n{conversation}\n--- FIM ---\n\n"
        "Resumo:"
    )


class SummarizationService:
    """Serviço para sumarização de conversas longas."""

    async def summarize_messages(self, messages: list[BaseMessage]) -> str:
        """Resume uma lista de mensagens usando o LLM.

        Args:
            messages: Mensagens a resumir.

        Returns:
            Texto do resumo.
        """
        from projects.agent.llm.provider import get_llm

        prompt = build_summarization_prompt(messages)
        llm = get_llm(temperature=0.1, max_tokens=600)
        response = await llm.ainvoke(prompt)
        return response.content.strip()

    async def load_summary(self, thread_id: str) -> Optional[str]:
        """Carrega sumário existente do banco.

        Args:
            thread_id: ID da thread.

        Returns:
            Texto do sumário ou None se não existir.
        """
        async with async_session_maker() as session:
            result = await session.execute(
                select(ConversationSummary).where(
                    ConversationSummary.thread_id == thread_id
                )
            )
            summary = result.scalar_one_or_none()
            return summary.summary_text if summary else None

    async def save_summary(
        self,
        thread_id: str,
        user_id: int,
        config_id: int,
        summary_text: str,
        messages_summarized: int,
        last_message_index: int,
    ) -> None:
        """Salva ou atualiza sumário no banco.

        Args:
            thread_id: ID da thread.
            user_id: ID do usuário.
            config_id: ID da configuração.
            summary_text: Texto do resumo.
            messages_summarized: Quantidade de mensagens resumidas.
            last_message_index: Índice da última mensagem resumida.
        """
        async with async_session_maker() as session:
            result = await session.execute(
                select(ConversationSummary).where(
                    ConversationSummary.thread_id == thread_id
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                existing.summary_text = summary_text
                existing.messages_summarized = messages_summarized
                existing.last_message_index = last_message_index
                # token_count estimado: ~4 chars por token
                existing.token_count = len(summary_text) // 4
            else:
                summary = ConversationSummary(
                    thread_id=thread_id,
                    user_id=user_id,
                    config_id=config_id,
                    summary_text=summary_text,
                    token_count=len(summary_text) // 4,
                    messages_summarized=messages_summarized,
                    last_message_index=last_message_index,
                )
                session.add(summary)

            await session.commit()

    async def get_or_create_summary(
        self,
        thread_id: str,
        user_id: int,
        config_id: int,
        messages: list[BaseMessage],
        threshold: int = 20,
        keep_recent: int = 10,
    ) -> Tuple[Optional[str], List[BaseMessage]]:
        """Obtém sumário existente ou cria um novo se necessário.

        Retorna o sumário e a lista de mensagens recentes (já truncada).

        Args:
            thread_id: ID da thread.
            user_id: ID do usuário.
            config_id: ID da configuração.
            messages: Todas as mensagens da conversa.
            threshold: Trigger para sumarização.
            keep_recent: Quantas mensagens recentes manter intactas.

        Returns:
            Tupla (summary_text, recent_messages).
        """
        if not should_summarize(messages, threshold):
            return None, messages

        # Separar system messages
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        # Mensagens antigas (para resumir) e recentes (manter intactas)
        old_messages = non_system[:-keep_recent]
        recent_messages = non_system[-keep_recent:]

        if not old_messages:
            return None, messages

        logger.info(
            "Sumarizando mensagens antigas",
            thread_id=thread_id,
            old_count=len(old_messages),
            recent_count=len(recent_messages),
        )

        # Gerar sumário
        summary_text = await self.summarize_messages(old_messages)

        # Persistir
        await self.save_summary(
            thread_id=thread_id,
            user_id=user_id,
            config_id=config_id,
            summary_text=summary_text,
            messages_summarized=len(old_messages),
            last_message_index=len(non_system) - keep_recent,
        )

        return summary_text, system_msgs + recent_messages
```

**Step 4: Rodar os testes e verificar que passam**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/memory/test_summarization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add projects/agent/memory/summarization.py tests/unit/agent/memory/test_summarization.py
git commit -m "feat(agent): add SummarizationService for long conversation memory"
```

---

### Task 1.4: Configurações de Sumarização

**Files:**
- Modify: `projects/agent/config.py`
- Test: `tests/unit/agent/test_config.py` (adicionar testes)

**Step 1: Escrever os testes**

Adicionar à suite existente em `tests/unit/agent/test_config.py`:

```python
def test_summarization_defaults():
    settings = AgentSettings()
    assert settings.summarization_enabled is True
    assert settings.summarization_threshold == 20
    assert settings.summarization_keep_recent == 10
    assert settings.summarization_max_tokens == 600
```

**Step 2: Rodar o teste e verificar que falha**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/test_config.py::test_summarization_defaults -v`
Expected: FAIL — `AttributeError`

**Step 3: Adicionar campos ao `AgentSettings`**

Adicionar ao `AgentSettings` em `projects/agent/config.py`, na seção de Persistência:

```python
    # Summarization Memory
    summarization_enabled: bool = Field(
        default=True,
        description="Habilitar sumarização automática de conversas longas"
    )
    summarization_threshold: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Número de mensagens não-system que triggera sumarização"
    )
    summarization_keep_recent: int = Field(
        default=10,
        ge=3,
        le=50,
        description="Quantas mensagens recentes manter intactas após sumarização"
    )
    summarization_max_tokens: int = Field(
        default=600,
        ge=100,
        le=2000,
        description="Máximo de tokens para o sumário gerado"
    )
```

**Step 4: Rodar o teste e verificar que passa**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/test_config.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add projects/agent/config.py tests/unit/agent/test_config.py
git commit -m "feat(agent): add summarization config fields to AgentSettings"
```

---

### Task 1.5: Integrar Sumarização no Grafo — Nó `load_memory`

**Files:**
- Create: `projects/agent/orchestrator/nodes/load_memory.py`
- Modify: `projects/agent/orchestrator/state.py` (adicionar campo `conversation_summary`)
- Modify: `projects/agent/orchestrator/graph.py` (inserir nó)
- Test: `tests/unit/agent/orchestrator/test_load_memory.py`

**Step 1: Adicionar campo ao estado**

Em `projects/agent/orchestrator/state.py`, adicionar ao `OrchestratorState`:

```python
    # Memória
    conversation_summary: Optional[str]
```

E ao `create_initial_orchestrator_state`:

```python
    conversation_summary=None,
```

**Step 2: Escrever os testes do nó**

```python
"""Testes para o nó load_memory."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from projects.agent.orchestrator.nodes.load_memory import load_memory


@pytest.mark.asyncio
class TestLoadMemory:
    async def test_loads_existing_summary(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "conversation_summary": None,
        }

        mock_service = AsyncMock()
        mock_service.load_summary.return_value = "Resumo anterior da conversa."

        with patch(
            "projects.agent.orchestrator.nodes.load_memory.SummarizationService",
            return_value=mock_service,
        ):
            with patch(
                "projects.agent.orchestrator.nodes.load_memory.get_agent_settings",
                return_value=MagicMock(summarization_enabled=True),
            ):
                result = await load_memory(state)

        assert result["conversation_summary"] == "Resumo anterior da conversa."
        # Deve injetar SystemMessage com o sumário
        injected = [m for m in result["messages"] if isinstance(m, SystemMessage)]
        assert any("Resumo anterior" in m.content for m in injected)

    async def test_no_summary_when_disabled(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "conversation_summary": None,
        }

        with patch(
            "projects.agent.orchestrator.nodes.load_memory.get_agent_settings",
            return_value=MagicMock(summarization_enabled=False),
        ):
            result = await load_memory(state)

        assert result.get("conversation_summary") is None

    async def test_no_summary_when_none_exists(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "conversation_summary": None,
        }

        mock_service = AsyncMock()
        mock_service.load_summary.return_value = None

        with patch(
            "projects.agent.orchestrator.nodes.load_memory.SummarizationService",
            return_value=mock_service,
        ):
            with patch(
                "projects.agent.orchestrator.nodes.load_memory.get_agent_settings",
                return_value=MagicMock(summarization_enabled=True),
            ):
                result = await load_memory(state)

        assert result.get("conversation_summary") is None
```

**Step 3: Implementar o nó**

```python
"""Nó load_memory do Orchestrator.

Carrega memória de longo prazo (sumários, entidades, contexto cross-thread)
e injeta no estado antes do processamento da requisição.
"""
from langchain_core.messages import SystemMessage

from projects.agent.config import get_agent_settings
from projects.agent.orchestrator.state import OrchestratorState
from shared.core.logging import get_logger

logger = get_logger("orchestrator.load_memory")


async def load_memory(state: OrchestratorState) -> dict:
    """Carrega memória persistente e injeta no estado.

    Atualmente carrega:
    - Sumário de conversas anteriores (Phase 1)

    Args:
        state: Estado atual do orchestrator.

    Returns:
        Atualizações para o estado.
    """
    settings = get_agent_settings()
    updates: dict = {}

    if not settings.summarization_enabled:
        return updates

    thread_id = state.get("thread_id")
    if not thread_id:
        return updates

    try:
        from projects.agent.memory.summarization import SummarizationService

        service = SummarizationService()
        summary = await service.load_summary(thread_id)

        if summary:
            logger.info("Sumário carregado", thread_id=thread_id)
            updates["conversation_summary"] = summary

            # Injetar como SystemMessage para contextualizar o LLM
            summary_msg = SystemMessage(
                content=(
                    f"## Contexto de conversas anteriores nesta thread\n\n"
                    f"{summary}\n\n"
                    f"Use este contexto para dar respostas mais relevantes."
                )
            )
            updates["messages"] = [summary_msg]

    except Exception as e:
        logger.warning("Falha ao carregar memória", error=str(e))

    return updates
```

**Step 4: Modificar o grafo para inserir o nó**

Em `projects/agent/orchestrator/graph.py`, adicionar:

```python
from projects.agent.orchestrator.nodes.load_memory import load_memory
```

E no `build_orchestrator_graph()`, adicionar o nó e ajustar as edges:

```python
    # Adicionar nó de memória
    graph.add_node("load_memory", load_memory)

    # Ajustar fluxo: START → load_memory → parse_request
    graph.add_edge(START, "load_memory")
    graph.add_edge("load_memory", "parse_request")
    # Remover: graph.add_edge(START, "parse_request")
```

**Step 5: Rodar os testes**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/orchestrator/test_load_memory.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add projects/agent/orchestrator/nodes/load_memory.py \
      projects/agent/orchestrator/state.py \
      projects/agent/orchestrator/graph.py \
      projects/agent/orchestrator/nodes/__init__.py \
      tests/unit/agent/orchestrator/test_load_memory.py
git commit -m "feat(agent): integrate load_memory node into orchestrator graph"
```

---

### Task 1.6: Nó `persist_memory` — Salvar Sumário Pós-Síntese

**Files:**
- Create: `projects/agent/orchestrator/nodes/persist_memory.py`
- Modify: `projects/agent/orchestrator/graph.py` (inserir nó após synthesize)
- Test: `tests/unit/agent/orchestrator/test_persist_memory.py`

**Step 1: Escrever os testes**

```python
"""Testes para o nó persist_memory."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import HumanMessage, AIMessage

from projects.agent.orchestrator.nodes.persist_memory import persist_memory


@pytest.mark.asyncio
class TestPersistMemory:
    async def test_summarizes_long_conversations(self):
        messages = [HumanMessage(content=f"msg {i}") for i in range(25)]

        state = {
            "messages": messages,
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "synthesized_response": "Resposta final.",
            "conversation_summary": None,
        }

        mock_service = AsyncMock()
        mock_service.get_or_create_summary.return_value = ("Resumo novo.", messages[-10:])

        with patch(
            "projects.agent.orchestrator.nodes.persist_memory.SummarizationService",
            return_value=mock_service,
        ):
            with patch(
                "projects.agent.orchestrator.nodes.persist_memory.get_agent_settings",
                return_value=MagicMock(
                    summarization_enabled=True,
                    summarization_threshold=20,
                    summarization_keep_recent=10,
                ),
            ):
                result = await persist_memory(state)

        mock_service.get_or_create_summary.assert_called_once()

    async def test_skips_when_disabled(self):
        state = {
            "messages": [HumanMessage(content="oi")],
            "thread_id": "t-1",
            "user_id": 1,
            "config_id": 1,
            "synthesized_response": "Resposta.",
            "conversation_summary": None,
        }

        with patch(
            "projects.agent.orchestrator.nodes.persist_memory.get_agent_settings",
            return_value=MagicMock(summarization_enabled=False),
        ):
            result = await persist_memory(state)

        assert result == {}
```

**Step 2: Implementar o nó**

```python
"""Nó persist_memory do Orchestrator.

Persiste memória de longo prazo após a síntese (sumários, entidades, embeddings).
"""
from projects.agent.config import get_agent_settings
from projects.agent.orchestrator.state import OrchestratorState
from shared.core.logging import get_logger

logger = get_logger("orchestrator.persist_memory")


async def persist_memory(state: OrchestratorState) -> dict:
    """Persiste memória após a síntese da resposta.

    Atualmente persiste:
    - Sumário de conversas longas (Phase 1)

    Args:
        state: Estado atual do orchestrator.

    Returns:
        Atualizações para o estado (vazio se nada persistido).
    """
    settings = get_agent_settings()

    if not settings.summarization_enabled:
        return {}

    thread_id = state.get("thread_id")
    if not thread_id:
        return {}

    messages = list(state.get("messages", []))

    try:
        from projects.agent.memory.summarization import SummarizationService

        service = SummarizationService()
        summary, _ = await service.get_or_create_summary(
            thread_id=thread_id,
            user_id=state.get("user_id", 0),
            config_id=state.get("config_id", 0),
            messages=messages,
            threshold=settings.summarization_threshold,
            keep_recent=settings.summarization_keep_recent,
        )

        if summary:
            logger.info("Sumário persistido", thread_id=thread_id)

    except Exception as e:
        logger.warning("Falha ao persistir memória", error=str(e))

    return {}
```

**Step 3: Integrar no grafo**

Em `projects/agent/orchestrator/graph.py`:

```python
from projects.agent.orchestrator.nodes.persist_memory import persist_memory

# No build_orchestrator_graph():
graph.add_node("persist_memory", persist_memory)

# Ajustar: synthesize → persist_memory → END
graph.add_edge("synthesize", "persist_memory")
graph.add_edge("persist_memory", END)
# Remover: graph.add_edge("synthesize", END)
```

**Step 4: Rodar os testes**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/orchestrator/test_persist_memory.py tests/unit/agent/orchestrator/test_load_memory.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add projects/agent/orchestrator/nodes/persist_memory.py \
      projects/agent/orchestrator/graph.py \
      tests/unit/agent/orchestrator/test_persist_memory.py
git commit -m "feat(agent): add persist_memory node for post-synthesis summarization"
```

---

## Phase 2: Vector Store / RAG

### Objetivo
Armazenar embeddings de mensagens e sumários para permitir busca semântica no histórico. Isso habilita o agente a encontrar conversas passadas relevantes.

### Estratégia
- **Extensão:** pgvector no PostgreSQL
- **Embeddings:** OpenAI `text-embedding-3-small` (1536 dims) ou fallback local
- **Armazenamento:** Tabela `agent_memory_embeddings` com coluna `vector`
- **Busca:** Cosine similarity para top-K resultados relevantes
- **Injeção:** Resultados relevantes adicionados ao contexto no nó `load_memory`

---

### Task 2.1: Habilitar pgvector no PostgreSQL

**Files:**
- Create: `alembic/versions/014_enable_pgvector.py`

**Step 1: Escrever a migração**

```python
"""Habilita extensão pgvector e cria tabela de embeddings."""

from alembic import op
import sqlalchemy as sa

revision = "014"
down_revision = "013"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Habilitar extensão pgvector
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "agent_memory_embeddings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, nullable=False, index=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column("thread_id", sa.String(255), nullable=False, index=True),
        sa.Column(
            "source_type",
            sa.String(50),
            nullable=False,
            comment="message | summary | entity",
        ),
        sa.Column("source_id", sa.String(255), nullable=True),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("metadata_", sa.JSON, nullable=True),
        # pgvector: 1536 dimensões (text-embedding-3-small)
        sa.Column("embedding", sa.LargeBinary, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )

    # Criar coluna vector nativa via raw SQL (pgvector)
    op.execute(
        "ALTER TABLE agent_memory_embeddings "
        "ADD COLUMN IF NOT EXISTS embedding_vector vector(1536)"
    )

    # Índice HNSW para busca rápida de similaridade
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_memory_embeddings_vector "
        "ON agent_memory_embeddings USING hnsw (embedding_vector vector_cosine_ops)"
    )

    op.create_index(
        "idx_memory_embeddings_user_type",
        "agent_memory_embeddings",
        ["user_id", "source_type"],
    )


def downgrade() -> None:
    op.drop_table("agent_memory_embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
```

**Step 2: Rodar a migração**

Run: `docker compose exec marketing-agent alembic upgrade head`
Expected: pgvector habilitado, tabela criada

**Step 3: Commit**

```bash
git add alembic/versions/014_enable_pgvector.py
git commit -m "feat(agent): enable pgvector and create embeddings table"
```

---

### Task 2.2: Modelo SQLAlchemy — `MemoryEmbedding`

**Files:**
- Modify: `projects/agent/db/models.py`
- Test: `tests/unit/agent/test_embedding_model.py`

**Step 1: Escrever o teste**

```python
"""Testes para o modelo MemoryEmbedding."""
import pytest
from projects.agent.db.models import MemoryEmbedding


class TestMemoryEmbeddingModel:
    def test_create_embedding(self):
        emb = MemoryEmbedding(
            user_id=1,
            config_id=1,
            thread_id="t-1",
            source_type="message",
            content="Como estão minhas campanhas?",
        )
        assert emb.source_type == "message"
        assert emb.thread_id == "t-1"

    def test_tablename(self):
        assert MemoryEmbedding.__tablename__ == "agent_memory_embeddings"
```

**Step 2: Implementar o modelo**

Adicionar ao `projects/agent/db/models.py`:

```python
class MemoryEmbedding(Base):
    """
    Embeddings vetoriais para busca semântica na memória do agente.
    Armazena representações vetoriais de mensagens, sumários e entidades.
    """
    __tablename__ = "agent_memory_embeddings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    thread_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata_", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_memory_embeddings_user_type", "user_id", "source_type"),
    )
```

**Step 3: Rodar o teste**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/test_embedding_model.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add projects/agent/db/models.py tests/unit/agent/test_embedding_model.py
git commit -m "feat(agent): add MemoryEmbedding model for vector store"
```

---

### Task 2.3: Serviço de Embeddings — `EmbeddingService`

**Files:**
- Create: `projects/agent/memory/embeddings.py`
- Test: `tests/unit/agent/memory/test_embeddings.py`

**Step 1: Escrever os testes**

```python
"""Testes para EmbeddingService."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from projects.agent.memory.embeddings import EmbeddingService


@pytest.mark.asyncio
class TestEmbeddingService:
    async def test_generate_embedding(self):
        mock_embeddings = MagicMock()
        mock_embeddings.embed_query.return_value = [0.1] * 1536

        service = EmbeddingService()
        with patch.object(service, "_get_embeddings_model", return_value=mock_embeddings):
            vector = await service.generate_embedding("Como estão minhas campanhas?")

        assert len(vector) == 1536
        assert vector[0] == 0.1

    async def test_search_similar_returns_results(self):
        service = EmbeddingService()

        mock_results = [
            {"content": "Conversa sobre CPL", "similarity": 0.92, "thread_id": "t-1"},
            {"content": "Análise de performance", "similarity": 0.85, "thread_id": "t-2"},
        ]

        with patch.object(service, "_search_by_vector", return_value=mock_results):
            with patch.object(service, "generate_embedding", return_value=[0.1] * 1536):
                results = await service.search_similar(
                    query="CPL das campanhas",
                    user_id=1,
                    limit=5,
                )

        assert len(results) == 2
        assert results[0]["similarity"] > results[1]["similarity"]
```

**Step 2: Implementar o serviço**

```python
"""
Serviço de embeddings para busca semântica na memória do agente.

Gera e armazena embeddings vetoriais de mensagens e sumários,
permitindo busca por similaridade semântica no histórico.
"""
import json
from typing import Optional

from sqlalchemy import text

from projects.agent.config import get_agent_settings
from projects.agent.db.models import MemoryEmbedding
from shared.db.session import async_session_maker
from shared.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Serviço para geração e busca de embeddings vetoriais."""

    def _get_embeddings_model(self):
        """Retorna o modelo de embeddings configurado."""
        settings = get_agent_settings()

        if settings.llm_provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=settings.openai_api_key,
            )
        else:
            # Fallback: usar OpenAI embeddings mesmo com Anthropic como LLM
            from langchain_openai import OpenAIEmbeddings

            if settings.openai_api_key:
                return OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=settings.openai_api_key,
                )
            raise ValueError(
                "OpenAI API key necessária para embeddings. "
                "Configure AGENT_OPENAI_API_KEY."
            )

    async def generate_embedding(self, text: str) -> list[float]:
        """Gera embedding vetorial para um texto.

        Args:
            text: Texto para gerar embedding.

        Returns:
            Lista de floats representando o vetor.
        """
        model = self._get_embeddings_model()
        return model.embed_query(text)

    async def store_embedding(
        self,
        user_id: int,
        config_id: int,
        thread_id: str,
        content: str,
        source_type: str,
        source_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Gera e armazena embedding no banco.

        Args:
            user_id: ID do usuário.
            config_id: ID da configuração.
            thread_id: ID da thread.
            content: Texto original.
            source_type: Tipo (message, summary, entity).
            source_id: ID da fonte original (opcional).
            metadata: Metadados adicionais (opcional).
        """
        vector = await self.generate_embedding(content)
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        async with async_session_maker() as session:
            record = MemoryEmbedding(
                user_id=user_id,
                config_id=config_id,
                thread_id=thread_id,
                source_type=source_type,
                source_id=source_id,
                content=content,
                metadata_=metadata,
            )
            session.add(record)
            await session.flush()

            # Atualizar coluna vector nativa via raw SQL
            await session.execute(
                text(
                    "UPDATE agent_memory_embeddings "
                    "SET embedding_vector = :vec "
                    "WHERE id = :id"
                ),
                {"vec": vector_str, "id": record.id},
            )
            await session.commit()

    async def _search_by_vector(
        self,
        vector: list[float],
        user_id: int,
        limit: int = 5,
        min_similarity: float = 0.7,
        exclude_thread_id: Optional[str] = None,
    ) -> list[dict]:
        """Busca registros similares por vetor.

        Args:
            vector: Vetor de busca.
            user_id: Filtro de usuário.
            limit: Máximo de resultados.
            min_similarity: Similaridade mínima (0-1).
            exclude_thread_id: Thread a excluir (para cross-thread).

        Returns:
            Lista de dicts com content, similarity, thread_id, source_type.
        """
        vector_str = "[" + ",".join(str(v) for v in vector) + "]"

        query = """
            SELECT
                content,
                thread_id,
                source_type,
                metadata_,
                1 - (embedding_vector <=> :vec::vector) AS similarity
            FROM agent_memory_embeddings
            WHERE user_id = :user_id
              AND 1 - (embedding_vector <=> :vec::vector) >= :min_sim
        """
        params = {
            "vec": vector_str,
            "user_id": user_id,
            "min_sim": min_similarity,
        }

        if exclude_thread_id:
            query += " AND thread_id != :exclude_tid"
            params["exclude_tid"] = exclude_thread_id

        query += " ORDER BY similarity DESC LIMIT :lim"
        params["lim"] = limit

        async with async_session_maker() as session:
            result = await session.execute(text(query), params)
            rows = result.fetchall()

        return [
            {
                "content": row[0],
                "thread_id": row[1],
                "source_type": row[2],
                "metadata": row[3],
                "similarity": float(row[4]),
            }
            for row in rows
        ]

    async def search_similar(
        self,
        query: str,
        user_id: int,
        limit: int = 5,
        min_similarity: float = 0.7,
        exclude_thread_id: Optional[str] = None,
    ) -> list[dict]:
        """Busca conteúdos semanticamente similares à query.

        Args:
            query: Texto de busca.
            user_id: ID do usuário.
            limit: Máximo de resultados.
            min_similarity: Similaridade mínima.
            exclude_thread_id: Thread a excluir.

        Returns:
            Lista de resultados com content, similarity, thread_id.
        """
        vector = await self.generate_embedding(query)
        return await self._search_by_vector(
            vector=vector,
            user_id=user_id,
            limit=limit,
            min_similarity=min_similarity,
            exclude_thread_id=exclude_thread_id,
        )
```

**Step 3: Rodar os testes**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/memory/test_embeddings.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add projects/agent/memory/embeddings.py tests/unit/agent/memory/test_embeddings.py
git commit -m "feat(agent): add EmbeddingService for vector store / RAG"
```

---

### Task 2.4: Configurações de Vector Store

**Files:**
- Modify: `projects/agent/config.py`

**Step 1: Adicionar campos ao `AgentSettings`**

```python
    # Vector Store / RAG
    vector_store_enabled: bool = Field(
        default=False,
        description="Habilitar armazenamento e busca vetorial"
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Modelo de embeddings (OpenAI)"
    )
    embedding_dimensions: int = Field(
        default=1536,
        description="Dimensões do vetor de embedding"
    )
    rag_top_k: int = Field(
        default=3,
        description="Número de resultados RAG a injetar no contexto"
    )
    rag_min_similarity: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Similaridade mínima para resultados RAG"
    )
```

**Step 2: Commit**

```bash
git add projects/agent/config.py
git commit -m "feat(agent): add vector store config fields to AgentSettings"
```

---

### Task 2.5: Integrar RAG no nó `load_memory`

**Files:**
- Modify: `projects/agent/orchestrator/nodes/load_memory.py`
- Modify: `projects/agent/orchestrator/state.py` (adicionar `retrieved_context`)
- Test: `tests/unit/agent/orchestrator/test_load_memory.py` (adicionar testes RAG)

**Step 1: Adicionar campo ao estado**

Em `OrchestratorState`:

```python
    retrieved_context: Optional[list[dict]]
```

**Step 2: Estender `load_memory` para buscar contexto RAG**

Adicionar ao final do `load_memory`:

```python
    # Phase 2: Vector Store / RAG
    if settings.vector_store_enabled:
        try:
            from projects.agent.memory.embeddings import EmbeddingService

            embedding_service = EmbeddingService()
            user_message = _get_last_user_message(state.get("messages", []))

            if user_message:
                results = await embedding_service.search_similar(
                    query=user_message,
                    user_id=state.get("user_id", 0),
                    limit=settings.rag_top_k,
                    min_similarity=settings.rag_min_similarity,
                    exclude_thread_id=thread_id,  # Excluir thread atual
                )

                if results:
                    updates["retrieved_context"] = results
                    context_text = "\n".join(
                        f"- [{r['source_type']}] {r['content'][:200]}"
                        for r in results
                    )
                    context_msg = SystemMessage(
                        content=(
                            "## Contexto relevante de conversas anteriores\n\n"
                            f"{context_text}\n\n"
                            "Use este contexto se for relevante para a pergunta atual."
                        )
                    )
                    msgs = updates.get("messages", [])
                    msgs.append(context_msg)
                    updates["messages"] = msgs

        except Exception as e:
            logger.warning("Falha ao buscar contexto RAG", error=str(e))
```

**Step 3: Integrar embedding no `persist_memory`**

Adicionar ao `persist_memory`:

```python
    # Phase 2: Salvar embeddings
    if settings.vector_store_enabled:
        try:
            from projects.agent.memory.embeddings import EmbeddingService

            embedding_service = EmbeddingService()
            synthesized = state.get("synthesized_response", "")

            if synthesized:
                await embedding_service.store_embedding(
                    user_id=state.get("user_id", 0),
                    config_id=state.get("config_id", 0),
                    thread_id=thread_id,
                    content=synthesized[:1000],  # Limitar tamanho
                    source_type="summary",
                    metadata={
                        "intent": state.get("user_intent"),
                        "agents_used": list(state.get("agent_results", {}).keys()),
                    },
                )
        except Exception as e:
            logger.warning("Falha ao salvar embedding", error=str(e))
```

**Step 4: Rodar os testes**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/orchestrator/ -v`
Expected: PASS

**Step 5: Commit**

```bash
git add projects/agent/orchestrator/nodes/load_memory.py \
      projects/agent/orchestrator/nodes/persist_memory.py \
      projects/agent/orchestrator/state.py \
      tests/unit/agent/orchestrator/test_load_memory.py
git commit -m "feat(agent): integrate RAG context retrieval into load_memory"
```

---

## Phase 3: Entity Memory

### Objetivo
Extrair e persistir entidades mencionadas nas conversas: nomes de campanhas, métricas discutidas, preferências do usuário, limiares mencionados. Estas entidades são injetadas no contexto para personalizar respostas.

### Estratégia
- **Extração:** LLM analisa a resposta sintetizada e extrai entidades estruturadas
- **Armazenamento:** Tabela `agent_user_entities`
- **Injeção:** Entidades relevantes do usuário carregadas no `load_memory`

---

### Task 3.1: Migração — Tabela `agent_user_entities`

**Files:**
- Create: `alembic/versions/015_create_user_entities.py`

**Step 1: Escrever a migração**

```python
"""Cria tabela agent_user_entities para Entity Memory."""

from alembic import op
import sqlalchemy as sa

revision = "015"
down_revision = "014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_user_entities",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer, nullable=False, index=True),
        sa.Column("config_id", sa.Integer, nullable=False),
        sa.Column(
            "entity_type",
            sa.String(50),
            nullable=False,
            comment="campaign | metric | preference | threshold | insight",
        ),
        sa.Column("entity_key", sa.String(255), nullable=False),
        sa.Column("entity_value", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, default=0.8),
        sa.Column("source_thread_id", sa.String(255), nullable=True),
        sa.Column("mention_count", sa.Integer, nullable=False, default=1),
        sa.Column(
            "created_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            server_default=sa.text("NOW()"),
            nullable=False,
        ),
    )
    op.create_index(
        "idx_user_entities_user_type",
        "agent_user_entities",
        ["user_id", "entity_type"],
    )
    op.create_index(
        "idx_user_entities_key",
        "agent_user_entities",
        ["user_id", "entity_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("agent_user_entities")
```

**Step 2: Commit**

```bash
git add alembic/versions/015_create_user_entities.py
git commit -m "feat(agent): add migration for user entities table"
```

---

### Task 3.2: Modelo e Serviço — `EntityMemoryService`

**Files:**
- Modify: `projects/agent/db/models.py` (adicionar `UserEntity`)
- Create: `projects/agent/memory/entities.py`
- Test: `tests/unit/agent/memory/test_entities.py`

**Step 1: Adicionar modelo**

```python
class UserEntity(Base):
    """
    Entidades extraídas de conversas para memória estruturada.
    Persiste conhecimento sobre campanhas, métricas, preferências
    e insights do usuário entre conversas.
    """
    __tablename__ = "agent_user_entities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    config_id: Mapped[int] = mapped_column(Integer, nullable=False)
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
    entity_key: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_value: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(sa.Float, nullable=False, default=0.8)
    source_thread_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    mention_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("idx_user_entities_user_type", "user_id", "entity_type"),
        Index("idx_user_entities_key", "user_id", "entity_key", unique=True),
    )
```

**Step 2: Escrever os testes**

```python
"""Testes para EntityMemoryService."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from projects.agent.memory.entities import EntityMemoryService, extract_entities_prompt


class TestExtractEntitiesPrompt:
    def test_includes_response_text(self):
        prompt = extract_entities_prompt("Campanha XYZ tem CPL de R$ 25,00")
        assert "XYZ" in prompt
        assert "CPL" in prompt


@pytest.mark.asyncio
class TestEntityMemoryService:
    async def test_extract_entities_from_response(self):
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content='[{"type":"campaign","key":"Campanha XYZ","value":"CPL R$ 25,00","confidence":0.9}]'
        )

        service = EntityMemoryService()

        with patch("projects.agent.memory.entities.get_llm", return_value=mock_llm):
            entities = await service.extract_entities("Campanha XYZ tem CPL de R$ 25,00")

        assert len(entities) == 1
        assert entities[0]["key"] == "Campanha XYZ"

    async def test_load_user_entities(self):
        service = EntityMemoryService()

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [
            MagicMock(
                entity_type="campaign",
                entity_key="Campanha XYZ",
                entity_value="CPL R$ 25,00",
                mention_count=3,
            )
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        with patch(
            "projects.agent.memory.entities.async_session_maker",
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_session),
                __aexit__=AsyncMock(),
            ),
        ):
            entities = await service.load_user_entities(user_id=1, limit=10)

        assert len(entities) == 1
        assert entities[0]["key"] == "Campanha XYZ"
```

**Step 3: Implementar o serviço**

```python
"""
Serviço de Entity Memory para extração e persistência de entidades.

Extrai entidades estruturadas (campanhas, métricas, preferências) das
respostas do agente e as persiste para enriquecer futuras conversas.
"""
import json
import re
from typing import Optional

from sqlalchemy import select, update

from projects.agent.db.models import UserEntity
from shared.db.session import async_session_maker
from shared.core.logging import get_logger

logger = get_logger(__name__)

VALID_ENTITY_TYPES = {"campaign", "metric", "preference", "threshold", "insight"}


def extract_entities_prompt(response_text: str) -> str:
    """Constrói prompt para extração de entidades."""
    return (
        "Extraia entidades estruturadas do texto abaixo. Retorne APENAS um JSON array.\n\n"
        "Tipos de entidade validos:\n"
        "- campaign: nome ou ID de campanha mencionada\n"
        "- metric: metrica discutida (CPL, CTR, leads, spend, etc) com seu valor\n"
        "- preference: preferencia expressa pelo usuario (ex: 'prefere CPL abaixo de R$30')\n"
        "- threshold: limiar mencionado (ex: 'CPL maximo aceitavel de R$40')\n"
        "- insight: insight ou conclusao importante\n\n"
        f"Texto:\n{response_text[:2000]}\n\n"
        "Formato de resposta (JSON array):\n"
        '[{"type":"campaign","key":"Nome","value":"detalhes","confidence":0.9}]\n\n'
        "Se nao houver entidades, retorne: []"
    )


class EntityMemoryService:
    """Serviço para extração e gerenciamento de entidades."""

    async def extract_entities(self, text: str) -> list[dict]:
        """Extrai entidades de um texto usando LLM.

        Args:
            text: Texto para extrair entidades.

        Returns:
            Lista de entidades extraídas.
        """
        from projects.agent.llm.provider import get_llm

        prompt = extract_entities_prompt(text)
        llm = get_llm(temperature=0.0, max_tokens=500)
        response = await llm.ainvoke(prompt)

        try:
            # Extrair JSON array da resposta
            content = response.content.strip()
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                # Validar tipos
                return [
                    e for e in entities
                    if isinstance(e, dict)
                    and e.get("type") in VALID_ENTITY_TYPES
                    and e.get("key")
                    and e.get("value")
                ]
            return []
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Falha ao parsear entidades", error=str(e))
            return []

    async def save_entities(
        self,
        user_id: int,
        config_id: int,
        thread_id: str,
        entities: list[dict],
    ) -> int:
        """Salva entidades no banco (upsert por user_id + entity_key).

        Args:
            user_id: ID do usuário.
            config_id: ID da configuração.
            thread_id: Thread de origem.
            entities: Entidades extraídas.

        Returns:
            Número de entidades salvas/atualizadas.
        """
        saved = 0
        async with async_session_maker() as session:
            for entity in entities:
                # Verificar se já existe
                result = await session.execute(
                    select(UserEntity).where(
                        UserEntity.user_id == user_id,
                        UserEntity.entity_key == entity["key"],
                    )
                )
                existing = result.scalar_one_or_none()

                if existing:
                    existing.entity_value = entity["value"]
                    existing.confidence = entity.get("confidence", 0.8)
                    existing.mention_count += 1
                    existing.source_thread_id = thread_id
                else:
                    new_entity = UserEntity(
                        user_id=user_id,
                        config_id=config_id,
                        entity_type=entity["type"],
                        entity_key=entity["key"],
                        entity_value=entity["value"],
                        confidence=entity.get("confidence", 0.8),
                        source_thread_id=thread_id,
                        mention_count=1,
                    )
                    session.add(new_entity)

                saved += 1

            await session.commit()

        return saved

    async def load_user_entities(
        self,
        user_id: int,
        entity_type: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict]:
        """Carrega entidades do usuário do banco.

        Args:
            user_id: ID do usuário.
            entity_type: Filtro por tipo (opcional).
            limit: Máximo de resultados.

        Returns:
            Lista de entidades.
        """
        async with async_session_maker() as session:
            query = select(UserEntity).where(
                UserEntity.user_id == user_id
            ).order_by(
                UserEntity.mention_count.desc(),
                UserEntity.updated_at.desc(),
            ).limit(limit)

            if entity_type:
                query = query.where(UserEntity.entity_type == entity_type)

            result = await session.execute(query)
            rows = result.scalars().all()

        return [
            {
                "type": row.entity_type,
                "key": row.entity_key,
                "value": row.entity_value,
                "confidence": row.confidence,
                "mention_count": row.mention_count,
            }
            for row in rows
        ]
```

**Step 4: Rodar os testes**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/memory/test_entities.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add projects/agent/db/models.py projects/agent/memory/entities.py tests/unit/agent/memory/test_entities.py
git commit -m "feat(agent): add EntityMemoryService for structured knowledge extraction"
```

---

### Task 3.3: Configurações e Integração de Entity Memory

**Files:**
- Modify: `projects/agent/config.py`
- Modify: `projects/agent/orchestrator/nodes/load_memory.py`
- Modify: `projects/agent/orchestrator/nodes/persist_memory.py`
- Modify: `projects/agent/orchestrator/state.py`

**Step 1: Adicionar configurações**

```python
    # Entity Memory
    entity_memory_enabled: bool = Field(
        default=False,
        description="Habilitar extração e persistência de entidades"
    )
    entity_max_per_user: int = Field(
        default=50,
        ge=10,
        le=200,
        description="Máximo de entidades por usuário"
    )
```

**Step 2: Adicionar campo ao estado**

Em `OrchestratorState`:

```python
    user_entities: Optional[list[dict]]
```

**Step 3: Estender `load_memory`**

```python
    # Phase 3: Entity Memory
    if settings.entity_memory_enabled:
        try:
            from projects.agent.memory.entities import EntityMemoryService

            entity_service = EntityMemoryService()
            entities = await entity_service.load_user_entities(
                user_id=state.get("user_id", 0),
                limit=settings.entity_max_per_user,
            )

            if entities:
                updates["user_entities"] = entities
                entity_text = "\n".join(
                    f"- [{e['type']}] {e['key']}: {e['value']}"
                    for e in entities[:10]  # Top 10 mais relevantes
                )
                entity_msg = SystemMessage(
                    content=(
                        "## Conhecimento persistente sobre este usuario\n\n"
                        f"{entity_text}\n\n"
                        "Use estas informacoes para personalizar respostas."
                    )
                )
                msgs = updates.get("messages", [])
                msgs.append(entity_msg)
                updates["messages"] = msgs

        except Exception as e:
            logger.warning("Falha ao carregar entidades", error=str(e))
```

**Step 4: Estender `persist_memory`**

```python
    # Phase 3: Extrair e salvar entidades
    if settings.entity_memory_enabled:
        try:
            from projects.agent.memory.entities import EntityMemoryService

            entity_service = EntityMemoryService()
            synthesized = state.get("synthesized_response", "")

            if synthesized:
                entities = await entity_service.extract_entities(synthesized)
                if entities:
                    await entity_service.save_entities(
                        user_id=state.get("user_id", 0),
                        config_id=state.get("config_id", 0),
                        thread_id=thread_id,
                        entities=entities,
                    )
                    logger.info(
                        "Entidades extraidas e salvas",
                        count=len(entities),
                        thread_id=thread_id,
                    )
        except Exception as e:
            logger.warning("Falha ao extrair entidades", error=str(e))
```

**Step 5: Rodar os testes**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/orchestrator/ tests/unit/agent/memory/ -v`
Expected: PASS

**Step 6: Commit**

```bash
git add projects/agent/config.py \
      projects/agent/orchestrator/nodes/load_memory.py \
      projects/agent/orchestrator/nodes/persist_memory.py \
      projects/agent/orchestrator/state.py
git commit -m "feat(agent): integrate entity memory into orchestrator graph"
```

---

## Phase 4: Cross-thread Memory

### Objetivo
Permitir que o agente aprenda entre conversas diferentes do mesmo usuário. Se o usuário discutiu preferências em uma conversa passada, o agente lembra disso em conversas novas.

### Estratégia
- **Já implementado:** Entity Memory (Phase 3) persiste por `user_id`, não por `thread_id`
- **Já implementado:** Vector Store (Phase 2) busca com `exclude_thread_id` para cross-thread
- **Falta:** Serviço unificado de User Memory que combina entidades + RAG cross-thread
- **Falta:** User preferences learning (padrões de interação)

---

### Task 4.1: Serviço de User Memory — `UserMemoryService`

**Files:**
- Create: `projects/agent/memory/user_memory.py`
- Test: `tests/unit/agent/memory/test_user_memory.py`

**Step 1: Escrever os testes**

```python
"""Testes para UserMemoryService."""
import pytest
from unittest.mock import AsyncMock, patch

from projects.agent.memory.user_memory import UserMemoryService


@pytest.mark.asyncio
class TestUserMemoryService:
    async def test_get_user_context_combines_entities_and_rag(self):
        mock_entity_service = AsyncMock()
        mock_entity_service.load_user_entities.return_value = [
            {"type": "preference", "key": "CPL máximo", "value": "R$ 30,00", "confidence": 0.9, "mention_count": 3},
        ]

        mock_embedding_service = AsyncMock()
        mock_embedding_service.search_similar.return_value = [
            {"content": "Campanha XYZ com CPL baixo", "similarity": 0.88, "thread_id": "t-old"},
        ]

        service = UserMemoryService()

        with patch.object(service, "_entity_service", mock_entity_service):
            with patch.object(service, "_embedding_service", mock_embedding_service):
                context = await service.get_user_context(
                    user_id=1,
                    current_query="Qual o CPL das campanhas?",
                    current_thread_id="t-new",
                )

        assert context["entities"] is not None
        assert len(context["entities"]) == 1
        assert context["related_conversations"] is not None
        assert len(context["related_conversations"]) == 1

    async def test_returns_empty_when_no_history(self):
        mock_entity_service = AsyncMock()
        mock_entity_service.load_user_entities.return_value = []

        mock_embedding_service = AsyncMock()
        mock_embedding_service.search_similar.return_value = []

        service = UserMemoryService()

        with patch.object(service, "_entity_service", mock_entity_service):
            with patch.object(service, "_embedding_service", mock_embedding_service):
                context = await service.get_user_context(
                    user_id=1,
                    current_query="Oi",
                    current_thread_id="t-1",
                )

        assert context["entities"] == []
        assert context["related_conversations"] == []
```

**Step 2: Implementar o serviço**

```python
"""
Serviço unificado de User Memory.

Combina Entity Memory e Vector Store para fornecer contexto
cross-thread personalizado por usuário.
"""
from typing import Optional

from projects.agent.memory.entities import EntityMemoryService
from projects.agent.memory.embeddings import EmbeddingService
from projects.agent.config import get_agent_settings
from shared.core.logging import get_logger

logger = get_logger(__name__)


class UserMemoryService:
    """Serviço que unifica todas as camadas de memória do usuário."""

    def __init__(self):
        self._entity_service = EntityMemoryService()
        self._embedding_service = EmbeddingService()

    async def get_user_context(
        self,
        user_id: int,
        current_query: str,
        current_thread_id: str,
    ) -> dict:
        """Obtém contexto completo do usuário para enriquecer a conversa.

        Combina:
        - Entidades persistidas (preferências, campanhas, métricas)
        - Conversas anteriores semanticamente relevantes (RAG cross-thread)

        Args:
            user_id: ID do usuário.
            current_query: Pergunta atual do usuário.
            current_thread_id: Thread atual (excluída da busca).

        Returns:
            Dict com entities e related_conversations.
        """
        settings = get_agent_settings()
        result = {
            "entities": [],
            "related_conversations": [],
        }

        # 1. Carregar entidades do usuário
        if settings.entity_memory_enabled:
            try:
                entities = await self._entity_service.load_user_entities(
                    user_id=user_id,
                    limit=settings.entity_max_per_user,
                )
                result["entities"] = entities
            except Exception as e:
                logger.warning("Falha ao carregar entidades", error=str(e))

        # 2. Buscar conversas relevantes (cross-thread)
        if settings.vector_store_enabled:
            try:
                related = await self._embedding_service.search_similar(
                    query=current_query,
                    user_id=user_id,
                    limit=settings.rag_top_k,
                    min_similarity=settings.rag_min_similarity,
                    exclude_thread_id=current_thread_id,
                )
                result["related_conversations"] = related
            except Exception as e:
                logger.warning("Falha ao buscar conversas", error=str(e))

        return result

    def format_context_for_injection(self, context: dict) -> Optional[str]:
        """Formata o contexto do usuário para injeção como SystemMessage.

        Args:
            context: Dict retornado por get_user_context.

        Returns:
            Texto formatado ou None se não houver contexto.
        """
        parts = []

        entities = context.get("entities", [])
        if entities:
            entity_lines = []
            for e in entities[:10]:
                entity_lines.append(f"- [{e['type']}] {e['key']}: {e['value']}")
            parts.append(
                "### Conhecimento sobre o usuario\n" + "\n".join(entity_lines)
            )

        conversations = context.get("related_conversations", [])
        if conversations:
            conv_lines = []
            for c in conversations[:3]:
                conv_lines.append(
                    f"- (similaridade {c['similarity']:.0%}) {c['content'][:200]}"
                )
            parts.append(
                "### Conversas anteriores relevantes\n" + "\n".join(conv_lines)
            )

        if not parts:
            return None

        return "## Memoria do usuario\n\n" + "\n\n".join(parts)
```

**Step 3: Rodar os testes**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/memory/test_user_memory.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add projects/agent/memory/user_memory.py tests/unit/agent/memory/test_user_memory.py
git commit -m "feat(agent): add UserMemoryService for cross-thread memory"
```

---

### Task 4.2: Refatorar `load_memory` para Usar `UserMemoryService`

**Files:**
- Modify: `projects/agent/orchestrator/nodes/load_memory.py`
- Test: `tests/unit/agent/orchestrator/test_load_memory.py` (atualizar)

**Step 1: Refatorar o nó**

Substituir as seções separadas de entity + RAG por chamada unificada:

```python
async def load_memory(state: OrchestratorState) -> dict:
    """Carrega toda memória persistente e injeta no estado."""
    settings = get_agent_settings()
    updates: dict = {}

    thread_id = state.get("thread_id")
    if not thread_id:
        return updates

    # Phase 1: Sumário de conversa
    if settings.summarization_enabled:
        try:
            from projects.agent.memory.summarization import SummarizationService

            service = SummarizationService()
            summary = await service.load_summary(thread_id)

            if summary:
                updates["conversation_summary"] = summary
                updates["messages"] = [
                    SystemMessage(
                        content=(
                            f"## Contexto desta conversa\n\n{summary}\n\n"
                            "Use este contexto para dar respostas mais relevantes."
                        )
                    )
                ]
        except Exception as e:
            logger.warning("Falha ao carregar sumário", error=str(e))

    # Phases 2+3+4: User Memory (entidades + RAG cross-thread)
    if settings.entity_memory_enabled or settings.vector_store_enabled:
        try:
            from projects.agent.memory.user_memory import UserMemoryService

            user_memory = UserMemoryService()
            user_message = _get_last_user_message(state.get("messages", []))

            if user_message:
                context = await user_memory.get_user_context(
                    user_id=state.get("user_id", 0),
                    current_query=user_message,
                    current_thread_id=thread_id,
                )

                formatted = user_memory.format_context_for_injection(context)
                if formatted:
                    if context.get("entities"):
                        updates["user_entities"] = context["entities"]
                    if context.get("related_conversations"):
                        updates["retrieved_context"] = context["related_conversations"]

                    msgs = updates.get("messages", [])
                    msgs.append(SystemMessage(content=formatted))
                    updates["messages"] = msgs

        except Exception as e:
            logger.warning("Falha ao carregar user memory", error=str(e))

    return updates


def _get_last_user_message(messages) -> Optional[str]:
    """Extrai texto da última mensagem do usuário."""
    from langchain_core.messages import HumanMessage

    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None
```

**Step 2: Rodar os testes completos**

Run: `docker compose exec marketing-agent python -m pytest tests/unit/agent/ -v`
Expected: PASS

**Step 3: Commit**

```bash
git add projects/agent/orchestrator/nodes/load_memory.py \
      tests/unit/agent/orchestrator/test_load_memory.py
git commit -m "refactor(agent): unify load_memory with UserMemoryService"
```

---

### Task 4.3: Configuração de Cross-thread Memory

**Files:**
- Modify: `projects/agent/config.py`

**Step 1: Adicionar campos**

```python
    # Cross-thread Memory
    cross_thread_enabled: bool = Field(
        default=False,
        description="Habilitar memória entre threads diferentes do mesmo usuário"
    )
    cross_thread_max_results: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Máximo de resultados cross-thread a injetar"
    )
```

**Step 2: Commit**

```bash
git add projects/agent/config.py
git commit -m "feat(agent): add cross-thread memory config fields"
```

---

## Task Final: Documentação

**Files:**
- Modify: `docs/agent.md`

Atualizar a documentação com:

1. Diagrama do novo grafo (com `load_memory` e `persist_memory`)
2. Descrição das 4 camadas de memória
3. Configurações disponíveis
4. Como habilitar cada feature (variáveis de ambiente)

```markdown
### Memória Avançada

O agente possui 4 camadas de memória:

| Camada | Config | Descrição |
|--------|--------|-----------|
| Summarization | `AGENT_SUMMARIZATION_ENABLED=true` | Resume conversas longas automaticamente |
| Vector Store | `AGENT_VECTOR_STORE_ENABLED=true` | Busca semântica em histórico (requer pgvector) |
| Entity Memory | `AGENT_ENTITY_MEMORY_ENABLED=true` | Extrai e persiste entidades mencionadas |
| Cross-thread | `AGENT_CROSS_THREAD_ENABLED=true` | Aprendizado entre conversas do mesmo usuário |
```

**Commit final:**

```bash
git add docs/agent.md
git commit -m "docs(agent): add advanced memory architecture documentation"
```

---

## Resumo de Dependências

### Python (requirements.txt)

```
# Adicionar:
pgvector>=0.2.0
langchain-openai>=0.2.0  # já existe, para embeddings
tiktoken>=0.5.0          # já existe, para contagem de tokens
```

### PostgreSQL

```sql
-- Necessário no servidor PostgreSQL:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Variáveis de Ambiente (docker-compose.yml)

```yaml
# Novas variáveis para o serviço marketing-agent:
AGENT_SUMMARIZATION_ENABLED: "true"
AGENT_SUMMARIZATION_THRESHOLD: "20"
AGENT_SUMMARIZATION_KEEP_RECENT: "10"
AGENT_VECTOR_STORE_ENABLED: "false"   # Habilitar após instalar pgvector
AGENT_ENTITY_MEMORY_ENABLED: "false"  # Habilitar gradualmente
AGENT_CROSS_THREAD_ENABLED: "false"   # Habilitar por último
AGENT_RAG_TOP_K: "3"
AGENT_RAG_MIN_SIMILARITY: "0.75"
AGENT_ENTITY_MAX_PER_USER: "50"
```

---

## Ordem de Habilitação em Produção

1. **Deploy Phase 1** (Summarization) — Sem dependências externas, pode habilitar imediatamente
2. **Deploy pgvector** — `CREATE EXTENSION vector` no PostgreSQL
3. **Deploy Phase 2** (Vector Store) — Habilitar `AGENT_VECTOR_STORE_ENABLED=true`
4. **Deploy Phase 3** (Entity Memory) — Habilitar `AGENT_ENTITY_MEMORY_ENABLED=true`
5. **Deploy Phase 4** (Cross-thread) — Habilitar `AGENT_CROSS_THREAD_ENABLED=true` após validar Phases 2+3

Cada fase pode ser habilitada/desabilitada independentemente via feature flags.
