"""
Servico de Entity Memory para extracao e persistencia de entidades.

Extrai entidades estruturadas (campanhas, metricas, preferencias) das
respostas do agente e as persiste para enriquecer futuras conversas.
"""
import json
import re
from typing import Optional

from sqlalchemy import select

from projects.agent.db.models import UserEntity
from shared.db.session import async_session_maker
from shared.core.logging import get_logger

logger = get_logger(__name__)

VALID_ENTITY_TYPES = {"campaign", "metric", "preference", "threshold", "insight"}


def extract_entities_prompt(response_text: str) -> str:
    """Constroi prompt para extracao de entidades."""
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
    """Servico para extracao e gerenciamento de entidades."""

    async def extract_entities(self, text: str) -> list[dict]:
        """Extrai entidades de um texto usando LLM.

        Args:
            text: Texto para extrair entidades.

        Returns:
            Lista de entidades extraidas.
        """
        from projects.agent.llm.provider import get_llm

        prompt = extract_entities_prompt(text)
        llm = get_llm(temperature=0.0, max_tokens=500)
        response = await llm.ainvoke(prompt)

        try:
            content = response.content.strip()

            # Tentar parse direto
            try:
                entities = json.loads(content)
            except json.JSONDecodeError:
                # Encontrar primeiro array JSON com brackets balanceados
                start = content.find("[")
                if start == -1:
                    return []
                depth = 0
                end = -1
                for i in range(start, len(content)):
                    if content[i] == "[":
                        depth += 1
                    elif content[i] == "]":
                        depth -= 1
                        if depth == 0:
                            end = i
                            break
                if end == -1:
                    return []
                entities = json.loads(content[start:end + 1])

            if not isinstance(entities, list):
                return []

            return [
                e for e in entities
                if isinstance(e, dict)
                and e.get("type") in VALID_ENTITY_TYPES
                and e.get("key")
                and e.get("value")
            ][:20]
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
            user_id: ID do usuario.
            config_id: ID da configuracao.
            thread_id: Thread de origem.
            entities: Entidades extraidas.

        Returns:
            Numero de entidades salvas/atualizadas.
        """
        saved = 0
        async with async_session_maker() as session:
            for entity in entities:
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
        """Carrega entidades do usuario do banco.

        Args:
            user_id: ID do usuario.
            entity_type: Filtro por tipo (opcional).
            limit: Maximo de resultados.

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
