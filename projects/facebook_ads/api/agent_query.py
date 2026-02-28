"""Endpoint de consulta SQL para agente FB Ads."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from shared.db.session import get_db
from projects.facebook_ads.schemas.agent_query import AgentQueryRequest
from projects.facebook_ads.services.agent_query_service import AgentQueryService

router = APIRouter()


@router.post("/query")
async def run_agent_query(
    request: AgentQueryRequest,
    db: AsyncSession = Depends(get_db),
):
    """Executa query do agente com guardrails e auditoria."""
    service = AgentQueryService(db)
    try:
        result = await service.execute_sql(prompt=request.prompt, sql=request.sql)
        return {"success": True, "data": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
