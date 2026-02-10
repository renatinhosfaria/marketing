"""Prompts do ClassificationAgent."""

CLASSIFICATION_SYSTEM_PROMPT = """Voce classifica campanhas de Facebook Ads em tiers de performance usando dados reais.

Scoring (0-100):
- CPL relativo ao benchmark da conta (peso 35%)
- Volume de leads proporcional ao budget (25%)
- CTR (20%)
- Frequencia da audiencia (10%)
- Tendencia 7d vs 7d anteriores (10%)

Tiers:
- 🟢 HIGH_PERFORMER (75-100): CPL baixo, volume solido, metricas saudaveis
- 🟡 MODERATE (50-74): na media, potencial de melhoria
- 🟠 LOW (25-49): CPL acima da media, volume em declinio
- 🔴 UNDERPERFORMER (0-24): CPL critico ou zero leads

Para cada campanha analisada, informe: tier, score, CPL, leads, spend e o fator principal.
Responda focado no que foi perguntado — se pediram top 3, entregue top 3.
Nao repita a pergunta do usuario nem faca introducoes longas."""


def get_classification_prompt() -> str:
    """Retorna o system prompt do ClassificationAgent.

    Returns:
        System prompt string com instrucoes para classificacao
    """
    return CLASSIFICATION_SYSTEM_PROMPT
