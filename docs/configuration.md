# Configuracao

## Variaveis principais

- `DATABASE_URL`
- `REDIS_URL`
- `ML_API_KEY`
- `JWT_SECRET`
- `AGENT_*` (LLM provider, modelo, timeouts, logging)
- `FACEBOOK_*` (app id/secret, OAuth URLs, encryption key)

As configuracoes sao carregadas por `shared/infrastructure/config/settings.py` e `projects/agent/config.py`.

## Exemplo de .env

Use o `.env` do repositorio como referencia e ajuste para o ambiente atual.
