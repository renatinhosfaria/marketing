# Configuration

## Variaveis principais

- `DATABASE_URL`: conexao PostgreSQL.
- `REDIS_URL`: conexao Redis para broker/cache.
- `ML_API_KEY`: autenticacao de chamadas de API.
- `JWT_SECRET`: assinatura de tokens.
- `FACEBOOK_APP_ID`, `FACEBOOK_APP_SECRET`: integracao OAuth.
- `FACEBOOK_OAUTH_CALLBACK_URL`: callback OAuth backend.
- `FACEBOOK_OAUTH_FRONTEND_REDIRECT_URL`: retorno OAuth frontend.
- `FACEBOOK_TOKEN_ENCRYPTION_KEY`: protecao de tokens sensiveis.
- `LOG_LEVEL`, `ENVIRONMENT`, `DEBUG`: comportamento de runtime.

## Exemplo de .env

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/marketing
REDIS_URL=redis://localhost:6379/0
ML_API_KEY=change-me
JWT_SECRET=change-me
FLOWER_PASSWORD=change-me

FACEBOOK_APP_ID=
FACEBOOK_APP_SECRET=
FACEBOOK_OAUTH_CALLBACK_URL=
FACEBOOK_OAUTH_FRONTEND_REDIRECT_URL=
FACEBOOK_TOKEN_ENCRYPTION_KEY=

NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_APP_NAME=Marketing
NEXT_PUBLIC_FACEBOOK_APP_ID=
```

## Diretrizes

- nunca versionar `.env`;
- separar valores por ambiente (dev/staging/prod);
- rotacionar segredos periodicamente;
- evitar credenciais hardcoded em manifests de deploy.
