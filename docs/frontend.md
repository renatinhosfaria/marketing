# Frontend

## Stack e organizacao

Frontend em Next.js (App Router) com React e TypeScript.

Estrutura relevante:

- `frontend/app`: rotas e paginas.
- `frontend/components`: componentes de UI e layout.
- `frontend/lib`: cliente HTTP e utilitarios.
- `frontend/hooks`: hooks de integracao de estado e rede.

## Scripts

Scripts principais em `frontend/package.json`:

- `npm run dev`: servidor local na porta 3001.
- `npm run build`: build de producao com output standalone.
- `npm run start`: executa artefato standalone na porta 3001.
- `npm run lint`: validacao estatica via ESLint.

## Variaveis de ambiente

Variaveis mais importantes:

- `NEXT_PUBLIC_API_URL`: base URL consumida no browser.
- `NEXT_PUBLIC_APP_NAME`: nome exibido na aplicacao.
- `NEXT_PUBLIC_FACEBOOK_APP_ID`: client id para fluxos de OAuth no frontend.
- `API_INTERNAL_URL`: destino interno para rewrites de `/api/*`.
- `FB_ADS_INTERNAL_URL`: destino interno para rewrites de `/api/v1/facebook-ads/*`.

## Boas praticas

- manter componentes por dominio funcional;
- evitar logica de negocio no layer de apresentacao;
- encapsular chamadas HTTP em funcoes de `frontend/lib`;
- padronizar tratamento de erro para UX consistente.
