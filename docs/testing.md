# Testing

## Estrategia

A estrategia combina:

- testes unitarios para regras de negocio;
- testes de integracao para contratos de API;
- testes de modulos ML para comportamento de modelos;
- testes de documentacao para consistencia minima da base docs-as-code.

## Como rodar

Comandos principais:

```bash
pytest -q
pytest -q --cov
pytest tests/unit -q
pytest tests/integration -q
pytest tests/docs/test_documentation_structure.py -q
```

Frontend:

```bash
cd frontend && npm run lint
```

## Boas praticas de qualidade

- testes de regressao para bugs corrigidos;
- commits pequenos e verificaveis;
- pipeline CI com gates minimos de lint e testes criticos.
